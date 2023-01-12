#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch
from botorch.sampling.pathwise.basis.initializers import GeneralizedLinearInitializer
from botorch.sampling.pathwise.utils.common import TensorTransform
from gpytorch.kernels import Kernel
from torch import Size, Tensor
from torch.nn import Module, Parameter


class BasisExpansion(Module):
    output_shape: Size
    batch_shape: Optional[Size]


class GeneralizedLinearBasis(BasisExpansion):
    r"""Generalized linear basis functions:
    `phi(x) = output_transform(input_transform(x)^T @ weight + bias)`."""

    input_transform: Optional[TensorTransform]
    output_transform: Optional[TensorTransform]

    def __init__(
        self,
        initializer: GeneralizedLinearInitializer,
        output_shape: Size,
        batch_shape: Optional[Size] = None,
    ) -> None:
        super().__init__()
        self.batch_shape = Size() if batch_shape is None else batch_shape
        self.output_shape = output_shape
        self.initializer = initializer
        self._initialized = Parameter(
            torch.zeros([], dtype=torch.bool), requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self._initialized:
            self.initialize(input_shape=x.shape[-1:])

        x = x if self.input_transform is None else self.input_transform(x)
        z = x @ self.weight.transpose(-2, -1)
        if self.bias is not None:
            z = z + self.bias

        return z if self.output_transform is None else self.output_transform(z)

    def initialize(self, input_shape: Size) -> None:
        weight, bias, input_transform, output_transform = self.initializer(
            input_shape=input_shape,
            output_shape=self.batch_shape + self.output_shape,
        )
        self.bias = bias
        self.weight = weight
        self.input_transform = input_transform
        self.output_transform = output_transform
        self._initialized[...] = True


class KernelBasis(BasisExpansion):
    def __init__(self, kernel: Kernel, centers: Tensor) -> None:
        r"""Canonical basis functions $\phi_{i}(x) = k(x, z_{i})."""
        try:
            torch.broadcast_shapes(centers.shape[:-2], kernel.batch_shape)
        except RuntimeError as e:
            raise RuntimeError(
                f"Shape mismatch: `centers` has shape {centers.shape}, "
                f"but kernel.batch_shape={kernel.batch_shape}."
            ) from e

        super().__init__()
        self.kernel = kernel
        self.centers = centers

    def forward(self, x: Tensor) -> Tensor:
        return self.kernel(x, self.centers)

    @property
    def batch_shape(self) -> Size:
        return self.kernel.batch_shape

    @property
    def output_shape(self) -> Size:
        return self.centers.shape[-2:-1]
