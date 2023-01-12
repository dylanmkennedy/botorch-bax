#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
.. [rahimi2007random]
    A. Rahimi and B. Recht. Random features for large-scale kernel machines.
    Advances in neural information processing systems 20 (2007).

.. [sutherland2015error]
    D. J. Sutherland and J. Schneider. On the error of random Fourier features.
    arXiv preprint arXiv:1506.02785 (2015).
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Protocol, runtime_checkable

import torch
from botorch.sampling.pathwise.utils.common import TensorTransform
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.sampling import draw_sobol_normal_samples
from gpytorch import kernels
from torch import Size, Tensor
from torch.distributions import Gamma

FourierFeatureInitializer = Dispatcher("fourier_feature_initializer")
NoneType = type(None)


class GeneralizedLinearInitialization(NamedTuple):
    r"""Initialization for  generalized linear basis functions `phi(x)`. Formally:
    `phi(x) = output_transform(input_transform(x)^T @ weight + bias)`.
    """
    weight: Tensor
    bias: Optional[Tensor]
    input_transform: Optional[TensorTransform]
    output_transform: Optional[TensorTransform]


@runtime_checkable
class GeneralizedLinearInitializer(Protocol):
    def __call__(
        self, input_shape: Size, output_shape: Size, **kwargs: Any
    ) -> GeneralizedLinearInitialization:
        pass


@runtime_checkable
class KernelFeatureInitializer(Protocol):
    def __call__(
        self,
        kernel: kernels.Kernel,
        input_shape: Size,
        output_shape: Size,
        **kwargs: Any,
    ) -> GeneralizedLinearInitialization:
        pass


def fourier_feature_initializer(
    kernel: kernels.Kernel,
    input_shape: Size,
    output_shape: Size,
    **kwargs: Any,
) -> GeneralizedLinearInitialization:
    return FourierFeatureInitializer(
        kernel, input_shape=input_shape, output_shape=output_shape, **kwargs
    )


def _fourier_initializer_stationary_sincos(
    kernel: kernels.Kernel,
    weight_initializer: Callable[[Size], Tensor],
    input_shape: Size,
    output_shape: Size,
) -> GeneralizedLinearInitialization:
    r"""Returns a (2 * l)-dimensional feature map `phi: X -> R^{2l}` whose inner product
    phi(x)^T phi(x') approximates the evaluation of a stationary kernel
    `k(x, x') = k(x - x')`. For details, see [rahimi2007random]_.

    As argued for in [sutherland2015error]_, we use Euler's formula to represent
    complex exponential basis functions as pairs of trigonometric bases:

        `phi_{i}(x) = sin(x^T w_{i})` and `phi_{i + l} = cos(x^T w_{i})`
    """
    assert (output_shape[-1] % 2) == 0
    shape = output_shape[:-1] + (output_shape[-1] // 2,)
    scale = (2 / output_shape[-1]) ** 0.5
    weight = weight_initializer(Size([shape.numel(), input_shape.numel()])).reshape(
        *shape, *input_shape
    )

    def input_transform(raw_inputs: Tensor) -> Tensor:
        return kernel.lengthscale.reciprocal() * raw_inputs

    def output_transform(raw_features: Tensor) -> Tensor:
        return scale * torch.concat([raw_features.sin(), raw_features.cos()], dim=-1)

    return GeneralizedLinearInitialization(
        weight, None, input_transform, output_transform
    )


@FourierFeatureInitializer.register(kernels.RBFKernel)
def _fourier_initializer_rbf(
    kernel: kernels.RBFKernel,
    *,
    input_shape: Size,
    output_shape: Size,
) -> GeneralizedLinearInitialization:
    def _weight_initializer(shape: Size) -> Tensor:
        if len(shape) != 2:
            raise NotImplementedError

        return draw_sobol_normal_samples(
            n=shape[0],
            d=shape[1],
            device=kernel.lengthscale.device,
            dtype=kernel.lengthscale.dtype,
        )

    return _fourier_initializer_stationary_sincos(
        kernel=kernel,
        weight_initializer=_weight_initializer,
        input_shape=input_shape,
        output_shape=output_shape,
    )


@FourierFeatureInitializer.register(kernels.MaternKernel)
def _fourier_initializer_matern(
    kernel: kernels.MaternKernel,
    *,
    input_shape: Size,
    output_shape: Size,
) -> GeneralizedLinearInitialization:
    def _weight_initializer(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise NotImplementedError(
                f"Expected `shape` to be size 2, but is size {len(shape)}."
            )

        dtype = kernel.lengthscale.dtype
        device = kernel.lengthscale.device
        nu = torch.tensor(kernel.nu, device=device, dtype=dtype)
        normals = draw_sobol_normal_samples(n=n, d=d, device=device, dtype=dtype)
        return Gamma(nu, nu).rsample((n, 1)).rsqrt() * normals

    return _fourier_initializer_stationary_sincos(
        kernel=kernel,
        weight_initializer=_weight_initializer,
        input_shape=input_shape,
        output_shape=output_shape,
    )


@FourierFeatureInitializer.register(kernels.ScaleKernel)
def _fourier_initializer_scale(
    kernel: kernels.ScaleKernel,
    *,
    input_shape: Size,
    output_shape: Size,
) -> GeneralizedLinearInitialization:

    weight, bias, input_transform, output_transform = fourier_feature_initializer(
        kernel.base_kernel,
        input_shape=input_shape,
        output_shape=output_shape,
    )

    def scaled_output_transform(raw_features: Tensor) -> Tensor:
        features = (
            raw_features if output_transform is None else output_transform(raw_features)
        )
        outputscale = kernel.outputscale
        while outputscale.ndim < features.ndim:
            outputscale = outputscale.unsqueeze(-1)

        return outputscale.sqrt() * features

    return GeneralizedLinearInitialization(
        weight, bias, input_transform, scaled_output_transform
    )
