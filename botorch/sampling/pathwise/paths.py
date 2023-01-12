#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Union

from botorch.sampling.pathwise.basis import BasisExpansion
from torch import stack, Tensor
from torch.nn import Module, ModuleDict, ModuleList, Parameter


class SamplePath(Module):
    pass


class AffinePath(SamplePath):
    def __init__(
        self,
        basis: BasisExpansion,
        weight: Union[Parameter, Tensor],
        bias_module: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()
        self.basis = basis
        self.weight = weight
        self.bias_module = bias_module

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        features = self.basis(x, **kwargs)
        outputs = features @ self.weight.unsqueeze(-1)
        if self.bias_module is not None:
            outputs = outputs + self.bias_module(x).unsqueeze(-1)

        if outputs.shape[-1] == 1:
            outputs = outputs.squeeze(dim=-1)

        return outputs


class PathList(SamplePath):
    def __init__(
        self,
        paths: Union[Iterable[SamplePath], ModuleList],
        stack_dim: int = -1,
        **kwargs: Any,
    ) -> None:
        if "input_transform" in kwargs or "output_transform" in kwargs:
            raise NotImplementedError

        if not isinstance(paths, ModuleList):
            paths = ModuleList(paths)

        super().__init__()
        self.add_module("paths", paths)
        self.stack_dim = stack_dim

    def __getitem__(self, key: Hashable) -> SamplePath:
        return self.paths[key]

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        names = None
        samples = []
        for path in self.paths:
            values = path(x, **kwargs)
            if names is None:
                names = values.names
            elif values.names != names:
                raise ValueError
            samples.append(values.rename(None))
        return stack(samples, dim=self.stack_dim).refine_names(..., *names)


class CompositePath(SamplePath):
    def __init__(
        self,
        paths: Mapping[str, SamplePath],
        join_rule: Callable[[Iterable[Tensor]], Tensor],
    ) -> None:
        if not isinstance(paths, ModuleDict):
            paths = ModuleDict(paths)

        super().__init__()
        self.paths = paths if isinstance(paths, ModuleDict) else ModuleDict(paths)
        self.join_rule = join_rule

    def __getitem__(self, key: Hashable) -> SamplePath:
        return self.paths[key]

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return self.join_rule(path(x, **kwargs) for path in self.paths.values())
