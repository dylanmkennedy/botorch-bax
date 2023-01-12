#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import zip_longest
from typing import Any, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import torch
from botorch.sampling.pathwise.utils.common import TensorTransform
from torch import Tensor
from torch.nn import Module

T = TypeVar("T")
MaybeTuple = Union[T, Tuple[T]]


class TransformedModule(Module):
    base_module: Module
    arg_transforms: Sequence[TensorTransform]
    ret_transforms: Sequence[TensorTransform]
    attr_transforms: Mapping[str, TensorTransform]
    _initialized: bool = False

    def __init__(
        self,
        base_module: Module,
        arg_transforms: Optional[Sequence[TensorTransform]] = None,
        ret_transforms: Optional[Sequence[TensorTransform]] = None,
        attr_transforms: Optional[Mapping[str, TensorTransform]] = None,
    ) -> None:
        r"""Helper class for transforming a base module's attributes and/or the
        positional arguments and return values of it's forward method.

        Args:
            base_module: The wrapped module.
            arg_transforms: Transforms for base_module.forward's positional arguments.
            ret_transforms: Transforms for base_module.forward's return values.
            attr_transforms: Transforms for base_module tensor-typed attributes.

        TODO: Determine whether or not to keep `attr_transforms`.
        """
        super().__init__()
        self.base_module = base_module
        self.arg_transforms = [] if arg_transforms is None else arg_transforms
        self.ret_transforms = [] if ret_transforms is None else ret_transforms
        self.attr_transforms = {} if attr_transforms is None else attr_transforms
        self._initialized = True  # used to control __setattr__ behavior

    def forward(self, *args: Tensor, **kwargs: Any) -> MaybeTuple[Tensor]:
        if self.arg_transforms:
            args = (func(obj) for func, obj in zip_longest(self.arg_transforms, args))

        # TODO: We need to alias the base module's hook so that they are run
        # by instance.__call__.
        rets = type(self.base_module).forward(self, *args, **kwargs)
        if not self.ret_transforms:
            return rets

        if torch.is_tensor(rets):
            if len(self.ret_transforms) != 1:
                raise RuntimeError
            return self.ret_transforms[0](rets)

        return tuple(func(obj) for func, obj in zip_longest(self.ret_transforms, rets))

    def __getattr__(self, key: Any) -> Any:
        try:
            return super().__getattr__(key)
        except AttributeError:
            obj = getattr(self.base_module, key)
            if key in self.attr_transforms:
                return self.attr_transforms[key](obj)
            return obj

    def __setattr__(self, key: Any, obj: Any) -> Any:
        if self._initialized:
            setattr(self.base_module, key, obj)
        else:  # initialize the module regularly
            super().__setattr__(key, obj)

    def train(self: T, mode: bool = True) -> T:
        super().__setattr__("training", mode)
        super().__getattr__("base_module").train(mode=mode)
        return self
