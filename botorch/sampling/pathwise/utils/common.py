#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor


class _DefaultType(type):
    pass


DEFAULT = _DefaultType("DEFAULT", (), {})
NoneType = type(None)
TensorTransform = Callable[[Tensor], Tensor]


def get_default_transforms(
    model: GPyTorchModel,
    input_transform: Union[TensorTransform, NoneType] = DEFAULT,
    output_transform: Union[TensorTransform, NoneType] = DEFAULT,
) -> Tuple[Optional[TensorTransform], Optional[TensorTransform]]:
    r"""Helper method for optionally obtaining tensor transforms from a module."""

    if input_transform is DEFAULT:
        input_transform = getattr(model, "input_transform", None)

    if output_transform is DEFAULT:
        if hasattr(model, "output_transform"):
            if hasattr(model, "outcome_transform"):
                raise NotImplementedError
            output_transform = model.output_transform

        elif hasattr(model, "outcome_transform"):

            def output_transform(raw_features: Tensor) -> Tensor:
                if model.num_outputs > 1:
                    # OutcomeTransforms expect multi-output dimension to be last
                    features, _ = model.outcome_transform.untransform(
                        raw_features.transpose(-2, -1)
                    )
                    return features.transpose(-2, -1)

                features, _ = model.outcome_transform.untransform(raw_features)
                return features

        else:
            output_transform = None

    return input_transform, output_transform
