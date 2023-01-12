#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Union

from botorch.models import ModelListGP
from botorch.sampling.pathwise.paths import CompositePath, PathList, SamplePath
from botorch.sampling.pathwise.prior_samplers import draw_bayes_linear_paths
from botorch.sampling.pathwise.update_strategies import exact_update
from botorch.sampling.pathwise.utils import (
    DEFAULT,
    get_default_transforms,
    TransformedModule,
)
from botorch.utils.dispatcher import Dispatcher
from gpytorch.models import ApproximateGP, ExactGP, GP
from torch import Size, Tensor
from torch.nn import Module

NoneType = type(None)
DrawMatheronPaths = Dispatcher("draw_matheron_paths")


class MatheronPath(CompositePath):
    r"""Represents function draws from a Gaussian process posterior via Matheron's rule:

                   Prior path
                       v
        (f | Y)(...) = f(...) + Cov(f(...), Y) Cov(Y, Y)^{-1} (Y - f(X)),
                               |________________________________________|
                                                   v
                                              Update path

    where `=` represents equality in distribution and `Y = {f(xi) + ei, for all xi ∈ X}`
    are the observed values of `f(X) = {f(x1), ..., f(xn)}` corrupted by independent
    realizations of noise variable `e ~ N(0, s^2)`.
    """

    def __init__(
        self,
        prior_paths: SamplePath,
        update_paths: SamplePath,
        join_rule: Callable[[Iterable[Tensor]], Tensor] = sum,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            paths={"prior_paths": prior_paths, "update_paths": update_paths},
            join_rule=join_rule,
            **kwargs,
        )

    @property
    def prior_paths(self) -> SamplePath:
        return self["prior_paths"]

    @property
    def update_paths(self) -> SamplePath:
        return self["update_paths"]


def draw_matheron_paths(
    model: GP,
    sample_shape: Size,
    prior_sampler: Callable = draw_bayes_linear_paths,
    update_strategy: Callable = exact_update,
    prior_kwargs: Optional[Dict[str, Any]] = None,
    update_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[MatheronPath, TransformedModule]:
    return DrawMatheronPaths(
        model,
        sample_shape=sample_shape,
        prior_sampler=prior_sampler,
        update_strategy=update_strategy,
        prior_kwargs=prior_kwargs,
        update_kwargs=update_kwargs,
        **kwargs,
    )


@DrawMatheronPaths.register(ModelListGP)
def _draw_matheron_paths_listGP(model: ModelListGP, **kwargs: Any):
    return PathList([draw_matheron_paths(m, **kwargs) for m in model.models])


@DrawMatheronPaths.register(ExactGP)
def _draw_matheron_paths_exactGP(
    model: ExactGP,
    *,
    sample_shape: Size,
    prior_sampler: Callable,
    update_strategy: Callable,
    prior_kwargs: Optional[Dict[str, Any]] = None,
    update_kwargs: Optional[Dict[str, Any]] = None,
    input_transform: Union[Module, NoneType] = DEFAULT,
    output_transform: Union[Module, NoneType] = DEFAULT,
) -> Union[MatheronPath, TransformedModule]:
    input_transform, output_transform = get_default_transforms(
        model=model, input_transform=input_transform, output_transform=output_transform
    )

    prior_paths = prior_sampler(
        model=model,
        sample_shape=sample_shape,
        output_transform=None,
        **(prior_kwargs or {}),
    )

    # `train_inputs`` and `train_targets` are assumed pre-transformed
    train_targets = model.train_targets
    prior_latents = (
        prior_paths.base_module(*model.train_inputs)
        if isinstance(prior_paths, TransformedModule)
        else prior_paths(*model.train_inputs)
    )

    update_paths = update_strategy(
        model=model,
        prior_latents=prior_latents,
        train_targets=train_targets,
        output_transform=None,
        **(update_kwargs or {}),
    )

    matheron_paths = MatheronPath(prior_paths=prior_paths, update_paths=update_paths)
    if output_transform is None:
        return matheron_paths

    return TransformedModule(matheron_paths, ret_transforms=[output_transform])


@DrawMatheronPaths.register(ApproximateGP)
def _draw_matheron_paths_approxGP(
    model: ApproximateGP,
    *,
    sample_shape: Size,
    prior_sampler: Callable,
    update_strategy: Callable,
    prior_kwargs: Optional[Dict[str, Any]] = None,
    update_kwargs: Optional[Dict[str, Any]] = None,
    input_transform: Union[Module, NoneType] = DEFAULT,
    output_transform: Union[Module, NoneType] = DEFAULT,
) -> Union[MatheronPath, TransformedModule]:
    input_transform, output_transform = get_default_transforms(
        model=model, input_transform=input_transform, output_transform=output_transform
    )

    prior_paths = prior_sampler(
        model=model,
        sample_shape=sample_shape,
        output_transform=None,
        **(prior_kwargs or {}),
    )

    # Inducing points are assume to be pre-transformed
    prior_latents = (
        prior_paths.base_module(model.variational_strategy.inducing_points)
        if isinstance(prior_paths, TransformedModule)
        else prior_paths(model.variational_strategy.inducing_points)
    )

    update_paths = update_strategy(
        model=model,
        prior_latents=prior_latents,
        output_transform=None,
        **(update_kwargs or {}),
    )

    matheron_paths = MatheronPath(prior_paths=prior_paths, update_paths=update_paths)
    if output_transform is None:
        return matheron_paths

    return TransformedModule(matheron_paths, ret_transforms=[output_transform])
