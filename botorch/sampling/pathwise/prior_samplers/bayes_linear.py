#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Union

from botorch.models import ModelListGP
from botorch.sampling.pathwise.basis import (
    fourier_feature_initializer,
    GeneralizedLinearBasis,
    KernelFeatureInitializer,
)
from botorch.sampling.pathwise.paths import AffinePath, PathList
from botorch.sampling.pathwise.utils import (
    DEFAULT,
    get_default_transforms,
    TensorTransform,
    TransformedModule,
)
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.sampling import draw_sobol_normal_samples
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.models import ApproximateGP, ExactGP, GP
from torch import Size, Tensor
from torch.nn import Module

NoneType = type(None)
DrawBayesLinearPaths = Dispatcher("draw_bayes_linear_paths")


def draw_bayes_linear_paths(
    model: GP, sample_shape: Size, **kwargs: Any
) -> Union[AffinePath, TransformedModule]:
    r"""Draws functions a Bayesian linear model approximation to `model`."""
    return DrawBayesLinearPaths(model, sample_shape=sample_shape, **kwargs)


def _draw_bayes_linear_paths_fallback(
    mean_module: Optional[Module],
    covar_module: Kernel,
    sample_shape: Size,
    batch_shape: Size,
    num_bases: int = 1024,
    num_outputs: int = 1,
    weight_sampler: Optional[Callable[[Size], Tensor]] = None,
    initializer: KernelFeatureInitializer = fourier_feature_initializer,
) -> TransformedModule:
    # Extract device and dtype from reference tensor
    if isinstance(covar_module, ScaleKernel):
        reference = covar_module.raw_outputscale
    elif hasattr(covar_module, "lengthscale"):
        reference = covar_module.lengthscale
    else:
        raise NotImplementedError("Failed to extract reference tensor.")

    basis = GeneralizedLinearBasis(
        initializer=partial(initializer, kernel=covar_module),
        output_shape=Size([num_outputs, num_bases]),  # TODO: Test sharing
        batch_shape=batch_shape,
    )
    if weight_sampler is None:
        weight = draw_sobol_normal_samples(
            n=sample_shape.numel() * batch_shape.numel() * num_outputs,
            d=num_bases,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(sample_shape + basis.batch_shape + basis.output_shape)
    else:
        weight = weight_sampler(
            sample_shape + basis.batch_shape + basis.output_shape
        ).to(device=reference.device, dtype=reference.dtype)

    return AffinePath(basis=basis, weight=weight, bias_module=mean_module)


@DrawBayesLinearPaths.register(ExactGP)
def _draw_bayes_linear_paths_exactGP(
    model: ExactGP,
    *,
    sample_shape: Size,
    weight_sampler: Optional[Callable[[Size, Any], Tensor]] = None,
    input_transform: Union[TensorTransform, NoneType] = DEFAULT,
    output_transform: Union[TensorTransform, NoneType] = DEFAULT,
    num_outputs: Optional[int] = None,
    **kwargs: Any,
) -> Union[AffinePath, TransformedModule]:
    if len(model.train_inputs) != 1:
        raise NotImplementedError

    if num_outputs is None:  # TODO: Improve me.
        num_outputs = getattr(model, "num_outputs", 1)

    paths = _draw_bayes_linear_paths_fallback(
        mean_module=model.mean_module,
        covar_module=model.covar_module,
        sample_shape=sample_shape,
        batch_shape=model.batch_shape,
        weight_sampler=weight_sampler,
        **kwargs,
    )

    input_transform, output_transform = get_default_transforms(
        model=model, input_transform=input_transform, output_transform=output_transform
    )
    if input_transform is None and output_transform is None:
        return paths

    return TransformedModule(
        base_module=paths,
        arg_transforms=[] if input_transform is None else [input_transform],
        ret_transforms=[] if output_transform is None else [output_transform],
    )


@DrawBayesLinearPaths.register(ModelListGP)
def _draw_bayes_linear_paths_listGP(
    model: ModelListGP, *, sample_shape: Size, **kwargs: Any
) -> PathList:
    return PathList(
        draw_bayes_linear_paths(m, sample_shape=sample_shape, **kwargs)
        for m in model.models
    )


@DrawBayesLinearPaths.register(ApproximateGP)
def _draw_bayes_linear_paths_approxGP(
    model: ApproximateGP, *, sample_shape: Size, **kwargs: Any
) -> Union[AffinePath, TransformedModule]:
    return DrawBayesLinearPaths(
        model,
        model.variational_strategy,
        model.variational_strategy._variational_distribution,
        sample_shape=sample_shape,
        **kwargs,
    )


@DrawBayesLinearPaths.register(ApproximateGP, object, object)
def _draw_bayes_linear_paths_approxGP_fallback(
    model: ApproximateGP,
    _: object,
    __: object,
    *,
    sample_shape: Size,
    weight_sampler: Optional[Callable[[Size, Any], Tensor]] = None,
    input_transform: Optional[TensorTransform] = None,
    output_transform: Optional[TensorTransform] = None,
    **kwargs: Any,
):
    paths = _draw_bayes_linear_paths_fallback(
        mean_module=model.mean_module,
        covar_module=model.covar_module,
        sample_shape=sample_shape,
        batch_shape=model.covar_module.batch_shape,  # TODO: Improve me!
        weight_sampler=weight_sampler,
        **kwargs,
    )

    if input_transform is None and output_transform is None:
        return paths

    return TransformedModule(
        base_module=paths,
        arg_transforms=[] if input_transform is None else [input_transform],
        ret_transforms=[] if output_transform is None else [output_transform],
    )
