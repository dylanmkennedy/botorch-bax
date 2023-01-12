#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from botorch.models.transforms.input import InputTransform
from botorch.sampling.pathwise.basis import KernelBasis
from botorch.sampling.pathwise.paths import AffinePath
from botorch.sampling.pathwise.utils import DEFAULT, TensorTransform, TransformedModule
from botorch.utils.dispatcher import Dispatcher
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ApproximateGP, ExactGP, GP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from linear_operator.operators import DiagLinearOperator, SumLinearOperator
from torch import Tensor

NoneType = type(None)
ExactUpdate = Dispatcher("exact_update")


def exact_update(
    model: GP,
    prior_latents: Tensor,
    train_targets: Optional[Tensor] = DEFAULT,
    train_inputs: Optional[Tuple[Tensor, ...]] = DEFAULT,
    **kwargs: Any,
) -> AffinePath:
    return ExactUpdate(
        model,
        prior_latents=prior_latents,
        train_targets=train_targets,
        train_inputs=train_inputs,
        **kwargs,
    )


def _exact_update_fallback(
    kernel: Kernel,
    train_inputs: Tuple[Tensor, ...],
    train_targets: Tensor,
    prior_latents: Tensor,
    scale_tril: Optional[Tensor] = None,
    noise_variance: Optional[Tensor] = None,
    input_transform: Optional[TensorTransform] = None,
) -> AffinePath:
    if len(train_inputs) != 1:
        raise not NotImplementedError

    centers = next(iter(train_inputs))  # assumed pre-transformed
    if noise_variance is None:
        prior_targets = prior_latents
        if scale_tril is None:
            scale_tril = kernel(centers).cholesky().to_dense()
    else:
        prior_targets = prior_latents + noise_variance.sqrt() * torch.randn_like(
            prior_latents
        )
        if scale_tril is None:
            scale_tril = (
                SumLinearOperator(kernel(centers), DiagLinearOperator(noise_variance))
                .cholesky()
                .to_dense()
            )

    basis = KernelBasis(kernel=kernel, centers=centers)
    if input_transform is not None:
        basis = TransformedModule(base_module=basis, arg_transforms=[input_transform])

    weight = torch.cholesky_solve(
        (train_targets - prior_targets).unsqueeze(-1), scale_tril
    )
    return AffinePath(basis=basis, weight=weight.squeeze(-1))


@ExactUpdate.register(ExactGP)
def _exact_update_exactGP(
    model: ExactGP,
    *,
    prior_latents: Tensor,
    train_targets: Optional[Tensor] = DEFAULT,
    train_inputs: Optional[Tuple[Tensor, ...]] = DEFAULT,
    noise_variance: Optional[Tensor] = DEFAULT,
    input_transform: Optional[InputTransform] = DEFAULT,
    **ignore: Any,
) -> AffinePath:
    if not isinstance(model.likelihood, (_GaussianLikelihoodBase, NoneType)):
        raise NotImplementedError

    if input_transform is DEFAULT:
        input_transform = getattr(model, "input_transform", None)

    if train_inputs is DEFAULT:
        train_inputs = model.train_inputs

    if train_targets is DEFAULT:
        train_targets = model.train_targets

    if noise_variance is DEFAULT:
        noise_variance = model.likelihood.noise_covar(shape=train_inputs[0].shape[:-1])
        noise_variance = noise_variance.diagonal(dim1=-2, dim2=-1)

    return _exact_update_fallback(
        kernel=model.covar_module,
        train_inputs=train_inputs,
        train_targets=train_targets,
        prior_latents=prior_latents,
        noise_variance=noise_variance,
        input_transform=input_transform,
    )


@ExactUpdate.register(ApproximateGP)
def _exact_update_approxGP(
    model: ApproximateGP,
    *,
    prior_latents: Tensor,
    **kwargs: Any,
) -> AffinePath:
    return ExactUpdate(
        model,
        model.variational_strategy,
        model.variational_strategy._variational_distribution,
        prior_latents=prior_latents,
        **kwargs,
    )


@ExactUpdate.register(
    ApproximateGP, VariationalStrategy, CholeskyVariationalDistribution
)
def _exact_update_vGeneric_qCholesky(
    model: ApproximateGP,
    _: VariationalStrategy,
    __: CholeskyVariationalDistribution,
    *,
    prior_latents: Tensor,
    noise_variance: Optional[Tensor] = DEFAULT,
    input_transform: Optional[InputTransform] = DEFAULT,
    **ignore: Any,
) -> AffinePath:
    # TODO: Noise latent values to account for jitter added by `psd_safe_cholesky`
    if not (noise_variance is DEFAULT or noise_variance is None):
        raise NotImplementedError

    if input_transform is DEFAULT:
        input_transform = getattr(model, "input_transform", None)

    # Inducing points are assumed to live in transformed space
    inducing_points = model.variational_strategy.inducing_points
    scale_tril = model.variational_strategy._cholesky_factor(
        model.variational_strategy(inducing_points, prior=True).lazy_covariance_matrix
    ).to_dense()

    # Sample latent inducing values f(Z) = u
    batch_shape = model.covar_module.batch_shape  # TODO: Improve me!
    inducing_latents = model.variational_strategy.variational_distribution.rsample(
        prior_latents.shape[: prior_latents.ndim - len(batch_shape) - 1]
    ) @ scale_tril.transpose(-1, -2)

    return _exact_update_fallback(
        kernel=model.covar_module,
        train_inputs=[inducing_points],
        train_targets=inducing_latents,
        prior_latents=prior_latents,
        scale_tril=scale_tril,
        input_transform=input_transform,
    )
