
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from algorithms import Algorithm

class ExpectedInformationGain(AnalyticAcquisitionFunction):
    r"""Single outcome expected information gain`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EIG = ExpectedInformationGain(model, algo, algo_params)
        >>> eig = EIG(test_X)
    """

    def __init__(
        self,
        model: Model,
        algo: type[Algorithm]
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            maximize: If True, consider the problem a maximization problem.
        """

        super().__init__(model=model)
        self.algo = algo
        self.xs_exe, self.ys_exe = self.algo.get_exe_paths(self.model) #get sampled execution paths and set self.xs_exe, self.ys_exe
            
        self.model(self.xs_exe) #call model on some data to avoid errors. we don't need this result.
        
    #     construct a batch of size n_samples fantasy models, 
    #     where each fantasy model is produced by taking the model at the current iteration and conditioning it on 
    #     one of the sampled execution path subsequences:
        xs_exe_transformed = self.model.input_transform(self.xs_exe)
        ys_exe_transformed = self.model.outcome_transform(self.ys_exe)[0]
        self.fmodels = self.model.condition_on_observations(xs_exe_transformed, ys_exe_transformed)    #xs_exe.shape = (n_samples, len_exe_path, ndim)
                                                                                    #ys_exe.shape = (n_samples, len_exe_path, 1)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Information Gain on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Information Gain is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Information Gain values at the
            given design points `X`.
        """



    #     Use the current model and fantasy models to compute a Monte-Carlo estimate of the Expected Information Gain:
    #     (see https://arxiv.org/pdf/2104.09460.pdf: eq (4) and the last sentence of page 7)
    
        #################################################################
        #calculcate the variance of the posterior at this iteration for each input x
        p = self.model.posterior(X)
        var_p = p.variance
        var_p = var_p.reshape(var_p.shape[:-1])

        #calculcate the variance of the fantasy posteriors
        pfs = self.fmodels.posterior(( X.reshape(*X.shape[:-2], 1, *X.shape[-2:]).expand(*X.shape[:-2], self.algo.n_samples, *X.shape[-2:]) ))
        var_pfs = pfs.variance
        var_pfs = var_pfs.reshape(var_pfs.shape[:-1])

        ##################################################################
        
        #calculate Shannon entropy for posterior given the current data
        h_current = 0.5 * torch.log(2*torch.pi * var_p) + 0.5

        #calculate the Shannon entropy for the fantasy posteriors
        h_fantasies = 0.5 * torch.log(2*torch.pi * var_pfs) + 0.5

        #compute the Monte-Carlo estimate of the Expected value of the entropy of the fantasy posteriors
        avg_h_fantasy = torch.mean(h_fantasies, dim=-2)

        #use the above entropies to compute the Expected Information Gain, 
        #where the terms in the equation below correspond to the terms in eq(4) of https://arxiv.org/pdf/2104.09460.pdf
        #(Note, again, that avg_h_fantasy is a Monte-Carlo estimate of the second term on the right)
        eig = h_current - avg_h_fantasy

        return eig.reshape(X.shape[:-2])
    
    
 
    def cuda(self):
        
        self.xs_exe, self.ys_exe = self.xs_exe.to('cuda'), self.ys_exe.to('cuda')
        self.model = self.model.cuda()

        xs_exe_transformed = self.model.input_transform(self.xs_exe)
        ys_exe_transformed = self.model.outcome_transform(self.ys_exe)[0]
        self.fmodels = self.model.condition_on_observations(xs_exe_transformed, ys_exe_transformed)
        
    def cpu(self):
        
        self.xs_exe, self.ys_exe = self.xs_exe.cpu(), self.ys_exe.cpu()
        self.model = self.model.cpu()

        xs_exe_transformed = self.model.input_transform(self.xs_exe)
        ys_exe_transformed = self.model.outcome_transform(self.ys_exe)[0]
        self.fmodels = self.model.condition_on_observations(xs_exe_transformed, ys_exe_transformed)
        