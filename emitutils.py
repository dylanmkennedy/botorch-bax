import torch
import gpytorch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize, InputStandardize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll

import torch.nn
from torch.nn.functional import mse_loss
from gpytorch.kernels import RBFKernel, PolynomialKernel


def beam_size_squared(k, d, l, s11, s12, s22):
    return (
        (1.0 + k * d * l) ** 2 * s11 + 2.0 * (1.0 + d * l * k) * d * s12 + d ** 2 * s22
    )
    
def toy_beam_size_squared_nd(x):
    distance = torch.tensor(1.0).double()
    q_len = torch.tensor(0.1).double()
    s11 = torch.tensor(3e-6).double()
    s12 = torch.tensor(1.5e-6).double()
    s22 = torch.tensor(2e-6).double()
    emit = torch.sqrt(s11 * s22 - s12 ** 2)
#     print(emit)
    bss = ((1 + torch.sum(x[:,:-1]**2, dim=1) )* beam_size_squared(x[:,-1], distance, q_len, s11, s12, s22)).reshape(-1,1) 
#     return  ( 1 + .05*torch.rand_like(bss) ) * bss 
    return bss 

    

def fit_gp_model_emittance(train_x, train_z):

########################################
    #covar
    
    covar_module = ( RBFKernel(active_dims=list(range(train_x.shape[1]-1))) * 
                    PolynomialKernel(power = 2, active_dims=[train_x.shape[1]-1] )
                   )
    scaled_covar_module = gpytorch.kernels.ScaleKernel(covar_module)

########################################
    #mean
    
#     constant_constraint = gpytorch.constraints.GreaterThan(50.)
#     constant_constraint = gpytorch.constraints.Positive()
#     constant_prior = gpytorch.priors.GammaPrior(10,10)
    constant_constraint = None
    constant_prior = None
    mean_module = gpytorch.means.ConstantMean(constant_prior = constant_prior, constant_constraint=constant_constraint)
#     mean_module = None
########################################
    #noise/likelihood
    
    noise_prior = gpytorch.priors.GammaPrior(1,10)
#     noise_prior = None
    noise_constraint = None
#     noise_constraint = gpytorch.constraints.GreaterThan(0.01)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior, noise_constraint=noise_constraint)
    
########################################
    #transforms
    
    outcome_transform = Standardize(m=1)
    input_transform = InputStandardize(d=train_x.shape[-1])  
#     input_transform = Normalize(d=train_x.shape[-1])  
#     outcome_transform = None
#     input_transform = None

########################################
    #model
    
    model = SingleTaskGP(train_x, train_z, likelihood, mean_module = mean_module, covar_module = scaled_covar_module, 
                         outcome_transform = outcome_transform, input_transform = input_transform)

########################################
    # Find optimal model hyperparameters
    
    model.train()
    model.likelihood.train()



    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    


    fit_gpytorch_mll(mll)
    


    model.eval()

    return model