import torch
import gpytorch
from botorch.models.gp_regression import SingleTaskGP


def build_parabolic_target_func(x_star):
    def paraboloid(x):
        z = 1 - torch.sum((x-x_star)**2, dim=1)
        z += torch.randn_like(z) * 0.05
        z = z.unsqueeze(-1)
        return z
    return paraboloid

def unif_random_sample_domain(n_samples, domain):
    ndim = len(domain)
    #Observe target function n_obs_init times using a uniform sample of the domain
    x_samples = ( torch.rand(n_samples,ndim).double()  * torch.tensor([bounds[1] - bounds[0] for bounds in domain])
              + torch.tensor([bounds[0] for bounds in domain])
                ) #uniform sample, rescaled, and shifted to cover the domain
    
    return x_samples

def fit_gp_model(train_x, train_z):
    
    noise_prior = gpytorch.priors.GammaPrior(1,10)
    noise_constraint = gpytorch.constraints.Interval(0.01,0.1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior, noise_constraint=noise_constraint)
    model = SingleTaskGP(train_x, train_z, likelihood)

    # Find optimal model hyperparameters
    model.train()
    model.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_z).sum()
        loss.backward()
#             if (i+1) % 20 == 0:
#                 print('Iter %d/%d - Loss: %.3f ' % (
#                     i + 1, training_iter, loss.item(),
#                 ))
        optimizer.step()

    model.eval()

    return model