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
    x_samples = ( torch.rand(n_samples,ndim)  * torch.tensor([bounds[1] - bounds[0] for bounds in domain])
              + torch.tensor([bounds[0] for bounds in domain])
                ) #uniform sample, rescaled, and shifted to cover the domain
    
    return x_samples

def build_mesh_domain(n_steps_sample_grid, domain):
    
    ndim = domain.shape[0]
    
    try: #check to see if n_steps_sample_grid is a list
        n_steps_sample_grid = list(n_steps_sample_grid)
    except: 
        #in case n_steps_sample_grid is an integer rather than a list, we make it a list with that integer repeated for every dimension
        n_steps_sample_grid = [n_steps_sample_grid]*int(ndim)

    if len(n_steps_sample_grid) != ndim:
        raise ValueError("If n_steps_sample_grid is a list, it must have length = ndim")
        
    linspace_list = [torch.linspace(bounds[0], bounds[1], n_steps) 
                     for n_steps, bounds in zip(n_steps_sample_grid, domain)]

    x_mesh_tuple = torch.meshgrid(*linspace_list, indexing='ij')

    x_mesh_columnized_tuple = tuple(x_mesh.reshape(-1,1) for x_mesh in x_mesh_tuple)


    if ndim == 1:
        xs_n_by_d = x_mesh_columnized_tuple[0]
    else:
        xs_n_by_d = torch.cat(x_mesh_columnized_tuple, dim=1)


    return xs_n_by_d, x_mesh_tuple
        
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