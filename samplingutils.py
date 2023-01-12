import torch
import gpytorch
from botorch.models.gp_regression import SingleTaskGP
from botorch.sampling.pathwise.prior_samplers import draw_bayes_linear_paths



def draw_quad_kernel_prior_paths(quad_kernel, n_samples): #quad_kernel is a scaled polynomial(power=2) kernel
    
    c = quad_kernel.offset
    ws = torch.randn(size=[n_samples,1,3])
    
    
    def paths(xs):
        
        if len(xs.shape) == 2 and xs.shape[1] == 1: #xs must be n_samples x npoints x 1 dim
            xs = xs.repeat(n_samples,1,1) #duplicate over batch (sample) dim
            
        X = torch.concat([xs * xs, (2 * c).sqrt() * xs, c.expand(*xs.shape)], dim=2)
        W = ws.repeat(1,xs.shape[1],1)     #ws is n_samples x 1 x 3 dim

        phis = W*X
        return torch.sum(phis, dim=-1) #result tensor is shape n_samples x npoints
    
    return paths
    
    
def draw_product_kernel_prior_paths(model, n_samples):
    
    ndim = model.train_inputs[0].shape[1]
    
    matern_covar_module = model.covar_module.base_kernel.kernels[0] #expects ProductKernel (Matern x Polynomial(dim=2))
    matern_covar_module = gpytorch.kernels.ScaleKernel(matern_covar_module)
    matern_covar_module.outputscale = model.covar_module.outputscale.detach()

    mean_module = gpytorch.means.ZeroMean()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = model.likelihood.noise.detach()
    
    outcome_transform = None
    input_transform = None

    ########################################
    #build zero-mean (ndim-1)-dimensional GP called matern_model 
    #with kernel matched to the Matern component of the passed model

        
    matern_model = SingleTaskGP(train_X = torch.tensor([[0.]*(ndim-1)]), 
                                train_Y = torch.tensor([[0.]]), 
                                likelihood = likelihood, 
                                mean_module = mean_module,
                                covar_module = matern_covar_module, 
                                 outcome_transform = outcome_transform, 
                                input_transform = input_transform
                               )

    
    ########################################
    
    matern_prior_paths = draw_bayes_linear_paths(
        model=matern_model,
        sample_shape=torch.Size([n_samples]),
        output_transform=None
    )
    
    
    
    quad_kernel = model.covar_module.base_kernel.kernels[1]


    quad_prior_paths = draw_quad_kernel_prior_paths(quad_kernel, n_samples)
    
    def product_kernel_prior_paths(xs):
        return (matern_prior_paths(xs[:,:-1].float()).reshape(n_samples,-1) * quad_prior_paths(xs[:,-1:].float())).double()
    
    return product_kernel_prior_paths


def draw_product_kernel_post_paths(model, n_samples, cpu=True):

    product_kernel_prior_paths = draw_product_kernel_prior_paths(model, n_samples=n_samples)
    
    train_x = model.train_inputs[0]
#     if model.input_transform is not None:
#         train_x = model.input_transform.untransform(train_x)
    
    train_y = model.train_targets.reshape(-1,1)
#     if model.outcome_transform is not None:
#         train_y = model.outcome_transform.untransform(train_y)[0]
        
    train_y = train_y - model.mean_module(train_x).reshape(train_y.shape)
    
#     Knn = model.covar_module.forward(train_x, train_x) #remove forward
    Knn = model.covar_module(train_x, train_x)
    
    sigma = torch.sqrt(model.likelihood.noise[0])

    K = Knn + sigma**2*torch.eye(Knn.shape[0])

    prior_residual = train_y.repeat(n_samples,1,1).reshape(n_samples,-1) - product_kernel_prior_paths(train_x)
    prior_residual -= sigma*torch.randn_like(prior_residual)

    # v = K.inv_matmul(prior_residual)
    v = torch.linalg.solve(torch.block_diag(*[K.to_dense()]*n_samples), prior_residual.reshape(-1,1)) #replace with cholesky approach
    v = v.reshape(n_samples,-1,1)

    def post_paths(xs):
        if model.input_transform is not None:
            xs = model.input_transform(xs)
        
#         K_update = model.covar_module.forward(train_x, xs.double()) #remove forward
        K_update = model.covar_module(train_x, xs.double()) #remove forward

        v_t = v.transpose(1,2)

        update = torch.matmul(v_t, K_update)
        update = update.reshape(n_samples,-1)
        
        prior = product_kernel_prior_paths(xs)
        
        post = prior + update + (model.mean_module(xs).reshape(1,-1)).repeat(n_samples,1)
        if model.outcome_transform is not None:
            post = model.outcome_transform.untransform(post)[0]
            
        return post
    
    return post_paths