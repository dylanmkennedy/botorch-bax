import torch
import gpytorch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize, InputStandardize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll

from gpytorch.kernels import MaternKernel, PolynomialKernel


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

def toy_emit_nd(x):
    distance = torch.tensor(1.0).double()
    q_len = torch.tensor(0.1).double()
    s11 = torch.tensor(3e-6).double()
    s12 = torch.tensor(1.5e-6).double()
    s22 = torch.tensor(2e-6).double()
    emit = torch.sqrt(s11 * s22 - s12 ** 2)
    return (1 + torch.sum(x**2, dim=1) ) * emit


def fit_gp_model_emittance(train_x, train_z):

########################################
    #covar
    
    covar_module = ( MaternKernel(active_dims=list(range(train_x.shape[1]-1))) * 
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




import torch.nn as nn

class EmittanceModule(nn.Module):
    def __init__(self, post_paths, n_samples, X_meas, samplewise=True, squared=True):
        super().__init__()
        self.post_paths = post_paths
        self.n_samples = n_samples
        self.X_meas = X_meas
        self.samplewise = samplewise
        self.squared = squared

    def forward(self, x):
        
        #need to wrap for botorch batching
        print(x.shape)
        return sum_samplewise_emittance_flat_X_botorch(self.post_paths, self.n_samples, x, self.X_meas)
    
    
    
    
    
def sum_samplewise_emittance_flat_X_wrapper_for_scipy(post_paths, n_samples, X_meas):
    
    def wrapped_func(X_tuning_flat):
        
        return sum_samplewise_emittance_flat_X(post_paths, n_samples, torch.tensor(X_tuning_flat), X_meas).detach().cpu().numpy()
    
    return wrapped_func

def sum_samplewise_emittance_flat_X_wrapper_for_torch(post_paths, n_samples, X_meas):
    
    def wrapped_func(X_tuning_flat):
        return sum_samplewise_emittance_flat_X(post_paths, n_samples, X_tuning_flat, X_meas)
    
    return wrapped_func

def sum_samplewise_emittance_flat_X_botorch(post_paths, n_samples, X_tuning_input, X_meas):
   
    X_tuning = X_tuning_input.double().reshape(*X_tuning_input.shape[:-2],n_samples,-1)
    
    return torch.sum(post_emit_batched_input(post_paths, X_tuning, X_meas, samplewise=True, squared=True))

def sum_samplewise_emittance_flat_X(post_paths, n_samples, X_tuning_flat, X_meas):
    
    X_tuning = X_tuning_flat.double().reshape(n_samples,-1)
    
    return torch.sum(post_emit(post_paths, X_tuning, X_meas, samplewise=True, squared=True))

def post_emit_batched_input(post_paths, X_tuning, X_meas, samplewise=False, squared=True):
    #each row of X_tuning defines a location in the tuning parameter space along which to perform a quad scan and evaluate emit

    #X_meas must be shape (n,) and represent a 1d scan along the measurement domain
    
    #if samplewise=False, X should be shape: n_tuning_configs x (ndim-1)
    #the emittance for every point specified by X will be evaluated for every posterior sample path (broadcasting)
    
    #if samplewise=True, X must be shape: nsamples x (ndim-1)
    #the emittance of the nth sample will be computed ONLY for the nth point (row) specified by X
    

    #expand the X tensor to represent quad measurement scans at the locations in tuning parameter space specified by X
    n_steps_quad_scan = len(X_meas) #number of points in the measurement scan uniformly spaced along measurement domain
    n_tuning_configs = X_tuning.shape[-2] #the number of points in the tuning parameter space specified by X
    
    #prepare column of measurement scans coordinates
    X_meas_repeated = X_meas.repeat(n_tuning_configs).reshape(n_steps_quad_scan*n_tuning_configs, 1).repeat(*X_tuning.shape[:-2], 1,1)

    
    #repeat tuning configs as necessary and concat with column from the line above to make xs shape: 
    #(n_tuning_configs*n_steps_quad_scan) x d ,
    #where d is the full dimension of the model/posterior space (tuning & meas)
    xs_tuning = torch.repeat_interleave(X_tuning, n_steps_quad_scan, dim=-2) 

    xs = torch.cat((xs_tuning, X_meas_repeated), dim=-1)

    ys = post_paths(xs) #ys will be shape nsamples x (n_tuning_configs*n_steps_quad_scan)

    n_samples = ys.shape[-2]
    
    if samplewise: #evaluate emittance for the nth sample at the nth point specified by the input tensor X
                
        #reshape to put the relevant scans at the beginning of each row
        ys_temp = torch.cat((ys, torch.zeros(ys.shape[1]).reshape(1,-1)), dim=0) #add a row of zeros (will be discarded later)
        ys_temp = ys_temp.reshape(n_samples, -1) #reshape back to the original number of rows
        
        #reduce to relevant scans
        ys_samplewise = ys_temp[:,:n_steps_quad_scan] #ys is now shape nsamples x n_steps_quad_scan 

        
        emits_flat, emits_squared_raw_flat = compute_emits_from_batched_beamsize_scans(X_meas, ys_samplewise)

        #emits_flat, emits_squared_raw_flat will be tensors of shape nsamples x 1 where the nth element
        #is the emittance of the nth sample evaluated at the nth point specified by the input tensor X

    else: #broadcast (evaluate emittance for every sample for every point specified by the input tensor X)
        
        ys = ys.reshape(n_samples*n_tuning_configs, n_steps_quad_scan) #reshape into batchshape x n_steps_quad_scan
        emits_flat, emits_squared_raw_flat = compute_emits_from_batched_beamsize_scans(X_meas, ys)
        
        emits_flat = emits_flat.reshape(n_samples, -1) 
        emits_squared_raw_flat = emits_squared_raw_flat.reshape(n_samples, -1)
        
        #emits_flat, emits_squared_raw_flat will be tensors of shape nsamples x n_tuning_configs,
        #where n_tuning_configs is the number of rows in the input tensor X.
        #The nth column of the mth row represents the emittance of the mth sample, 
        #evaluated at the nth tuning config specified by the input tensor X.


    if squared:
        out = emits_squared_raw_flat
    else:
        out = emits_flat
            
    return out

   
def post_emit(post_paths, X_tuning, X_meas, samplewise=False, squared=True):
    #each row of X_tuning defines a location in the tuning parameter space along which to perform a quad scan and evaluate emit

    #X_meas must be shape (n,) and represent a 1d scan along the measurement domain
    
    #if samplewise=False, X should be shape: n_tuning_configs x (ndim-1)
    #the emittance for every point specified by X will be evaluated for every posterior sample path (broadcasting)
    
    #if samplewise=True, X must be shape: nsamples x (ndim-1)
    #the emittance of the nth sample will be computed for the nth point (row) specified by X
    

    #expand the X tensor to represent quad measurement scans at the locations in tuning parameter space specified by X
    n_steps_quad_scan = len(X_meas) #number of points in the measurement scan uniformly spaced along measurement domain
    n_tuning_configs = X_tuning.shape[0] #the number of points in the tuning parameter space specified by X
    
    #prepare column of measurement scans coordinates
    X_meas_repeated = X_meas.repeat(n_tuning_configs).reshape(n_steps_quad_scan*n_tuning_configs, 1)
#     X_meas_repeated = X_meas.repeat(n_tuning_configs).reshape(n_steps_quad_scan*n_tuning_configs, 1)
    
    #repeat tuning configs as necessary and concat with column from the line above to make xs shape: 
    #(n_tuning_configs*n_steps_quad_scan) x d ,
    #where d is the full dimension of the model/posterior space (tuning & meas)
    xs_tuning = torch.repeat_interleave(X_tuning, n_steps_quad_scan, dim=0)    
    xs = torch.cat((xs_tuning, X_meas_repeated), dim=1)
    
    ys = post_paths(xs) #ys will be shape nsamples x (n_tuning_configs*n_steps_quad_scan)

    n_samples = ys.shape[0]
    
    if samplewise: #evaluate emittance for the nth sample only at the nth point specified by the input tensor X
                
        #reshape to put the relevant scans at the beginning of each row
        ys_temp = torch.cat((ys, torch.zeros(ys.shape[1]).reshape(1,-1)), dim=0) #add a row of zeros (will be discarded later)
        ys_temp = ys_temp.reshape(n_samples, -1) #reshape back to the original number of rows
        
        #reduce to relevant scans
        ys_samplewise = ys_temp[:,:n_steps_quad_scan] #ys is now shape nsamples x n_steps_quad_scan 

        
        emits_flat, emits_squared_raw_flat = compute_emits_from_batched_beamsize_scans(X_meas, ys_samplewise)

        #emits_flat, emits_squared_raw_flat will be tensors of shape nsamples x 1 where the nth element
        #is the emittance of the nth sample evaluated at the nth point specified by the input tensor X

    else: #broadcast (evaluate emittance for every sample for every point specified by the input tensor X)
        
        ys = ys.reshape(n_samples*n_tuning_configs, n_steps_quad_scan) #reshape into batchshape x n_steps_quad_scan
        emits_flat, emits_squared_raw_flat = compute_emits_from_batched_beamsize_scans(X_meas, ys)
        
        emits_flat = emits_flat.reshape(n_samples, -1) 
        emits_squared_raw_flat = emits_squared_raw_flat.reshape(n_samples, -1)
        
        #emits_flat, emits_squared_raw_flat will be tensors of shape nsamples x n_tuning_configs,
        #where n_tuning_configs is the number of rows in the input tensor X.
        #The nth column of the mth row represents the emittance of the mth sample, 
        #evaluated at the nth tuning config specified by the input tensor X.


    if squared:
        out = emits_squared_raw_flat
    else:
        out = emits_flat
            
    return out


def post_mean_emit_flat_X_wrapper_for_scipy(model, X_meas, squared=True):
    def wrapped_func(X_tuning_flat):
        return post_mean_emit(model, torch.tensor(X_tuning_flat).reshape(1,-1), X_meas, squared=squared).flatten().detach().cpu().numpy()
    
    return wrapped_func

def post_mean_emit(model, X_tuning, X_meas, squared=True):
    
    
    X_meas_col = X_meas.repeat(X_tuning.shape[0]).reshape(-1,1)
    xs_tuning = torch.repeat_interleave(X_tuning, len(X_meas), dim=0) 
    xs = torch.cat((xs_tuning, X_meas_col), dim=1)
    
    ys = model.posterior(xs).mean
    
    ys_batch = ys.reshape(X_tuning.shape[0],-1)
    
    emits_flat, emits_squared_raw_flat = compute_emits_from_batched_beamsize_scans(X_meas, ys_batch)
    
    if squared:
        out = emits_squared_raw_flat
    else:
        out = emits_flat
            
    return out

def compute_emits_from_batched_beamsize_scans(xs_meas, ys_batch):

    #xs_meas is assumed to be a 1d tensor of shape (n_steps_quad_scan,) representing the measurement parameter inputs of the emittance scan
    #ys_batch is assumed to be shape batchsize x n_steps_quad_scan, where each row represents the beamsize outputs of an emittance scan with input given by xs_meas
    
    #note that every measurement scan is assumed to have been evaluated at the single set of inputs described by xs_meas
    
    xs_meas = xs_meas.reshape(-1,1)

    # least squares method to calculate parabola coefficients
    A_block = torch.cat((xs_meas**2, xs_meas, torch.tensor([1]).repeat(len(xs_meas)).reshape(xs_meas.shape)), dim=1)
    A = A_block.repeat(*ys_batch.shape[:-1], 1, 1).double()
    B = ys_batch.double()
    X = torch.linalg.lstsq(A, B).solution #these are the a,b,c coefficients for the parabolic measurement scan samples





    #analytically calculate the Sigma (beam) matrices from parabola coefficients (non-physical results are possible)
    l = 0.1
    d = 1.0
    M = torch.tensor([[1/((d*l)**2), 0, 0],
                     [-1/((d*l)**2), 1/(2*d*l), 0],
                      [1/((d*l)**2), -1/(d**3*l), 1/(d**2)]])

    sigs = torch.matmul(M.repeat(*X.shape[:-1],1,1).double(), X.reshape(*X.shape[:-1],3,1).double()) #column vectors of sig11, sig12, sig22

    Sigmas = sigs.reshape(-1,3).repeat_interleave(torch.tensor([1,2,1]), dim=1).reshape(*sigs.shape[:-2],2,2) #2x2 Sigma/covar beam matrix


    #compute emittances from Sigma (beam) matrices
    emits_squared_raw = torch.linalg.det(Sigmas)

    emits = torch.sqrt(emits_squared_raw) #these are the emittances for every tuning parameter combination.
    emits = torch.nan_to_num(emits, nan=50.)


    emits_squared_raw_flat = emits_squared_raw.reshape(ys_batch.shape[0], -1)
    emits_flat = emits.reshape(ys_batch.shape[0], -1)    

    return emits_flat, emits_squared_raw_flat      
