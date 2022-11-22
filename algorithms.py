import torch
from typing import Union, Optional
from torch import Tensor
from botorch.models.model import Model
from utils import unif_random_sample_domain

class Algorithm:
    
    def __init__(
        self,
        n_samples: int
    ) -> None:
        
        self.n_samples = n_samples
        
        
class GridScanAlgo(Algorithm):
    
    def __init__(
        self,
        domain: Tensor, #shape (ndim, 2) tensor domain[i,0], domain[i,1] are lower, upper bound respectively for ith input dimension. 
        n_samples: int,
        n_steps_sample_grid: Union[int, list[int]]
    ) -> None:
        
        self.domain = domain
        self.ndim = domain.shape[0]
        self.n_samples = n_samples 
        
        try: #check to see if n_steps_sample_grid is a list
            n_steps_sample_grid = list(n_steps_sample_grid)
        except: 
            #in case n_steps_sample_grid is an integer rather than a list, we make it a list with that integer repeated for every dimension
            n_steps_sample_grid = [n_steps_sample_grid]*int(self.ndim)
            
        if len(n_steps_sample_grid) != self.ndim:
            raise ValueError("If n_steps_sample_grid is a list, it must have length = ndim")
            
        # a list (length=ndim) of integers specifying the number of steps per dimension in the sample grid.    
        self.n_steps_sample_grid = n_steps_sample_grid 
                
            
    def build_input_mesh(self):
        linspace_list = [torch.linspace(bounds[0], bounds[1], n_steps).double() 
                         for n_steps, bounds in zip(self.n_steps_sample_grid, self.domain)]

        x_mesh_tuple = torch.meshgrid(*linspace_list, indexing='ij')

        x_mesh_columnized_tuple = tuple(x_mesh.reshape(-1,1) for x_mesh in x_mesh_tuple)


        if self.ndim == 1:
            xs_n_by_d = x_mesh_columnized_tuple[0]
        else:
            xs_n_by_d = torch.cat(x_mesh_columnized_tuple, dim=1)
            
        return xs_n_by_d, x_mesh_tuple
            
    def eval_sample_grid_scans(self, model: Model):
        
        sample_xs, x_mesh_tuple = self.build_input_mesh()
        

        #evaluate grid scans for each posterior sample
        with torch.no_grad(): 
            p = model.posterior(sample_xs)
            sample_ys = p.rsample(torch.Size([self.n_samples]))
            
        y_mesh_samples = sample_ys.reshape(self.n_samples, *x_mesh_tuple[0].shape)

        self.sample_xs, self.sample_ys, self.x_mesh_tuple, self.y_mesh_samples = sample_xs, sample_ys, x_mesh_tuple, y_mesh_samples
        
        return sample_xs, sample_ys, x_mesh_tuple, y_mesh_samples
    
    
class GridOpt(GridScanAlgo):
    
    def get_exe_paths(self, model: Model):
        
        sample_xs, sample_ys = self.eval_sample_grid_scans(model)[:2]
        
        #get exe path subsequences (in this case, just 1 (x,y) pair from each posterior sample grid scan)
        ys_opt, max_ids = torch.max(sample_ys, dim=1)
        xs_opt = sample_xs[max_ids]

        xs_exe = xs_opt.reshape(-1,1,self.ndim) #xs_exe.shape = (n_samples, len_exe_path, ndim)
        ys_exe = ys_opt.reshape(-1,1,1)    #ys_exe.shape = (n_samples, len_exe_path, 1)
        
        return xs_exe, ys_exe

class GridMinimizeEmittance(GridScanAlgo):
    
    def __init__(
    self,
    domain: Tensor, #shape (ndim, 2) tensor such that domain[i,0], domain[i,1] are lower, upper bound respectively for ith input dimension. 
    n_samples: int,
    n_steps_tuning_params: Union[int, list[int]],
    n_steps_measurement_param: Optional[int] = 3,
    quad_length: Optional[float] = 0.1,
    drift_length: Optional[float] = 1.0,
    squared: Optional[bool] = False

) -> None:

        if n_steps_measurement_param < 3:
            raise ValueError("n_steps_measurement_param must be at least 3 for parabola fitting.")
        self.domain = domain
        self.ndim = domain.shape[0]
        self.n_samples = n_samples
        self.n_steps_measurement_param = n_steps_measurement_param
        self.quad_length = torch.tensor(quad_length).double()
        self.drift_length = torch.tensor(drift_length).double()
        self.squared = squared
        
        try: #check to see if n_steps_sample_grid is a list
            n_steps_tuning_params = list(n_steps_tuning_params)
        except:
            #in case n_steps_sample_grid is an integer rather than a list, 
            #we make it a list with that integer repeated for every tuning parameter dimension (all but the last dimension)
            n_steps_tuning_params = [n_steps_tuning_params]*int(self.ndim - 1)
            
        if len(n_steps_tuning_params) != (self.ndim - 1):
            raise ValueError("If n_steps_tuning_params is a list, it must have length = (ndim - 1)")
        self.n_steps_tuning_params = n_steps_tuning_params
        self.n_steps_sample_grid = n_steps_tuning_params + [n_steps_measurement_param] #append the number of steps for the measurement param
        
        
    def get_exe_paths(self, model: Model):
    
        #evaluate samples from posterior at mesh points (same mesh points for every sample)
        sample_xs, sample_ys, x_mesh_tuple, y_mesh_samples = self.eval_sample_grid_scans(model)
        
        emits_flat, emits_squared_raw_flat, xs_meas = self.compute_emits_grid_batch(x_mesh_tuple, y_mesh_samples)
        
   
        
        if self.squared:
            emit_best_ids = torch.argmin(emits_squared_raw_flat, dim=1)
        else:
            emit_best_ids = torch.argmin(emits_flat, dim=1)


        tuning_param_meshes = [x_mesh.select(dim=-1, index=0) for x_mesh in x_mesh_tuple[:-1]]
        
        tuning_configs_flat = torch.cat(tuple(mesh.reshape(-1,1) for mesh in tuning_param_meshes), dim=1)

        tuning_configs_best = torch.index_select(tuning_configs_flat, dim=0, index=emit_best_ids) #best tuning params for each sample
        
        xs_exe = torch.cat( ( torch.repeat_interleave(tuning_configs_best, torch.ones(len(tuning_configs_best)).int()*len(xs_meas), dim=0), 
            xs_meas.repeat(self.n_samples, 1) 
           ),
          dim = 1).reshape(self.n_samples,len(xs_meas),-1) #these are the execution path inputs
        
        
        temp = y_mesh_samples.reshape(self.n_samples, -1, len(xs_meas)) #temp is the sample_ys with flattened tuning params
        
        ys_exe = temp[torch.arange(temp.size(0)), emit_best_ids].reshape(self.n_samples, len(xs_meas), 1)
        
        self.xs_exe, self.ys_exe, self.emits_flat, self.emits_squared_raw_flat = xs_exe, ys_exe, emits_flat, emits_squared_raw_flat
        
        return xs_exe, ys_exe
    
    def mean_prediction(self, model: Model):
        
        sample_xs, x_mesh_tuple = self.build_input_mesh()
        
            
        #evaluate grid scans for each posterior sample
        with torch.no_grad(): 
            p = model.posterior(sample_xs)
            mean_ys = p.mean
            
        y_mean_mesh = mean_ys.reshape(1, *x_mesh_tuple[0].shape)
        
        emits_flat, emits_squared_raw_flat, xs_meas = self.compute_emits_grid_batch(x_mesh_tuple, y_mean_mesh)

        if self.squared:
            emit_best_ids = torch.argmin(emits_squared_raw_flat, dim=1)
        else:
            emit_best_ids = torch.argmin(emits_flat, dim=1)


        tuning_param_meshes = [x_mesh.select(dim=-1, index=0) for x_mesh in x_mesh_tuple[:-1]]
        
        tuning_configs_flat = torch.cat(tuple(mesh.reshape(-1,1) for mesh in tuning_param_meshes), dim=1)

        tuning_config_best = torch.index_select(tuning_configs_flat, dim=0, index=emit_best_ids) #best tuning params for each sample
        
        return tuning_config_best
    
    def compute_emits_grid_batch(self, x_mesh_tuple, y_mesh_batch):
        #extract the 1d measurement scan inputs from the mesh
        xs_meas_mesh = x_mesh_tuple[-1]
        xs_meas = xs_meas_mesh[tuple([0]*(len(x_mesh_tuple)-1))].reshape(-1,1)
        
        
        
#         ############################## 
        # least squares method to calculate parabola coefficients
        A = torch.cat((xs_meas**2, xs_meas, torch.tensor([1]).repeat(len(xs_meas)).reshape(xs_meas.shape)), dim=1)
        A = A.repeat(*y_mesh_batch.shape[:-1], 1, 1)

        B = y_mesh_batch
#         print(y_mesh_samples.shape)
#         print(A.shape)
        X = torch.linalg.lstsq(A, B).solution #these are the a,b,c coefficients for the parabolic measurement scan samples
#         print(X.shape)
#         print(X)


        
        
        
        #analytically calculate the Sigma (beam) matrices from parabola coefficients (non-physical results are possible)
        l = self.quad_length
        d = self.drift_length
        M = torch.tensor([[1/((d*l)**2), 0, 0],
                         [-1/((d*l)**2), 1/(2*d*l), 0],
                          [1/((d*l)**2), -1/(d**3*l), 1/(d**2)]])
        
        sigs = torch.matmul(M.repeat(*X.shape[:-1],1,1).double(), X.reshape(*X.shape[:-1],3,1)) #column vectors of sig11, sig12, sig22
        
        Sigmas = sigs.reshape(-1,3).repeat_interleave(torch.tensor([1,2,1]), dim=1).reshape(*sigs.shape[:-2],2,2) #2x2 Sigma/covar beam matrix
        
        
        #compute emittances from Sigma (beam) matrices
        emits_squared_raw = torch.linalg.det(Sigmas)
        
        emits = torch.sqrt(emits_squared_raw) #these are the emittances for every tuning parameter combination.
        emits = torch.nan_to_num(emits, nan=50.)

        
        #select the measurement scan with lowest computed emittance from each sample as that sample's execution path subsequence.
        emits_squared_raw_flat = emits_squared_raw.reshape(y_mesh_batch.shape[0], -1)
        emits_flat = emits.reshape(y_mesh_batch.shape[0], -1)    
    
        return emits_flat, emits_squared_raw_flat, xs_meas
    
    
    
class GradBasedMinimizeEmittance(Algorithm):
    def __init__(
        self,
        domain: Tensor, #shape (ndim, 2) tensor domain[i,0], domain[i,1] are lower, upper bound respectively for ith input dimension. 
        n_samples: int,
        n_iter: int,
        n_steps_measurement_param: Optional[int] = 3,
        quad_length: Optional[float] = 0.1,
        drift_length: Optional[float] = 1.0
    ) -> None:
        
        self.domain = domain
        self.ndim = domain.shape[0]
        self.n_samples = n_samples 
        self.n_iter = n_iter
        self.n_steps_measurement_param = n_steps_measurement_param
        self.xs_meas = torch.linspace(*self.domain[-1], self.n_steps_measurement_param)
        self.quad_length = quad_length
        self.drift_length = drift_length
        
    def get_exe_paths(self, model: Model):
        
        #need some initial set of tuning config candidates. Use uniform random sample from domain for now?
        
        X = unif_random_sample_domain(self.n_samples, self.domain[:-1]).reshape(-1)
        
        X.requires_grad = True
        
        optimizer = torch.optim.Adam([X], lr=0.01)
        
        #evaluate sample measurement scans using current X and initial (non-batch) model
        sample_xs, sample_ys = self.eval_sample_meas_scans(model, X) #this should broadcast for the first iteration?
        
        #condition initial (non-batch) model on batched sample observations
        fmodels = self.condition_fmodels(model, sample_xs, sample_ys) #this should broadcast for the first iteration?
        
        for i in range(self.n_iter):
            
            #use batched sample results to evaluate sum of sample emits (loss)
            loss = self.sum_sample_scan_emits(sample_ys)
            print(loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step() #updates X
            
            #evaluate sample measurement scans using current X and batched models
            sample_xs, sample_ys = self.eval_sample_meas_scans(fmodels, X)  #these shapes should be compatible?   
            

            #condition batched models on the batched sample results
            fmodels = self.condition_fmodels(fmodels, sample_xs, sample_ys) #these batch shapes should be compatible?
            
            sample_xs, sample_ys = sample_xs, sample_ys
            print(sample_xs[:,0,:-1])
        self.xs_exe, self.ys_exe = sample_xs, sample_ys
        
        return self.xs_exe, self.ys_exe
    
    def eval_sample_meas_scans(self, fmodels, X):
    #X is shape 1 x (n_samples * (ndim-1)) representing the tuning config candidates for each sample, concatenated together and flattened
        xs_tuning = X.reshape((self.n_samples, self.ndim-1))
        xs_tuning = torch.repeat_interleave(xs_tuning, self.n_steps_measurement_param, dim=0)
        
        xs_meas = self.xs_meas.repeat(1, self.n_samples).reshape(-1,1)
        
        sample_xs = torch.cat((xs_tuning, xs_meas), dim=1).reshape(self.n_samples, self.n_steps_measurement_param, self.ndim) 
                
        p = fmodels.posterior(sample_xs)
        sample_ys = p.rsample(torch.Size([1]))[0] 
        
        return sample_xs, sample_ys
    
    def sum_sample_scan_emits(self, sample_ys):
        
        sample_emits_squared = self.compute_scan_emits_batch(sample_ys)[1].reshape(-1)
        
        return torch.sum(sample_emits_squared)
    
    def compute_scan_emits_batch(self, sample_ys):
        
        xs_meas = self.xs_meas.reshape(-1,1)
        
        
        
#         ############################## 
        # least squares method to calculate parabola coefficients
        A = torch.cat((xs_meas**2, xs_meas, torch.tensor([1]).repeat(len(xs_meas)).reshape(xs_meas.shape)), dim=1)
        A = A.repeat(*sample_ys.shape[:-2], 1, 1).double()

        B = sample_ys
#         print('A.shape =', A.shape)
#         print('B.shape =', B.shape)

#         print(y_mesh_samples.shape)
#         print(A.shape)
        X = torch.linalg.lstsq(A, B).solution #these are the a,b,c coefficients for the parabolic measurement scan samples
#         print(X.shape)
#         print(X)

        
        
        
        #analytically calculate the Sigma (beam) matrices from parabola coefficients (non-physical results are possible)
        l = self.quad_length
        d = self.drift_length
        M = torch.tensor([[1/((d*l)**2), 0, 0],
                         [-1/((d*l)**2), 1/(2*d*l), 0],
                          [1/((d*l)**2), -1/(d**3*l), 1/(d**2)]])
        
        sigs = torch.matmul(M.repeat(*X.shape[:-2],1,1).double(), X.reshape(*X.shape[:-2],3,1)) #column vectors of sig11, sig12, sig22
        
        Sigmas = sigs.reshape(-1,3).repeat_interleave(torch.tensor([1,2,1]), dim=1).reshape(*sigs.shape[:-2],2,2) #2x2 Sigma/covar beam matrix
        
        
        #compute emittances from Sigma (beam) matrices
        emits_squared_raw = torch.linalg.det(Sigmas)
        
        emits = torch.sqrt(emits_squared_raw) #these are the emittances for every tuning parameter combination.
        emits = torch.nan_to_num(emits, nan=50.)

        
        emits_squared_raw_flat = emits_squared_raw.reshape(sample_ys.shape[0], -1)
        emits_flat = emits.reshape(sample_ys.shape[0], -1)    
    
        return emits_flat, emits_squared_raw_flat
        
    def condition_fmodels(self, fmodels, xs, ys):
        #ADD HERE: input/outcome transforms
        xs_transformed = fmodels.input_transform(xs)
        ys_transformed = fmodels.outcome_transform(ys)[0]
        return fmodels.condition_on_observations(xs, ys)
        
        
        
        
        
        