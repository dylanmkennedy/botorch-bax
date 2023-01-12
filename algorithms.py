import torch
from typing import Union, Optional
from torch import Tensor
from botorch.models.model import Model


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
        n_steps_sample_grid: Union[int, list[int]],
        gpu: Optional[bool] = False
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
        
        if gpu:
            if torch.cuda.is_available():
                self.gpu = True
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            else:
                print('CUDA not available. Using CPU.')
                self.gpu = False
        else:
            self.gpu = False
                
            
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
    
    

    
from samplingutils import draw_product_kernel_post_paths
from utils import unif_random_sample_domain
from emitutils import (sum_samplewise_emittance_flat_X_wrapper_for_scipy, sum_samplewise_emittance_flat_X, 
                       post_mean_emit_flat_X_wrapper_for_scipy, post_mean_emit)
from scipy.optimize import minimize
import os
from pathos.multiprocessing import ProcessingPool as Pool

class ScipyMinimizeEmittanceParallel(Algorithm):
    def __init__(
        self,
        domain: Tensor, #shape (ndim, 2) tensor domain[i,0], domain[i,1] are lower, upper bound respectively for ith input dimension. 
        n_samples_per_batch: int,
        n_sample_batches: Optional[int] = 4,
        n_steps_measurement_param: Optional[int] = 3
    ) -> None:
        
        self.domain = domain
        self.ndim = domain.shape[0]
        self.n_samples_per_batch = n_samples_per_batch 
        self.n_steps_measurement_param = n_steps_measurement_param
        self.X_meas = torch.linspace(*self.domain[-1], self.n_steps_measurement_param)
        self.n_sample_batches = n_sample_batches
        self.n_samples = self.n_samples_per_batch * self.n_sample_batches
        self.p = Pool(min(n_sample_batches, os.cpu_count()))
        
    def get_exe_paths(self, model: Model):
        
        self.post_paths = [draw_product_kernel_post_paths(model, n_samples=self.n_samples_per_batch) for i in range(self.n_sample_batches)]
        
        self.post_paths_for_scipy = [draw_product_kernel_post_paths(model.cpu(), n_samples=self.n_samples_per_batch) for i in range(self.n_sample_batches)]
                
        X_tuning_inits = [unif_random_sample_domain(self.n_samples_per_batch, self.domain[:-1]).double().flatten() for i in range(self.n_sample_batches)]
        
        #########################################
        #minimize
        def minimize_batch(args):
            post_paths_for_scipy, X_tuning_init = args
            target_func_for_scipy = sum_samplewise_emittance_flat_X_wrapper_for_scipy(post_paths_for_scipy, self.n_samples_per_batch, self.X_meas.cpu())
        
            res = minimize(target_func_for_scipy, X_tuning_init.detach().cpu().numpy(), 
                       bounds = self.domain[:-1].repeat(self.n_samples_per_batch,1).detach().cpu().numpy(), 
                       tol=1e-3, options = {'eps': 1e-03})
        
            x_stars_flat = torch.tensor(res.x)
                          
            return x_stars_flat
        
        args = [(self.post_paths_for_scipy[i], X_tuning_inits[i]) for i in range(self.n_sample_batches)]
        
        self.p.restart()
        
#         print('we made it this far1')

        X_star_flats = self.p.map(minimize_batch, args)
        
        self.p.close()
        self.p.join()
        #########################################
        
#         print('we made it this far2')
        x_stars = [X_star_flat.reshape(self.n_samples_per_batch,-1) for X_star_flat in X_star_flats]#each row represents its respective sample's optimal tuning config
        
        #expand the X tensor to represent quad measurement scans at the locations in tuning parameter space specified by X
        n_steps_quad_scan = len(self.X_meas) #number of points in the measurement scan uniformly spaced along measurement domain
        n_tuning_configs = self.n_samples_per_batch #should be the same as n_samples

        #prepare column of measurement scans coordinates
        X_meas_repeated = self.X_meas.repeat(n_tuning_configs).reshape(self.n_steps_measurement_param*n_tuning_configs, 1)

        #repeat tuning configs as necessary and concat with column from the line above to make xs shape: 
        #(n_tuning_configs*n_steps_measurement_scan) x d ,
        #where d is the full dimension of the model/posterior space (tuning & meas)
        xs = [torch.cat((torch.repeat_interleave(x_star, self.n_steps_measurement_param, dim=0), X_meas_repeated), dim=1) for x_star in x_stars]    
        
        ys = [post_path(x) for post_path, x in zip(self.post_paths, xs)]#evaluate posterior samples at input locations
        
        #reshape to put the relevant scans at the beginning of each row
        ys = [torch.cat((y, torch.zeros(y.shape[1]).reshape(1,-1)), dim=0).reshape(self.n_samples_per_batch, -1)[:,:self.n_steps_measurement_param] for y in ys]#add a row of zeros (will be discarded later)
 #reshape back to the original number of rows
        
        #reduce to relevant scans

        
        xs = torch.cat(xs,dim=0)
        ys = torch.cat(ys,dim=0)
        #reshape output

        xs_exe = xs.reshape(self.n_samples, self.n_steps_measurement_param, -1)
        ys_exe = ys.reshape(self.n_samples, self.n_steps_measurement_param, -1)
#         print('xs_exe.shape =', xs_exe.shape)
#         print('ys_exe.shape =', ys_exe.shape)    
        return xs_exe, ys_exe 
        

class ScipyMinimizeEmittance(Algorithm):
    def __init__(
        self,
        domain: Tensor, #shape (ndim, 2) tensor domain[i,0], domain[i,1] are lower, upper bound respectively for ith input dimension. 
        n_samples: int,
        n_steps_measurement_param: Optional[int] = 3,
        n_steps_exe_paths: Optional[int] = 50
    ) -> None:
        
        self.domain = domain
        self.ndim = domain.shape[0]
        self.n_samples = n_samples 
        self.n_steps_measurement_param = n_steps_measurement_param
        self.X_meas = torch.linspace(*self.domain[-1], self.n_steps_measurement_param)
        self.n_steps_exe_paths = n_steps_exe_paths


    def get_exe_paths(self, model: Model):
        
        self.post_paths = draw_product_kernel_post_paths(model, n_samples=self.n_samples)
        
        self.post_paths_for_scipy = draw_product_kernel_post_paths(model.cpu(), n_samples=self.n_samples)
                
        xs_tuning_init = unif_random_sample_domain(self.n_samples, self.domain[:-1]).double()

        X_tuning_init = xs_tuning_init.flatten()
        
        #########################################
        #minimize
        target_func_for_scipy = sum_samplewise_emittance_flat_X_wrapper_for_scipy(self.post_paths_for_scipy, self.n_samples, self.X_meas.cpu())
        
        res = minimize(target_func_for_scipy, X_tuning_init.detach().cpu().numpy(), 
                       bounds = self.domain[:-1].repeat(self.n_samples,1).detach().cpu().numpy(), 
                       tol=1e-3, options = {'eps': 1e-03})
        
        x_stars_flat = torch.tensor(res.x)
        #########################################
        
        
        x_stars = x_stars_flat.reshape(self.n_samples,-1) #each row represents its respective sample's optimal tuning config
        
        #expand the X tensor to represent quad measurement scans at the locations in tuning parameter space specified by X
        #prepare column of measurement scans coordinates
        X_meas_dense = torch.linspace(*self.domain[-1], self.n_steps_exe_paths)
        X_meas_repeated = X_meas_dense.repeat(self.n_samples).reshape(self.n_steps_exe_paths*self.n_samples, 1)

        #repeat tuning configs as necessary and concat with column from the line above to make xs shape: 
        #(n_tuning_configs*n_steps_measurement_scan) x d ,
        #where d is the full dimension of the model/posterior space (tuning & meas)
        xs = torch.repeat_interleave(x_stars, self.n_steps_exe_paths, dim=0)    
        xs = torch.cat((xs, X_meas_repeated), dim=1)
        
        ys = self.post_paths(xs) #evaluate posterior samples at input locations
        
        #reshape to put the relevant scans at the beginning of each row
        ys = torch.cat((ys, torch.zeros(ys.shape[1]).reshape(1,-1)), dim=0) #add a row of zeros (will be discarded later)
        ys = ys.reshape(self.n_samples, -1) #reshape back to the original number of rows
        
        #reduce to relevant scans
        ys = ys[:,:self.n_steps_exe_paths] #ys is now shape n_samples x self.n_steps_exe_paths 
        
        
        #reshape output
        xs_exe = xs.reshape(self.n_samples, self.n_steps_exe_paths, -1)
        ys_exe = ys.reshape(self.n_samples, self.n_steps_exe_paths, -1)
        
        return xs_exe, ys_exe 
            
            
    def mean_output(self, model, num_restarts=None):
        
        target_func_for_scipy = post_mean_emit_flat_X_wrapper_for_scipy(model, self.X_meas, squared=True)
        
        
        
        if num_restarts is None:
            
            X_tuning_flat_init = unif_random_sample_domain(1, self.domain[:-1]).double().flatten()

            res = minimize(target_func_for_scipy, X_tuning_flat_init.detach().cpu().numpy(), 
                           bounds = self.domain[:-1].detach().cpu().numpy(), 
                           tol=1e-3, options = {'eps': 1e-03})

            X_tuning_star = torch.tensor(res.x).reshape(1,-1)
            emit_star = post_mean_emit(model, X_tuning_star, self.X_meas, squared=True).flatten()
            
            
        else:
            
            def minimize_batch(args):
                X_tuning_init = args

                res = minimize(target_func_for_scipy, X_tuning_init.flatten().detach().cpu().numpy(), 
                           bounds = self.domain[:-1].detach().cpu().numpy(), 
                           tol=1e-3, options = {'eps': 1e-03})

                X_tuning_star = torch.tensor(res.x).reshape(1,-1)

                return X_tuning_star

            X_tuning_inits = unif_random_sample_domain(num_restarts, self.domain[:-1]).double()
            args = [X_tuning_init for X_tuning_init in X_tuning_inits]

            p = Pool()

            p.restart()


            results = p.map(minimize_batch, args)

            p.close()
            p.join()

            X_tuning_stars = torch.tensor([])
            for result in results:
                X_tuning_stars = torch.cat((X_tuning_stars, result), dim=0)
            emit_stars = post_mean_emit(model, X_tuning_stars, self.X_meas, squared=True).flatten()

            id_min_emit = torch.argmin(emit_stars)

            X_tuning_star = X_tuning_stars[id_min_emit].reshape(1,-1)
            emit_star = emit_stars[id_min_emit].reshape(1)

                
        
        return X_tuning_star, emit_star
        
            
            