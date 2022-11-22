import torch
from emitutils import toy_beam_size_squared_nd, fit_gp_model_emittance
from utils import unif_random_sample_domain
from matplotlib import pyplot as plt
from algorithms import GridMinimizeEmittance
from acquisition import ExpectedInformationGain
from botorch.optim import optimize_acqf
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import dill


def convergence_results(trial_data, plot=False):
        
    target_func = toy_beam_size_squared_nd

    settings = trial_data['settings']
    
    domain = settings['domain']
    ndim = settings['ndim']
    n_obs_init = settings['n_obs_init']
    n_samples = settings['n_samples']
    n_steps_tuning_params = settings['n_steps_tuning_params']
    n_steps_measurement_param = settings['n_steps_measurement_param']
    n_trials = settings['n_trials']
    n_iter = settings['n_iter']
    squared = settings['squared']
    
    all_dists = []
    all_stds = []
    all_gt_emits_at_x_star_pred = []
    all_avg_sample_dists = []
    for key in trial_data.keys():
        if key == 'settings':
            pass
        else:
            trial = key
            distances_apart = []
            std_devs = []
            gt_emits_at_x_star_pred = []
            avg_sample_distances = []
            
            for i in trial_data[trial].keys():
                iter_data = trial_data[trial][i]
                ##########################################
                rng_state = iter_data['rng_state']
                model = iter_data['model']
                acq_fn = reconstruct_acq_fn(settings, model, rng_state)
                ##########################################
#                 acq_fn = iter_data['acq_fn']
                ##########################################
                xs_exe = acq_fn.algo.xs_exe
                x_stars = xs_exe[:,0,:-1]
#                 x_star_pred = torch.mean(x_stars, dim=0)


    
                pred_algo = GridMinimizeEmittance(domain = domain, 
                               n_samples = 1, 
                               n_steps_tuning_params = 51,
                                n_steps_measurement_param = 11,
                                squared = squared)
    
                x_star_pred = pred_algo.mean_prediction(acq_fn.model)
        
#                 x_star_pred = acq_fn.algo.mean_prediction(acq_fn.model)


                #save memory?
                del acq_fn
                del pred_algo
            
                single_tuning_config_domain = torch.Tensor([[x_star_pred_i, x_star_pred_i] for x_star_pred_i in x_star_pred.reshape(-1)])
                single_tuning_config_domain = torch.cat((single_tuning_config_domain, domain[-1:]), dim=0)
        
                single_scan_algo = GridMinimizeEmittance(domain = single_tuning_config_domain, 
                               n_samples = 1, 
                               n_steps_tuning_params = 1,
                                n_steps_measurement_param = 11,
                                squared = squared)

                xs, x_mesh_tuple = single_scan_algo.build_input_mesh()

                ys = target_func(xs)

                y_mesh = ys.reshape(1, *x_mesh_tuple[0].shape)

                emits_flat, emits_squared_raw_flat, xs_meas = single_scan_algo.compute_emits_grid_batch(x_mesh_tuple, y_mesh)
                
                gt_emit_at_x_star_pred = emits_flat.reshape(-1)
                
#                 print(x_star_pred)
#                 print(xs)
#                 print(gt_emit_at_x_star_pred)
                
#                 break

                
                x_star_gt = torch.zeros(ndim-1)
                
                distance_apart = torch.sqrt(torch.sum((x_star_pred - x_star_gt)**2.))
                
                avg_sample_distance_apart = torch.mean(torch.sqrt(torch.sum((x_stars - x_star_gt)**2., dim=1)), dim=0)
                
                distances_apart += [distance_apart]
                avg_sample_distances += [avg_sample_distance_apart]
                
                if ndim == 2:
                    std_dev = torch.std(x_stars)
                else:
                    std_dev = torch.sqrt(torch.linalg.det(torch.cov(x_stars.T)))
                std_devs += [std_dev]
                
                
                gt_emits_at_x_star_pred += [gt_emit_at_x_star_pred]
                
            all_dists += [distances_apart]    
            all_stds += [std_devs]
            all_gt_emits_at_x_star_pred += [gt_emits_at_x_star_pred]
            all_avg_sample_dists += [avg_sample_distances]
            
    all_dists = torch.Tensor(all_dists)
    all_stds = torch.Tensor(all_stds)
    all_gt_emits_at_x_star_pred = torch.Tensor(all_gt_emits_at_x_star_pred)
    all_avg_sample_dists = torch.Tensor(all_avg_sample_dists)
    
    if plot:
        plt.plot(torch.mean(all_dists, dim=0))
        plt.title('Distance from predicted x* to ground truth value')
        plt.show()

        plt.plot(torch.mean(all_stds, dim=0))
        plt.title('Std_dev(x*)')
        plt.show()
    
    return all_dists, all_stds, all_gt_emits_at_x_star_pred, all_avg_sample_dists

def reconstruct_acq_fn(settings, model, rng_state):
    
    domain = settings['domain']
    n_samples = settings['n_samples']
    n_steps_tuning_params = settings['n_steps_tuning_params']
    n_steps_measurement_param = settings['n_steps_measurement_param']
    squared = settings['squared']
    
    algo = GridMinimizeEmittance(domain = domain, 
                               n_samples = n_samples, 
                               n_steps_tuning_params = n_steps_tuning_params,
                                n_steps_measurement_param = n_steps_measurement_param,
                                squared = squared)
    
    torch.set_rng_state(rng_state)
    
    acq_fn = ExpectedInformationGain(model = model, algo = algo)
    
    
    return acq_fn



def iter_plot3d(trial_data, trial, iter_list):
    
    target_func = toy_beam_size_squared_nd
    
    settings = trial_data['settings']
    
    domain = settings['domain']
    ndim = settings['ndim']
    n_obs_init = settings['n_obs_init']
    n_samples = settings['n_samples']
    n_steps_tuning_params = settings['n_steps_tuning_params']
    n_steps_measurement_param = settings['n_steps_measurement_param']
    n_trials = settings['n_trials']
    n_iter = settings['n_iter']
    

    print('Trial', trial, '\n')
    for i in iter_list:
        print('Iteration ' + str(i) + ':')
        iter_data = trial_data[trial][i]
        ##########################################
        rng_state = iter_data['rng_state']
        model = iter_data['model']
        acq_fn = reconstruct_acq_fn(settings, model, rng_state)
        ##########################################
#         acq_fn = iter_data['acq_fn']
        ##########################################
        x_obs = iter_data['x_obs']
        y_obs = iter_data['y_obs']
        x_next = iter_data['x_next']
        
        xs = acq_fn.algo.sample_xs
        x_mesh_tuple = acq_fn.algo.x_mesh_tuple
        s = acq_fn.algo.y_mesh_samples
        

        xs_exe, ys_exe, emits_flat, emits_squared_raw_flat = acq_fn.algo.xs_exe, acq_fn.algo.ys_exe, acq_fn.algo.emits_flat, acq_fn.algo.emits_squared_raw_flat 
        
        emit_stars = torch.min(emits_flat, dim=1)[0]

        x0s_exe, x1s_exe = xs_exe[:,0,0], xs_exe[:,0,1]

        with torch.no_grad():
            p = acq_fn.model.posterior(xs)
            m = p.mean
            var = p.variance
            eig = torch.tensor([acq_fn(x.reshape(1,xs.shape[1])) for x in xs])

        eig = eig.reshape(x_mesh_tuple[0].shape)
        var_mesh = var.reshape(x_mesh_tuple[0].shape)
        m_mesh = m.reshape(x_mesh_tuple[0].shape)


        y_mesh = target_func(xs).reshape(x_mesh_tuple[0].shape)

        y_mesh_gt = y_mesh.reshape(1, *y_mesh.shape) #reshape for next step (which expects multiple batches for multiple samples)

        gt_emits_flat = acq_fn.algo.compute_emits_grid_batch(x_mesh_tuple, y_mesh_gt)[0]
        gt_emits = gt_emits_flat.reshape(n_steps_tuning_params,n_steps_tuning_params)

        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #plotting
        
        fig, axes = plt.subplots(2,2)
        fig.set_size_inches((8, 8))


        ####################################
        ax = axes[0,0]

        im = ax.pcolor(x_mesh_tuple[0].select(dim=2,index=0), x_mesh_tuple[1].select(dim=2,index=0),gt_emits*1.e6)
        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.scatter(x0s_exe, x1s_exe, marker='x', s=80, c='orange', label='Sample Min Emit')
        if len(x_obs) > n_obs_init:
            ax.scatter(x_obs[n_obs_init:,0], x_obs[n_obs_init:,1], marker='o', s=40, c='magenta', label='Acquisitions')

        ax.set_xlabel('Tuning Param 0')
        ax.set_ylabel('Tuning Param 1')
        ax.set_title('Ground Truth Emittance')
        ax.legend()
        ####################################
        ax = axes[0,1]

        sid = 0
        im = ax.pcolor(x_mesh_tuple[0].select(dim=2,index=0), x_mesh_tuple[1].select(dim=2,index=0),emits_flat[sid].reshape(n_steps_tuning_params,n_steps_tuning_params))
        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_xlabel('Tuning Param 0')
        ax.set_ylabel('Tuning Param 1')
        ax.set_title('Estimated Emittance (Sample ' + str(sid) + ')')
    #     ax.legend()

        ####################################
        ax = axes[1,0]

        h, xedges, yedges, im = ax.hist2d(x0s_exe.tolist(), x1s_exe.tolist(), bins=torch.linspace(-2,2,12), vmax=100)
        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_xlabel('Tuning Param 0')
        ax.set_ylabel('Tuning Param 1')
        ax.set_title('Distribution of Sample Min Emits')    
        #####################################
        ax = axes[1,1]

        slice = 5 
        tuning_param0 = x_mesh_tuple[0][slice,slice,0]
        tuning_param1 = x_mesh_tuple[1][slice,slice,0]

        sample = s[0]
        ax.plot(x_mesh_tuple[2][slice,slice,:], sample[slice,slice,:], c='b', alpha=0.25, label='Samples')
        for sample in s[1:]:
            ax.plot(x_mesh_tuple[2][slice,slice,:], sample[slice,slice,:], c='b', alpha=0.25)

        ax.plot(x_mesh_tuple[2][slice,slice,:], y_mesh[slice,slice,:]*1.e6,c='r', label='Ground Truth')


        ax.set_xlabel('Measurement Param')
        ax.set_ylabel('Beam Size Squared')
        ax.set_title('Measurement Scans for Optimal Tuning Config')
        ax.legend()

        plt.tight_layout()
        print('Results after', len(x_obs), 'observations')
        print('Highest frequency = ', str(torch.max(torch.tensor(h)).numpy()))

        plt.show()

    #     plt.hist(torch.sqrt(emits_squared_raw_flat.reshape(-1, n_steps, n_steps)[:,5,5]).numpy())
        plt.hist(emits_flat.reshape(-1, n_steps_tuning_params, n_steps_tuning_params)[:,5,5].numpy())
        plt.title('Model Results at True Optimal Tuning Config')
        plt.ylabel('Frequency')
        plt.xlabel('Emittance')
        plt.tight_layout()
        plt.show()
        
        
        plt.hist(emit_stars.numpy())
        plt.title('Sample Minimum Emittances')
        plt.ylabel('Frequency')
        plt.xlabel('Emittance')
        plt.tight_layout()
        plt.show()
        

def iter_plot2d(trial_data, trial, iter_list):
    
    target_func = toy_beam_size_squared_nd

    settings = trial_data['settings']
    
    domain = settings['domain']
    ndim = settings['ndim']
    n_obs_init = settings['n_obs_init']
    n_samples = settings['n_samples']
    n_steps_tuning_params = settings['n_steps_tuning_params']
    n_steps_measurement_param = settings['n_steps_measurement_param']
    n_trials = settings['n_trials']
    n_iter = settings['n_iter']
    
    
    print('Trial', trial, '\n')
    for i in iter_list:
        print('Iteration ' + str(i) + ':')
        iter_data = trial_data[trial][i]
        ##########################################
        rng_state = iter_data['rng_state']
        model = iter_data['model']
        acq_fn = reconstruct_acq_fn(settings, model, rng_state)
        ##########################################
#         acq_fn = iter_data['acq_fn']
        ##########################################        x_obs = iter_data['x_obs']
        y_obs = iter_data['y_obs']
        x_next = iter_data['x_next']
        
        xs = acq_fn.algo.sample_xs
        x_mesh_tuple = acq_fn.algo.x_mesh_tuple
        s = acq_fn.algo.y_mesh_samples
        

        xs_exe, ys_exe, emits_flat = acq_fn.algo.xs_exe, acq_fn.algo.ys_exe, acq_fn.algo.emits_flat 








        with torch.no_grad():
            p = acq_fn.model.posterior(xs)
            m = p.mean
            var = p.variance
            eig = torch.tensor([acq_fn(x.reshape(1,xs.shape[1])) for x in xs])

        eig = eig.reshape(x_mesh_tuple[0].shape)
        var_mesh = var.reshape(x_mesh_tuple[0].shape)
        m_mesh = m.reshape(x_mesh_tuple[0].shape)


        y_mesh = target_func(xs).reshape(x_mesh_tuple[0].shape)

        y_mesh_gt = y_mesh.reshape(1, *y_mesh.shape) #reshape for next step (which expects multiple batches for multiple samples)

        gt_emits_flat = acq_fn.algo.compute_emits_grid_batch(x_mesh_tuple, y_mesh_gt)[0]
        gt_emits = gt_emits_flat.reshape(-1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #plotting


        fig, axes = plt.subplots(2, 4)
        fig.set_size_inches((16, 8))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes 
        ax = axes[1,0]
        im = ax.pcolor(x_mesh_tuple[0], x_mesh_tuple[1], m_mesh)
        ax.scatter(x_obs[:n_obs_init,0], x_obs[:n_obs_init,1], c='cyan', label='Init Data')
        if len(x_obs)>n_obs_init:
            ax.scatter(x_obs[n_obs_init:,0], x_obs[n_obs_init:,1], c='m', label='Acquisitions')
        for x_exe in xs_exe[:-1]:
            ax.axvline(x=x_exe[0,0], ymax=0.1, c='orange')
        ax.axvline(x=xs_exe[-1][0,0], ymax=0.1, c='orange', label='Sample Min Emit')    
        ax.axvline(x=0, ymax=0.2, c='r', label='True Min Emit')
        if x_next is not None:
            ax.scatter(x_next[0,0], x_next[0,1], marker='x', s=150, c='r', label='Max EIG')

        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_xlabel('Tuning Param')
        ax.set_ylabel('Measurement Quad')
        ax.set_title('Posterior Mean (Beam Size Squared)')
    #     ax.legend()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes 
        ax = axes[0,0]
        im = ax.pcolor(x_mesh_tuple[0], x_mesh_tuple[1], y_mesh*1.e6)
        ax.axvline(x=0, ymax=0.2, c='r', label='True Min Emit')

        for x_exe in xs_exe[:-1]:
            ax.axvline(x=x_exe[0,0], ymax=0.1, c='orange')
        ax.axvline(x=xs_exe[-1][0,0], ymax=0.1, c='orange', label='Sample Min Emit')  

        ax.scatter(x_obs[:n_obs_init,0], x_obs[:n_obs_init,1], c='cyan', label='Init Data')
        if len(x_obs)>n_obs_init:
            ax.scatter(x_obs[n_obs_init:,0], x_obs[n_obs_init:,1], c='m', label='Acquisitions')

        if x_next is not None:
            ax.scatter(x_next[0,0], x_next[0,1], marker='x', s=150, c='r', label='Max EIG')

        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


        ax.set_xlabel('Tuning Param')
        ax.set_ylabel('Measurement Quad')
        ax.set_title('Ground Truth (Beam Size Squared)')
        ax.legend(loc='upper left')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes
        ax = axes[1,1]
        im = ax.pcolor(x_mesh_tuple[0], x_mesh_tuple[1],var_mesh)
        ax.scatter(x_obs[:n_obs_init,0], x_obs[:n_obs_init,1], c='cyan', label='Init Data')
        if len(x_obs)>n_obs_init:
            ax.scatter(x_obs[n_obs_init:,0], x_obs[n_obs_init:,1], c='m', label='Acquisitions')
        for x_exe in xs_exe[:-1]:
            ax.axvline(x=x_exe[0,0], ymax=0.1, c='orange')
        ax.axvline(x=xs_exe[-1][0,0], ymax=0.1, c='orange', label='Sample Min Emit')    
        ax.axvline(x=0, ymax=0.2, c='r', label='True Min Emit')
        if x_next is not None:
            ax.scatter(x_next[0,0], x_next[0,1], marker='x', s=150, c='r', label='Max EIG')

        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


        ax.set_xlabel('Tuning Param')
        ax.set_ylabel('Measurement Quad')
        ax.set_title('Posterior Variance (Beam Size Squared)')
    #     ax.legend()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes
        ax = axes[0,1]
        im = ax.pcolor(x_mesh_tuple[0], x_mesh_tuple[1],eig)
        ax.scatter(x_obs[:n_obs_init,0], x_obs[:n_obs_init,1], c='cyan', label='Init Data')
        if len(x_obs)>n_obs_init:
            ax.scatter(x_obs[n_obs_init:,0], x_obs[n_obs_init:,1], c='m', label='Acquisitions')
        for x_exe in xs_exe[:-1]:
            ax.axvline(x=x_exe[0,0], ymax=0.1, c='orange')
        ax.axvline(x=xs_exe[-1][0,0], ymax=0.1, c='orange', label='Sample Min Emit')
        ax.axvline(x=0, ymax=0.2, c='r', label='True Min Emit')

        if x_next is not None:
            ax.scatter(x_next[0,0], x_next[0,1], marker='x', s=150, c='r', label='Max EIG')

        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


        ax.set_xlabel('Tuning Param')
        ax.set_ylabel('Measurement Quad')
        ax.set_title('Expected Information Gain')
    #     ax.legend()


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes
        ax = axes[0,2]


        ax.plot(x_mesh_tuple[0][:,0], gt_emits*1.e6, c='r', label='Ground Truth')

        ax.plot(x_mesh_tuple[0][:,0], emits_flat[0], c='b', alpha=0.1, label='Samples')
        for emits in emits_flat[1:]:
            ax.plot(x_mesh_tuple[0][:,0], emits, c='b', alpha=0.1)

        ax.set_ylim(-1,10)
        ax.set_xlabel('Tuning Param')
        ax.set_ylabel('Emittance')
        ax.set_title('Emittance Results')
        ax.legend()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes
        ax = axes[1,2]

        slice = 25 #vertical slice (i.e. along the measurement scan direction) of above plots indexed from left to right
        tuning_param = x_mesh_tuple[0][slice,0]


        ax.plot(x_mesh_tuple[1][slice,:], y_mesh[slice,:]*1.e6,c='r', label='Ground Truth')

        sample = s[0]
        ax.plot(x_mesh_tuple[1][slice,:], sample[slice,:], c='b', alpha=0.1, label='Samples')
        for sample in s[1:]:
            ax.plot(x_mesh_tuple[1][slice,:], sample[slice,:], c='b', alpha=0.1)



        ax.set_xlabel('Measurement Quad')
        ax.set_ylabel('Beam Size Squared')
        ax.set_title('Quad Scan for True Opt. Tuning Param')
        ax.legend()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes
        ax = axes[0,3]


        ax.hist(xs_exe[:,0,0], bins=torch.linspace(-2,2,22))
        ax.axvline(x=0, c='r', label='Ground Truth')

        ax.set_xlabel('Tuning Param')
        ax.set_ylabel('Frequency')
        ax.set_title('Predicted Optimal Tuning Param')
        ax.set_xlim(-2,2)
        ax.legend()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #axes
        ax = axes[1,3]


        ax.hist(emits_flat[:,slice], bins=torch.linspace(0,50,51))
        ax.axvline(x=1.9365, c='r', label='Ground Truth')

        ax.set_xlim(0,50)
        ax.set_xlabel('Emittance')
        ax.set_ylabel('Frequency')
        ax.set_title('Predicted Emittance for True Opt. Tuning Param')
        ax.legend()

        plt.tight_layout()
        plt.show()
#         plt.savefig('IterPlot2d_BAX_'+str(len(x_obs)), format='pdf')




        #~~~~~~~~~~~~~~~~
        # #find grid point with greatest expected information gain
        # idmax = torch.argmax(eig.reshape(1,-1))
        # x_best = xs[idmax]
        # print('Input with highest EIG: \n', 'x_best =', x_best)
    
    
    