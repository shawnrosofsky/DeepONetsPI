import jax
import os
import jax.numpy as jnp
import numpy as np
import torch
from jax import random, grad, vmap, jit, hessian
from jax.experimental import optimizers
from jax.experimental.optimizers import adam, exponential_decay
from jax.experimental.ode import odeint
from jax.nn import relu, elu, softplus
from jax.config import config
from jax.ops import index_update, index
from jax import lax
from jax.lax import while_loop, scan, cond
from jax.flatten_util import ravel_pytree
import flax

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import traceback
import sys



from layers import MLP, FlaxMLP

from scipy.interpolate import griddata


# Define the model
class DeepONetPI:
    def __init__(self, 
                 branch_layers, 
                 trunk_layers, 
                 branch_net=MLP,
                 trunk_net=MLP,
                 branch_activation=jnp.tanh,
                 trunk_activation=jnp.tanh,
                 branch_rng_key=random.PRNGKey(1234),
                 trunk_rng_key=random.PRNGKey(4321),
                 optimizer=adam(exponential_decay(1e-3, decay_steps=5000,decay_rate=0.9)),
                 operator_loss_const=1.0,
                 physics_loss_const=1.0,
                 bcs_loss_const=1.0,
                 ics_loss_const=1.0,
                 ckpt_dir='DeepONetPI',
                 ckpt_file='params.npy',
                 loss_file='loss.npy',
                 loss_operator_file='loss_operator.npy',
                 loss_physics_file='loss_physics.npy',
                 loss_bcs_file='loss_bcs.npy',
                 loss_ics_file='loss_ics.npy',      
                 ):
        # Network initialization and evaluation functions
        # These are reserved for the flax case
        self.branch_net = None
        self.trunk_net = None
        # Branch network
        self.branch_init, self.branch_apply = branch_net(branch_layers, activation=branch_activation)
        # Trunk network
        self.trunk_init, self.trunk_apply = trunk_net(trunk_layers, activation=trunk_activation)

        # Initialize
        branch_params = self.branch_init(rng_key=branch_rng_key)
        trunk_params = self.trunk_init(rng_key=trunk_rng_key)
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.opt_state = self.opt_init(params)

        # Initialize itercounter
        self.itercount = itertools.count()
        
        # Get Number of coordinate dimensions
        self.dim = trunk_layers[0]
        
         # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)
        
        
        # Define loss constants used to weight the loss
        self.operator_loss_const = operator_loss_const
        self.physics_loss_const = physics_loss_const
        self.bcs_loss_const = bcs_loss_const
        self.ics_loss_const = ics_loss_const
        
        # Loggers
        self.loss_log = []
        self.loss_operator_log = []
        self.loss_physics_log = []
        self.loss_bcs_log = []
        self.loss_ics_log = []
        
        # Checkpointing file names
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)
        self.loss_path = os.path.join(self.ckpt_dir, loss_file)
        self.loss_operator_path = os.path.join(self.ckpt_dir, loss_operator_file)
        self.loss_physics_path = os.path.join(self.ckpt_dir, loss_physics_file)
        self.loss_bcs_path = os.path.join(self.ckpt_dir, loss_bcs_file)
        self.loss_ics_path = os.path.join(self.ckpt_dir, loss_ics_file)

        
    # Define DeepONet architecture
    def operator_net(self, params, u, *y):
        # Apply DeepONet
        # inputs: (u, *y), shape = (N, m), (N, 1) * dim or (N, dim)
        # outputs: s, shape = (N, 1)
        # Note that each coordinate dimension can be a separate input or a single array. This is done to make differentiation easier. In future we may allow u to be a tuple of shapes (N, m_i) * len(u) to allow for multiple branch networks to represent ics, bcs, 
        
        y = jnp.stack(y)
        branch_params, trunk_params = params
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = jnp.sum(B * T)
        return outputs
    
    # Define ODE/PDE residual (initial definition)
    def pde_net(self, params, u, y):
        # s_ = self.operator_net # shorthand for operator_net function call
        # s = s_(params, u, y) # actual value of s (may not be needed)
        # # optionally split s up here       
        # s_y = grad(s_, 2)(params, u, y) # the 2nd argument in grad specifies the argument number, here y is 2 for example
        # res = 0.0 # here we output 0 because we want the user to specify the pde
        # return res
        pass
    
    # Define boundary condition (initial definition)
    def bc_net(self, params, u, y):
        pass
    
    # Define initial condition (initial definition)
    def ic_net(self, params, u, y):
        pass
    
    # Define operator loss
    def loss_operator(self, params, batch):
        # Fetch data
        # inputs: (u, y), shape = (N, m), (N, dim)
        # outputs: s, shape = (N, 1)
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.operator_net, (None, 0, 0))(params, u, y)
        # Compute loss
        loss = jnp.mean((outputs.flatten() - pred.flatten())**2)
        return loss

    # Define physics loss
    def loss_physics(self, params, batch):
        # Fetch data
        # inputs: (u_pde, y_pde), shape = (N_pde, m), (N_pde, dim)
        # outputs: s_pde, shape = (N_pde, 1)
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.pde_net, (None, 0, 0))(params, u, y)
        # Compute loss
        loss = jnp.mean((outputs.flatten() - pred.flatten())**2)
        return loss
    
    def loss_bcs(self, params, batch):
        # Fetch data
        # inputs: (u_bc, y_bc), shape = (N_bc, m), (N_bc, dim)
        # outputs: s_bc, shape = (N_bc, 1)
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.bc_net, (None, 0, 0))(params, u, y)
        # Compute loss
        loss = jnp.mean((outputs.flatten() - pred.flatten())**2)
        return loss
    
    def loss_ics(self, params, batch):
        # Fetch data
        # inputs: (u_ic, y_ic), shape = (N_ic, m), (N_ic, dim)
        # outputs: s_ic, shape = (N_ic, 1)
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.ic_net, (None, 0, 0))(params, u, y)
        # Compute loss
        loss = jnp.mean((outputs.flatten() - pred.flatten())**2)
        return loss
    
    # Define total loss
    def loss(self, params, operator_batch=None, physics_batch=None, bcs_batch=None, ics_batch=None):
        loss = loss_operator = loss_physics = loss_bcs = loss_ics = 0.0
        # lax.cond(operator_batch is not None, )
        if operator_batch is not None:
            loss_operator = self.loss_operator(params, operator_batch)
        if physics_batch is not None:
            loss_physics = self.loss_physics(params, physics_batch)
        if bcs_batch is not None:
            loss_bcs = self.loss_bcs(params, bcs_batch)
        if ics_batch is not None:
            loss_ics = self.loss_ics(params, ics_batch)
        # losses = jnp.array([loss_operator, loss_physics, loss_bcs, loss_ics])
        # loss = jnp.nansum(losses)
        
        loss = self.operator_loss_const*loss_operator + self.physics_loss_const*loss_physics + self.bcs_loss_const*loss_bcs + self.ics_loss_const*loss_ics
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, operator_batch=None, physics_batch=None, bcs_batch=None, ics_batch=None):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, operator_batch, physics_batch, bcs_batch, ics_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, 
              operator_dataset=None, 
              physics_dataset=None, 
              bcs_dataset=None, 
              ics_dataset=None,
              operator_val_dataset=None, 
              physics_val_dataset=None,
              bcs_val_dataset=None,
              ics_val_dataset=None,
              nIter=10000, 
              log_freq=10, 
              val_freq=10,
              ckpt_freq=1000,
              history_freq=1000):
        # Define the data iterator
        operator_data = physics_data = ics_data = bcs_data = None
        if operator_dataset is not None:
            operator_data = iter(operator_dataset)
        if physics_dataset is not None:
            physics_data = iter(physics_dataset)
        if bcs_dataset is not None:
            bcs_data = iter(bcs_dataset)
        if ics_dataset is not None:
            ics_data = iter(ics_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            operator_batch = physics_batch = bcs_batch = ics_batch = None
            if operator_data is not None:
                operator_batch = next(operator_data)
            if physics_data is not None:
                physics_batch = next(physics_data)
            if bcs_data is not None:
                bcs_batch = next(bcs_data)
            if ics_data is not None:
                ics_batch = next(ics_data)
                
            self.opt_state = self.step(next(self.itercount), self.opt_state, operator_batch, physics_batch, bcs_batch, ics_batch)
            
            if (log_freq != 0) and (it % log_freq == 0):
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, operator_batch, physics_batch, bcs_batch, ics_batch)
                
                if operator_batch is not None:
                    loss_operator_value = self.loss_operator(params, operator_batch)
                else:
                    loss_operator_value = None
                if physics_batch is not None:
                    loss_physics_value = self.loss_physics(params, physics_batch)
                else:
                    loss_physics_value = None
                if bcs_batch is not None:
                    loss_bcs_value = self.loss_bcs(params, bcs_batch)
                else:
                    loss_bcs_value = None
                if ics_batch is not None:
                    loss_ics_value = self.loss_ics(params, ics_batch)
                else:
                    loss_ics_value = None
                

                # Store losses
                loss_dict = {} # for printing losses in pbar
                if loss_value is not None:
                    self.loss_log.append(loss_value)
                    loss_dict['loss'] = loss_value                    
                if loss_operator_value is not None:
                    self.loss_operator_log.append(loss_operator_value)
                    loss_dict['loss_operator'] = loss_operator_value
                if loss_physics_value is not None:
                    self.loss_physics_log.append(loss_physics_value)
                    loss_dict['loss_physics'] = loss_physics_value
                if loss_bcs_value is not None:
                    self.loss_bcs_log.append(loss_bcs_value)
                    loss_dict['loss_bcs'] = loss_bcs_value
                if loss_ics_value is not None:
                    self.loss_ics_log.append(loss_ics_value)
                    loss_dict['loss_ics'] = loss_ics_value

                # Print losses during training
                pbar.set_postfix(loss_dict)
            
            if (ckpt_freq != 0) and (it % ckpt_freq == 0):
                # may want to add an iteration number to ckpt_path in future
                self.save(ckpt_path=self.ckpt_path)
                
            if (history_freq != 0) and (it % history_freq == 0):
                # may want to add an itteration number to the loss logs in future
                self.save_history()
                
    def save(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.ckpt_path
        ckpt_dir = os.path.split(ckpt_path)[0]
        os.makedirs(ckpt_dir, exist_ok=True)
        params = self.get_params(self.opt_state)
        flat_params = flat_params, _  = ravel_pytree(params)
        jnp.save(ckpt_path, flat_params)
        
    def restore(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.ckpt_path
        try:
            flat_params = jnp.load(ckpt_path)
        except:
            print(f'Failed to load file {ckpt_path}')
            traceback.print_exc()
            return
        params = self.unravel_params(flat_params)
        self.opt_state = self.opt_init(params)

    def save_history(self, loss_path=None, loss_operator_path=None, loss_physics_path=None, loss_bcs_path=None, loss_ics_path=None):
        if loss_path is None:
            loss_path = self.loss_path
        if loss_operator_path is None:
            loss_operator_path = self.loss_operator_path
        if loss_physics_path is None:
            loss_physics_path = self.loss_physics_path
        if loss_bcs_path is None:
            loss_bcs_path = self.loss_bcs_path
        if loss_ics_path is None:
            loss_ics_path = self.loss_ics_path
        
        loss_dir = os.path.split(loss_path)[0]
        loss_operator_dir = os.path.split(loss_operator_path)[0]
        loss_physics_dir = os.path.split(loss_physics_path)[0]
        loss_bcs_dir = os.path.split(loss_bcs_path)[0]
        loss_ics_dir = os.path.split(loss_ics_path)[0]
        
        if self.loss_log:
            os.makedirs(loss_dir, exist_ok=True)
            jnp.save(loss_path, self.loss_log)
        if self.loss_operator_log:
            os.makedirs(loss_operator_dir, exist_ok=True)
            jnp.save(loss_operator_path, self.loss_operator_log)
        if self.loss_physics_log:
            os.makedirs(loss_physics_dir, exist_ok=True)
            jnp.save(loss_physics_path, self.loss_physics_log)
        if self.loss_bcs_log:
            os.makedirs(loss_bcs_dir, exist_ok=True)
            jnp.save(loss_bcs_path, self.loss_bcs_log)
        if self.loss_ics_log:
            os.makedirs(loss_ics_dir, exist_ok=True)
            jnp.save(loss_ics_path, self.loss_ics_log)
            
    def restore_history(self, loss_path=None, loss_operator_path=None, loss_physics_path=None, loss_bcs_path=None, loss_ics_path=None):
        if loss_path is None:
            loss_path = self.loss_path
        if loss_operator_path is None:
            loss_operator_path = self.loss_operator_path
        if loss_physics_path is None:
            loss_physics_path = self.loss_physics_path
        if loss_bcs_path is None:
            loss_bcs_path = self.loss_bcs_path
        if loss_ics_path is None:
            loss_ics_path = self.loss_ics_path
        
        # Try to load the loss files and add them to loss logs.  It is ok if they fail beacause the loss files might not exist.
        try:
            loss_log = jnp.load(loss_path)
            self.loss_log = list(loss_log)
        except:
            print(f'Failed to load file {loss_path}')
        try:
            loss_operator_log = jnp.load(loss_operator_path)
            self.loss_operator_log = list(loss_operator_log)
        except:
            print(f'Failed to load file {loss_operator_path}')
        try:
            loss_physics_log = jnp.load(loss_physics_path)
            self.loss_physics_log = list(loss_physics_log)
        except:
            print(f'Failed to load file {loss_physics_path}')
        try:
            loss_bcs_log = jnp.load(loss_bcs_path)
            self.loss_bcs_log = list(loss_bcs_log)
        except:
            print(f'Failed to load file {loss_bcs_path}')
        try:
            loss_ics_log = jnp.load(loss_ics_path)
            self.loss_ics_log = list(loss_ics_log)
        except:
            print(f'Failed to load file {loss_ics_path}')
       
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0))(params, U_star, Y_star)
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_pde(self, params, U_star, Y_star):
        pde_pred = vmap(self.pde_net, (None, 0, 0))(params, U_star, Y_star)
        return pde_pred
    
    
# Model for flax

class FlaxDeepONetPI(DeepONetPI):
    def __init__(self, 
                 branch_layers, 
                 trunk_layers, 
                 branch_net=FlaxMLP,
                 trunk_net=FlaxMLP,
                 branch_activation=jnp.tanh,
                 trunk_activation=jnp.tanh,
                 branch_rng_key=random.PRNGKey(1234),
                 trunk_rng_key=random.PRNGKey(4321),
                 optimizer=adam(exponential_decay(1e-3, decay_steps=5000,decay_rate=0.9)),
                 operator_loss_const=1.0,
                 physics_loss_const=1.0,
                 bcs_loss_const=1.0,
                 ics_loss_const=1.0,
                 ckpt_dir='DeepONetPI',
                 ckpt_file='params.npy',
                 loss_file='loss.npy',
                 loss_operator_file='loss_operator.npy',
                 loss_physics_file='loss_physics.npy',
                 loss_bcs_file='loss_bcs.npy',
                 loss_ics_file='loss_ics.npy',      
                 ):
        super().__init__(branch_layers, 
                         trunk_layers, 
                         MLP, # branch network placeholder
                         MLP, # trunk network placeholder
                         branch_activation, 
                         trunk_activation, 
                         branch_rng_key, 
                         trunk_rng_key, 
                         optimizer, 
                         operator_loss_const, 
                         physics_loss_const, 
                         bcs_loss_const, 
                         ics_loss_const, 
                         ckpt_dir, 
                         ckpt_file, 
                         loss_file, 
                         loss_operator_file, 
                         loss_physics_file, 
                         loss_bcs_file, 
                         loss_ics_file)
        # Need to reconfigure netorks
        # Network initialization and evaluation functions
        self.branch_net = branch_net(branch_layers, branch_activation)
        self.trunk_net = trunk_net(trunk_layers, trunk_activation)
        # These are for the no flax case
        # Branch network
        self.branch_init, self.branch_apply = None, None
        # Trunk network
        self.trunk_init, self.trunk_apply = None, None

        # Initialize
        branch_init_shape = jnp.ones((1, branch_layers[0]))
        trunk_init_shape = jnp.ones((1, trunk_layers[0]))
        branch_params = self.branch_net.init(branch_rng_key, branch_init_shape)
        trunk_params = self.trunk_net.init(trunk_rng_key, trunk_init_shape)
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)
        
        # Define DeepONet architecture
    def operator_net(self, params, u, *y):
        # Apply DeepONet
        # inputs: (u, *y), shape = (N, m), (N, 1) * dim or (N, dim)
        # outputs: s, shape = (N, 1)
        # Note that each coordinate dimension can be a separate input or a single array. This is done to make differentiation easier. In future we may allow u to be a tuple of shapes (N, m_i) * len(u) to allow for multiple branch networks to represent ics, bcs, 
        
        y = jnp.stack(y)
        branch_params, trunk_params = params
        B = self.branch_net.apply(branch_params, u)
        T = self.trunk_net.apply(trunk_params, y)
        outputs = jnp.sum(B * T)
        return outputs