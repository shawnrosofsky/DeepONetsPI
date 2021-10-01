import jax
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

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm


from layers import MLP, modified_MLP, FF_MLP

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
                 optimizer=adam(exponential_decay(1e-3, decay_steps=5000,decay_rate=0.9)),
                 operator_loss_const=1.0,
                 physics_loss_const=1.0,
                 bcs_loss_const=1.0,
                 ics_loss_const=1.0
                 ):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = branch_net(branch_layers, activation=branch_activation)
        # Apply Fourier feature network to the trunk net
        self.trunk_init, self.trunk_apply = trunk_net(trunk_layers, activation=trunk_activation)

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()
        
         # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)
        
        # Define constraint functions
        # if pde is not None:
        #     self.pde_net = pde
        # if bc is not None:
        #     self.bc_net = bc
        # if ic is not None:
        #     self.ic_net = ic
        
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
        # print(f"outputs: {outputs}")
        # print(f"pred: {pred}")
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
    def train(self, operator_dataset=None, 
              physics_dataset=None, 
              bcs_dataset=None, 
              ics_dataset=None, 
              nIter=10000, 
              log_freq=10, 
              val_freq=10):
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
            
            if it % log_freq == 0:
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
       
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0))(params, U_star, Y_star)
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_pde(self, params, U_star, Y_star):
        pde_pred = vmap(self.pde_net, (None, 0, 0))(params, U_star, Y_star)
        return pde_pred