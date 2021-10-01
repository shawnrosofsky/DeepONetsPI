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
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from layers import MLP, modified_MLP, FF_MLP
from utils import RBF
from DataGenerator import DataGenerator
from model import DeepONetPI




class DRDeepONetPI(DeepONetPI):
    def __init__(self,
                 branch_layers, 
                 trunk_layers, 
                 branch_net=MLP,
                 trunk_net=MLP,
                 branch_activation=jnp.tanh,
                 trunk_activation=jnp.tanh,
                 optimizer=adam(exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9))
                 ):
        super().__init__(branch_layers, trunk_layers, branch_net, trunk_net, branch_activation, trunk_activation, optimizer)
        
    def pde_net(self, params, u, y):
        # note that y here can include additional coord not passed to the original operator network
        # print(y.shape)
        x = y[0]
        t = y[1]
        s_ = self.operator_net # shorthand for operator_net function call
        s = s_(params, u, x, t) # actual value of s (may not be needed)
        s_x = grad(s_, 2)(params, u, x, t) 
        s_t = grad(s_, 3)(params, u, x, t)
        s_xx = grad(grad(s_, 2), 2)(params, u, x, t)
        res = s_t - 0.01 * s_xx - 0.01 * s**2 # this is equal to u(x), this avoids the interpolation
        return res
    
    # here we are provided with values for BC/IC (zero), so for this case just call the operator net.  If we had Robin BC for example, we would output an array with outputs of [value, derivative]
    # Also, if we had the case where BC/IC = u, we would return res = s - u
    def bc_net(self, params, u, y):
        s_ = self.operator_net # shorthand for operator_net function call
        s = s_(params, u, y) # actual value of s
        return s
    
    def ic_net(self, params, u, y):
        s_ = self.operator_net # shorthand for operator_net function call
        s = s_(params, u, y) # actual value of s
        return s
    




# Use double precision to generate data (due to GP sampling)

# A diffusion-reaction numerical solver
def solve_ADR(key, Nx, Nt, N_op, length_scale):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01*jnp.ones_like(x)
    v = lambda x: jnp.zeros_like(x)
    g = lambda u: 0.01*u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: jnp.zeros_like(x)

    # Generate subkeys
    subkeys = random.split(key, 2)

    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (N,)))
    # Create a callable interpolation function  
    f_fn = lambda x: jnp.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = jnp.eye(Nx, k=1) - jnp.eye(Nx, k=-1)
    D2 = -2 * jnp.eye(Nx) + jnp.eye(Nx, k=-1) + jnp.eye(Nx, k=1)
    D3 = jnp.eye(Nx - 2)
    M = -jnp.diag(D1 @ k) @ D1 - 4 * jnp.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * jnp.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * jnp.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx, Nt))
    u = index_update(u, index[:,0], u0(x))
    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = jnp.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u = index_update(u, index[1:-1, i + 1], jnp.linalg.solve(A, b1 + b2))
        return u
    # Run loop
    S = lax.fori_loop(0, Nt-1, body_fn, u)

    # Input sensor locations and measurements
    xx = jnp.linspace(xmin, xmax, m)
    u = f_fn(xx)
    # Output sensor locations and measurements
    idx = random.randint(subkeys[1], (N_op,2), 0, max(Nx,Nt))
    y = jnp.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
    s = S[idx[:,0], idx[:,1]]
    # x, t: sampled points on grid
    return (x, t, S), (u, y, s)

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, N_op, N_pde, N_bcs, N_ics):
    # Numerical solution
    (x, t, S), (u, y, s) = solve_ADR(key, Nx , Nt, N_op, length_scale)

    # Geneate subkeys
    subkeys = random.split(key, 4)
    
    # Sample the data
    u_op = jnp.tile(u, (N_op, 1))
    y_op = y
    s_op = s
    
    # Generate data for PDE constraints
    # Sample collocation points
    # Chooses N_pde random points in domain of (x, t).  We don't need to have a correct prediction for solution s, only that our prediction by the ANN satisfies the pde at the specified (x,t). For s, we take set s(x) = u(x) assuming that u is defined at each coordinate x (guarenteed in solve_ADR).
    x_pde_idx = random.choice(subkeys[0], jnp.arange(Nx), shape=(N_pde,1))
    x_pde = x[x_pde_idx]
    t_pde = random.uniform(subkeys[1], minval=0, maxval=1, shape=(N_pde,1))

    u_pde = jnp.tile(u, (N_pde, 1))
    y_pde = jnp.hstack((x_pde, t_pde))
    s_pde = u[x_pde_idx]
    
    # Sample points from the boundary conditions
    # handle odd number of bc
    N_bc_half1 = N_bcs // 2
    N_bc_half2 = N_bcs - N_bc_half1
    x_bc1 = jnp.zeros((N_bc_half1, 1))
    x_bc2 = jnp.ones((N_bc_half2, 1))
    x_bcs = jnp.vstack((x_bc1, x_bc2))
    t_bcs = random.uniform(subkeys[2], shape=(N_bcs, 1))
    
    u_bcs = jnp.tile(u, (N_bcs, 1))
    y_bcs = jnp.hstack([x_bcs, t_bcs])
    s_bcs = jnp.zeros((N_bcs, 1))
    
    # Sample points from initial conditions
    x_ics = random.uniform(subkeys[3], shape=(N_ics, 1))
    t_ics = jnp.zeros((N_ics, 1))

    u_ics = jnp.tile(u, (N_ics, 1))
    y_ics = jnp.hstack([x_ics, t_ics])
    s_ics = jnp.zeros((N_ics, 1))
    
    
    # Shorthand tuples to pass fewer outputs
    train_op = (u_op, y_op, s_op)
    train_pde = (u_pde, y_pde, s_pde)
    train_bcs = (u_bcs, y_bcs, s_bcs)
    train_ics = (u_ics, y_ics, s_ics)
    

    return train_op, train_pde, train_bcs, train_ics

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, N_op):
    Nx = N_op
    Nt = N_op
    (x, t, S), (u, y, s) = solve_ADR(key, Nx , Nt, N_op, length_scale)

    XX, TT = jnp.meshgrid(x, t)

    u_test = jnp.tile(u, (N_op**2,1))
    y_test = jnp.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = S.T.flatten()

    return u_test, y_test, s_test

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, N_op, N_pde, N_ics, N_bcs):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    train_op, train_pde, train_bcs, train_ics = vmap(generate_one_training_data, (0, None, None, None, None))(keys, N_op, N_pde, N_bcs, N_ics)
    
    u_op, y_op, s_op = train_op
    u_pde, y_pde, s_pde = train_pde
    u_bcs, y_bcs, s_bcs = train_bcs
    u_ics, y_ics, s_ics = train_ics

    u_op = jnp.float32(u_op.reshape(N * N_op,-1))
    y_op = jnp.float32(y_op.reshape(N * N_op,-1))
    s_op = jnp.float32(s_op.reshape(N * N_op,-1))

    u_pde = jnp.float32(u_pde.reshape(N * N_pde,-1))
    y_pde = jnp.float32(y_pde.reshape(N * N_pde,-1))
    s_pde = jnp.float32(s_pde.reshape(N * N_pde,-1))

    u_bcs = jnp.float32(u_bcs.reshape(N * N_bcs,-1))
    y_bcs = jnp.float32(y_bcs.reshape(N * N_bcs,-1))
    s_bcs = jnp.float32(s_bcs.reshape(N * N_bcs,-1))

    u_ics = jnp.float32(u_ics.reshape(N * N_ics,-1))
    y_ics = jnp.float32(y_ics.reshape(N * N_ics,-1))
    s_ics = jnp.float32(s_ics.reshape(N * N_ics,-1))
    
    config.update("jax_enable_x64", False)
    
    # Shorthand tuples to pass fewer outputs
    train_op = (u_op, y_op, s_op)
    train_pde = (u_pde, y_pde, s_pde)
    train_bcs = (u_bcs, y_bcs, s_bcs)
    train_ics = (u_ics, y_ics, s_ics)


    return train_op, train_pde, train_bcs, train_ics

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, N_op):

    config.update("jax_enable_x64", True)
    keys = random.split(key, N)

    u_test, y_test, s_test = vmap(generate_one_test_data, (0, None))(keys, N_op)

    u_test = jnp.float32(u_test.reshape(N * N_op**2,-1))
    y_test = jnp.float32(y_test.reshape(N * N_op**2,-1))
    s_test = jnp.float32(s_test.reshape(N * N_op**2,-1))

    config.update("jax_enable_x64", False)
    return u_test, y_test, s_test

# Compute relative l2 error over N_op test samples.
def compute_error(key, N_op):
    # Generate one test sample
    u_test, y_test, s_test = generate_test_data(key, 1, N_op)
    # Predict  
    s_pred = model.predict_s(params, u_test, y_test)[:,None]
    # Compute relative l2 error
    error_s = jnp.linalg.norm(s_test - s_pred) / jnp.linalg.norm(s_test) 
    return error_s

if __name__ == "__main__":
    key = random.PRNGKey(0)

    # GRF length scale
    length_scale = 0.2

    # Resolution of the solution
    Nx = 100
    Nt = 100

    N = 500 # number of input samples (different u values)
    m = Nx   # number of input sensors
    N_op_train = 100 # number data outputs per sample 
    N_pde_train = 100  # number of points for PDE constrains pe sample
    N_bcs_train = 100 # number of BC points 
    N_ics_train = 100 # number of IC points
    train_op, train_pde, train_bcs, train_ics = generate_training_data(key, N, N_op_train, N_pde_train, N_bcs_train, N_ics_train)
    
    u_op, y_op, s_op = train_op
    u_pde, y_pde, s_pde = train_pde
    u_bcs, y_bcs, s_bcs = train_bcs
    u_ics, y_ics, s_ics = train_ics
    
    
    # Initialize model
    branch_layers = [m, 50, 50, 50, 50, 50]
    trunk_layers =  [2, 50, 50, 50, 50, 50]
    model = DRDeepONetPI(branch_layers, trunk_layers)
    
    # Create data set
    batch_size = 10000
    op_dataset = DataGenerator(u_op, y_op, s_op, batch_size)
    pde_dataset = DataGenerator(u_pde, y_pde, s_pde, batch_size)
    bcs_dataset = DataGenerator(u_bcs, y_bcs, s_bcs, batch_size)
    ics_dataset = DataGenerator(u_ics, y_ics, s_ics, batch_size)
    
    # Train
    model.train(op_dataset, pde_dataset, bcs_dataset, ics_dataset, nIter=10000)

    # Test data
    N_test = 100 # number of input samples 
    m_test = m   # number of sensors 
    key_test = random.PRNGKey(1234567)
    keys_test = random.split(key_test, N_test)

    # Predict
    params = model.get_params(model.opt_state)

    # Compute error
    error_s = vmap(compute_error, (0, None))(keys_test, m_test) 

    print('mean of relative L2 error of s: {:.2e}'.format(error_s.mean()))
    print('std of relative L2 error of s: {:.2e}'.format(error_s.std()))
    
    #Plot for loss function
    plt.figure(figsize = (6,5))
    plt.plot(model.loss_operator_log, lw=2, label='operator')
    plt.plot(model.loss_physics_log, lw=2, label='physics')
    plt.plot(model.loss_bcs_log, lw=2, label='bcs')
    plt.plot(model.loss_ics_log, lw=2, label='ics')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Generate one test sample
    key = random.PRNGKey(12345)
    P_test = 100
    Nx = m
    u_test, y_test, s_test = generate_test_data(key, 1, P_test)

    # Predict
    params = model.get_params(model.opt_state)
    s_pred = model.predict_s(params, u_test, y_test)

    # Generate an uniform mesh
    x = jnp.linspace(0, 1, Nx)
    t = jnp.linspace(0, 1, Nt)
    XX, TT = jnp.meshgrid(x, t)

    # Grid data
    S_pred = griddata(y_test, s_pred.flatten(), (XX,TT), method='cubic')
    S_test = griddata(y_test, s_test.flatten(), (XX,TT), method='cubic')

    # Compute the relative l2 error 
    error = jnp.linalg.norm(S_pred - S_test, 2) / jnp.linalg.norm(S_test, 2) 
    print('Relative l2 errpr: {:.3e}'.format(error))
    
    # Plot
    fig = plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.pcolor(XX,TT, S_test, cmap='seismic')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact $s(x,t)$')
    plt.tight_layout()

    plt.subplot(1,3,2)
    plt.pcolor(XX,TT, S_pred, cmap='seismic')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$')
    plt.tight_layout()

    plt.subplot(1,3,3)
    plt.pcolor(XX,TT, S_pred - S_test, cmap='seismic')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()