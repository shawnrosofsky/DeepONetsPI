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


from DataGenerator import DataGenerator
from model import DeepONetPI


class AdvDeepONetPI(DeepONetPI):
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
        ux = y[2]
        s_ = self.operator_net # shorthand for operator_net function call
        s = s_(params, u, x, t) # actual value of s (may not be needed)
        s_x = grad(s_, 2)(params, u, x, t) 
        s_t = grad(s_, 3)(params, u, x, t)
        res = s_t + ux * s_x
        return res




# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


# Deinfe initial and boundary conditions for advection equation
# IC: f(x, 0)  = sin(pi x)
# BC: g(0, t) = sin (pi t / 2) 
f = lambda x: jnp.sin(jnp.pi * x)
g = lambda t: jnp.sin(jnp.pi * t/2)

# Advection solver 
def solve_CVC(key, gp_sample, Nx, Nt, m, P):
    # Solve u_t + a(x) * u_x = 0
    # Wendroff for a(x)=V(x) - min(V(x)+ + 1.0, u(x,0)=f(x), u(0,t)=g(t)  (f(0)=g(0))
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    
    N = gp_sample.shape[0]
    X = jnp.linspace(xmin, xmax, N)[:,None]
    V = lambda x: jnp.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h

    # Compute advection velocity
    v_fn = lambda x: V(x) - V(x).min() + 1.0
    v =  v_fn(x)

    # Initialize solution and apply initial & boundary conditions
    u = jnp.zeros([Nx, Nt])
    u = index_update(u, index[0, :], g(t))
    u = index_update(u, index[:, 0], f(x))
    
    # Compute finite difference operators
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = jnp.eye(Nx - 1, k=0)
    K_temp = jnp.eye(Nx - 1, k=0)
    Trans = jnp.eye(Nx - 1, k=-1)
    def body_fn_x(i, carry):
        K, K_temp = carry
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
        return K, K_temp
    K, _ = lax.fori_loop(0, Nx-2, body_fn_x, (K, K_temp))
    D = jnp.diag(k) + jnp.eye(Nx - 1, k=-1)
    
    def body_fn_t(i, u):
        b = jnp.zeros(Nx - 1)
        b = index_update(b, index[0], g(i * dt) - k[0] * g((i + 1) * dt))
        u = index_update(u, index[1:, i + 1], K @ (D @ u[1:, i] + b))
        return u
    UU = lax.fori_loop(0, Nt-1, body_fn_t, u)

    # Input sensor locations and measurements
    xx = jnp.linspace(xmin, xmax, m)
    u = v_fn(xx)
    # Output sensor locations and measurements
    idx = random.randint(key, (P,2), 0, max(Nx,Nt))
    y = jnp.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
    s = UU[idx[:,0], idx[:,1]]

    return (x, t, UU), (u, y, s)

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, P, Q):

    subkeys = random.split(key, 10)
    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (N,)))

    v_fn = lambda x: jnp.interp(x, X.flatten(), gp_sample)
    u_fn = lambda x: v_fn(x) - v_fn(x).min() + 1.0

    (x, t, UU), (u, y, s) = solve_CVC(subkeys[1], gp_sample, Nx, Nt, m, P)

    x_bc1 = jnp.zeros((P // 2, 1))
    x_bc2 = random.uniform(subkeys[2], shape = (P // 2, 1))
    x_bcs = jnp.vstack((x_bc1, x_bc2))

    t_bc1 = random.uniform(subkeys[3], shape = (P//2, 1))
    t_bc2 = jnp.zeros((P//2, 1))
    t_bcs = jnp.vstack([t_bc1, t_bc2])

    u_train = jnp.tile(u, (P, 1))
    y_train = jnp.hstack([x_bcs, t_bcs])

    s_bc1 = g(t_bc1)
    s_bc2 = f(x_bc2)
    s_train =  jnp.vstack([s_bc1, s_bc2])

    x_r = random.uniform(subkeys[4], shape=(Q,1), minval=xmin, maxval=xmax)
    t_r = random.uniform(subkeys[5], shape=(Q,1), minval=tmin, maxval=tmax)
    ux_r = u_fn(x_r)

    u_r_train = jnp.tile(u, (Q,1))
    y_r_train = jnp.hstack([x_r, t_r, ux_r])
    s_r_train = jnp.zeros((Q, 1))
    
    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, Nx, Nt, P):
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))

    (x, t, UU), (u, y, s) = solve_CVC(key, gp_sample, Nx, Nt, m, P)

    XX, TT = jnp.meshgrid(x, t)

    u_test = jnp.tile(u, (Nx*Nt,1))
    y_test = jnp.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = UU.T.flatten()

    return u_test, y_test, s_test

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P, Q):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    u_train, y_train, s_train, u_r_train, y_r_train, s_r_train = vmap(generate_one_training_data, (0, None, None))(keys, P, Q)

    u_train = jnp.float32(u_train.reshape(N * P,-1))
    y_train = jnp.float32(y_train.reshape(N * P,-1))
    s_train = jnp.float32(s_train.reshape(N * P,-1))

    u_r_train = jnp.float32(u_r_train.reshape(N * Q,-1))
    y_r_train = jnp.float32(y_r_train.reshape(N * Q,-1))
    s_r_train = jnp.float32(s_r_train.reshape(N * Q,-1))

    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train, u_r_train, y_r_train,  s_r_train

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, Nx, Nt, P):

    config.update("jax_enable_x64", True)
    keys = random.split(key, N)

    u_test, y_test, s_test = vmap(generate_one_test_data, (0, None, None, None))(keys, Nx, Nt, P)

    u_test = jnp.float32(u_test.reshape(N * Nx * Nt,-1))
    y_test = jnp.float32(y_test.reshape(N * Nx * Nt,-1))
    s_test = jnp.float32(s_test.reshape(N * Nx * Nt,-1))

    config.update("jax_enable_x64", False)
    return u_test, y_test, s_test

# Compute relative l2 error over N test samples.
def compute_error(key, Nx, Nt, P):
    u_test, y_test, s_test = generate_test_data(key, 1, Nx, Nt, P)
    s_pred = model.predict_s(params, u_test, y_test)[:,None]
    error_s = jnp.linalg.norm(s_test - s_pred, 2) / jnp.linalg.norm(s_test, 2) 
    return error_s


if __name__ == "__main__":
    key = random.PRNGKey(0)

    # GRF length scale
    length_scale = 0.2

    # Resolution of the solution
    Nx = 100
    Nt = 100

    # Computational domain
    xmin = 0.0
    xmax = 1.0

    tmin = 0.0
    tmax = 1.0

    N = 1000 # number of input samples
    m = Nx   # number of input sensors
    P_train = 200   # number of output sensors, 100 for each side 
    Q_train = 2000  # number of collocation points for each input sample

    # Generate training data
    u_op_train, y_op_train, s_op_train, u_pde_train, y_pde_train, s_pde_train = generate_training_data(key, N, P_train, Q_train)
    
    # Initialize model
    branch_layers = [m, 100, 100, 100, 100, 100, 100]
    trunk_layers =  [2, 100, 100, 100, 100, 100, 100]

    model = AdvDeepONetPI(branch_layers, 
                          trunk_layers,
                          branch_net=modified_MLP,
                          trunk_net=modified_MLP,)
    
    # Create data set
    batch_size = 10000
    op_dataset = DataGenerator(u_op_train, y_op_train, s_op_train, batch_size)
    pde_dataset = DataGenerator(u_pde_train, y_pde_train, s_pde_train, batch_size)
    
    # Debugging
    params = model.get_params(model.opt_state)
    operator_data = iter(op_dataset)
    physics_data = iter(pde_dataset)
    operator_batch= next(operator_data)
    physics_batch = next(physics_data)
    model.train(op_dataset, pde_dataset, nIter=1000)
    
    params = model.get_params(model.opt_state)

    # Save the trained model
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    jnp.save('adv_params.npy', flat_params)
    jnp.save('adv_loss_operator.npy', model.loss_operator_log)
    jnp.save('adv_loss_physics.npy', model.loss_physics_log)

    # Restore the trained model
    flat_params = jnp.load('adv_params.npy')
    params = model.unravel_params(flat_params)
    
    loss_operator_log = model.loss_operator_log
    loss_physics_log = model.loss_physics_log

    # # Restore losses
    # loss_bcs_log = np.load('adv_loss_bcs.npy')
    # loss_res_log = np.load('adv_loss_res.npy')

    #Plot for loss function
    plt.figure(figsize = (6,5))
    plt.plot(loss_operator_log, lw=2, label='operator')
    plt.plot(loss_physics_log, lw=2, label='physics')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Test data over 100 random input samples
    N_test = 10 
    P_test = m  

    key_test = random.PRNGKey(7654321)
    subkeys_test = random.split(key_test, N_test)

    # Compute test error
    error_list = []
    for k in range(10):
        keys_test = random.split(subkeys_test[k], N_test)
        error_s = vmap(compute_error, (0, None, None, None))(keys_test, Nx, Nt, P_test) 
        error_list.append(error_s)

    error = jnp.stack(error_list)

    print('mean of relative L2 error of s: {:.2e}'.format(error.mean()))
    print('std of relative L2 error of s: {:.2e}'.format(error.std()))
    
    # Compute relative l2 error for one random input sample
    key = keys_test[1]
    length_scale= 0.2

    P_test = 100
    Nx = 200
    Nt = 200

    N_test = 1
    u_test, y_test, s_test = generate_test_data(key, N_test, Nx, Nt, P_test)

    # Predict
    s_pred = model.predict_s(params, u_test, y_test)

    # Evulate solution at a uniform grid
    x = jnp.linspace(0, 1, Nx)
    t = jnp.linspace(0, 1, Nt)
    XX, TT = jnp.meshgrid(x, t)

    S_pred = griddata(y_test, s_pred.flatten(), (XX,TT), method='cubic')
    S_test = griddata(y_test, s_test.flatten(), (XX,TT), method='cubic')

    error = jnp.linalg.norm(S_pred - S_test) / jnp.linalg.norm(S_test) 
    print('Relative l2 errpr: {:.3e}'.format(error))
    
    
    # Visualization
    fig = plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.pcolor(XX,TT, S_test, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact $s(x,t)$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1,3,2)
    plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1,3,3)
    plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error')
    plt.colorbar()
    plt.tight_layout()
    plt.show()