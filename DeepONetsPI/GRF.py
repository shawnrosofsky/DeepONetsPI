from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pandas.core import indexing
from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from scipy.special import gamma
import scipy as sci
from sklearn import gaussian_process as gp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from IPython.display import display
import sys
import h5py
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap, random
import jax.scipy as jsci


def construct_points(grid):
    N = grid[0].size
    dim = len(grid)
    R = jnp.zeros((N, dim))
    for i in range(dim):
        R[:,i] = grid[i].ravel()
    return R

def construct_grid(points,values,shape):
    N = points.shape[0]
    dim = points.shape[1]
    grid = tuple(points[:,i].reshape(shape) for i in range(dim))
    vals = values.reshape(shape)
    return grid, vals


def RBF(x1, x2, output_scale, lengthscales):
    # output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


def dirichlet_mattern(x1, x2=None, Nk=None, l=0.1, sigma=1.0, nu=jnp.inf):
    # do even only 
    d = x1.shape[1]
    if Nk is None:
        Nk = jnp.int64(jnp.power(x1.shape[0],1.0/d))
    # Nk = 2 * (Nk - Nk//2)
    L = x1[-1] - x1[0]
    N_full = jnp.array([Nk] * d)
    N_tot = jnp.prod(N_full)
    alpha = nu + 0.5*d
    a = jnp.sqrt(2.0/L)
    a0 = jnp.sqrt(1.0/L)
    kappa = jnp.sqrt(2*nu)/l
    eta2 = sigma*(4.0*jnp.pi)**(0.5*d)*gamma(alpha)/(kappa**d * gamma(nu))
    k = [jnp.linspace(1, Nk, Nk) for _ in range(d)]
    
    k_grid = jnp.meshgrid(*k, indexing='ij')
    K = jnp.zeros((N_tot, d))
    for i in range(d):
        K = K.at[:, i].set(k_grid[i].ravel())
    # print(jnp.linalg.norm(K, axis=1))
    # Lnorm = jnp.linalg.norm(L)
    # Lval = L[0] # this is from coppying the paper, I think we may need to use the Norm of L
    Knorm = jnp.linalg.norm(K/L, axis=1)
    eigs_k = 1 + (jnp.pi/(kappa))**2 * Knorm**2
    # print(jnp.sum(K*x1, axis=1).shape)
    eigs_k = eigs_k[:, None]
    eigs_alpha = eigs_k**(-alpha)
    if jnp.isinf(nu):
        eigs_k = jnp.ones_like(eigs_k)
        eta2 = sigma*(jnp.sqrt(2.0*jnp.pi)*l)**d
        eigs_alpha = jnp.exp(-0.5*(l*jnp.pi*Knorm)**2)[:, None] # 1/2 instead of 2 for nonperiodic case
    Kx1 = jnp.expand_dims(K,1)*x1
    wk = jnp.prod(a*jnp.sin(jnp.pi/L * Kx1), axis=2)
    if x2 is not None:
        Kx2 = jnp.expand_dims(K,1)*x2
        wk_star = jnp.prod(a*jnp.sin(jnp.pi/L * Kx2), axis=2).T
    else:
        wk_star = wk.T
    
    cov = eta2 * (wk_star @ (wk * eigs_alpha))
    
    return cov
    
    
def neumann_mattern(x1, x2=None, Nk=None, l=0.1, sigma=1.0, nu=jnp.inf):
    # do even only 
    d = x1.shape[1]
    d = x1.shape[1]
    if Nk is None:
        Nk = jnp.int64(jnp.power(x1.shape[0],1.0/d))
    # N = 2 * (N - N//2)
    L = x1[-1] - x1[0]
    N_full = jnp.array([Nk] * d)
    N_tot = jnp.prod(N_full)
    alpha = nu + 0.5*d
    a1 = jnp.sqrt(2.0/L)
    a0 = jnp.sqrt(1.0/L)
    kappa = jnp.sqrt(2*nu)/l
    eta2 = sigma*(4.0*jnp.pi)**(0.5*d)*gamma(alpha)/(kappa**d * gamma(nu))
    k = [jnp.linspace(0, Nk-1, Nk) for _ in range(d)]
    k_grid = jnp.meshgrid(*k, indexing='ij')
    K = jnp.zeros((N_tot, d))
    ones = jnp.ones((Nk))
    a = [ones*a1[i] for i in range(d)]
    a = [a[i].at[0].set(a0[i]) for i in range(d)]
    a_grid = jnp.meshgrid(*a, indexing='ij')
    A = jnp.zeros((N_tot, d)) # probably inefficient to do it like this, but it nicely keeps track of when k_i is 0 which requires us to use a0 instead of a
    for i in range(d):
        K = K.at[:, i].set(k_grid[i].ravel())
        A = A.at[:, i].set(a_grid[i].ravel())
    # print(jnp.linalg.norm(K, axis=1))
    # Lnorm = jnp.linalg.norm(L)
    # Lval = L[0] # this is from coppying the paper, I think we may need to use the Norm of L
    Knorm = jnp.linalg.norm(K/L, axis=1)
    eigs_k = 1 + (jnp.pi/(kappa))**2 * Knorm**2
    # print(jnp.sum(K*x1, axis=1).shape)
    eigs_k = eigs_k[:, None]
    eigs_alpha = eigs_k**(-alpha)
    if jnp.isinf(nu):
        eigs_k = jnp.ones_like(eigs_k)
        eta2 = sigma*(jnp.sqrt(2.0*jnp.pi)*l)**d
        eigs_alpha = jnp.exp(-0.5*(l*jnp.pi*Knorm)**2)[:, None] # 1/2 instead of 2 for nonperiodic case
    Kx1 = jnp.expand_dims(K,1)*x1
    wk = jnp.prod(A*jnp.cos(jnp.pi/L * Kx1), axis=2)
    if x2 is not None:
        Kx2 = jnp.expand_dims(K,1)*x2
        wk_star = jnp.prod(A*jnp.cos(jnp.pi/L * Kx2), axis=2).T
    else:
        wk_star = wk.T
    
    cov = eta2 * (wk_star @ (wk * eigs_alpha))
    
    return cov
    
    
    
def periodic_mattern(x1, x2=None, Nk=None, l=0.1, sigma=1.0, nu=jnp.inf):
    d = x1.shape[1]
    L = x1[-1] - x1[0]
    if Nk is None:
        Nk = jnp.int64(jnp.power(x1.shape[0],1.0/d))    # Nk = 2 * (Nk - Nk//2)
    # Nk = 2 * (Nk - Nk//2)
    N_full = jnp.array([Nk] * d)
    N_tot = jnp.prod(N_full)
    alpha = nu + 0.5*d
    a1 = jnp.sqrt(2.0/L)
    a0 = jnp.sqrt(1.0/L)
    kappa = jnp.sqrt(2*nu)/l
    
    eta2 = sigma*(4.0*jnp.pi)**(0.5*d)*gamma(alpha)/(kappa**d * gamma(nu))
    k = [jnp.linspace(0, Nk-1, Nk) for _ in range(d)]
    k_grid = jnp.meshgrid(*k, indexing='ij')
    
    K = jnp.zeros((N_tot, d))
    ones = jnp.ones((Nk))
    a = [ones*a1[i] for i in range(d)]
    a = [a[i].at[0].set(a0[i]) for i in range(d)]
    a_grid = jnp.meshgrid(*a, indexing='ij')
    A = jnp.zeros((N_tot, d)) # probably inefficient to do it like this, but it nicely keeps track of when k_i is 0 which requires us to use a0 instead of a
    for i in range(d):
        K = K.at[:, i].set(k_grid[i].ravel())
        A = A.at[:, i].set(a_grid[i].ravel())
    # print(jnp.linalg.norm(K, axis=1))
    # Lnorm = jnp.linalg.norm(L)
    # Lval = L[0] # this is from coppying the paper, I think we may need to use the Norm of L
    Knorm = jnp.linalg.norm(K/L, axis=1)
    eigs_k = 1 + (2.0*jnp.pi/(kappa))**2 * Knorm**2
    eigs_k = eigs_k[:, None]
    eigs_alpha = eigs_k**(-alpha)
    if jnp.isinf(nu):
        eigs_k = jnp.ones_like(eigs_k)
        eta2 = sigma*(jnp.sqrt(2.0*jnp.pi)*l)**d
        eigs_alpha = jnp.exp(-2.0*(l*jnp.pi*Knorm)**2)[:, None]
        # display(eigs_alpha.shape)
    Kx1 = jnp.expand_dims(K,1)*x1
    wk = jnp.prod(a1*(jnp.cos(2*jnp.pi/L * Kx1) + jnp.sin(2*jnp.pi/L * Kx1)), axis=2)
    if x2 is not None:
        Kx2 = jnp.expand_dims(K,1)*x2
        wk_star = jnp.prod(A*(jnp.cos(2*jnp.pi/L * Kx2) + jnp.sin(2*jnp.pi/L * Kx2)), axis=2).T
    else:
        wk_star = wk.T
    
    cov = eta2 * (wk_star @ (wk * eigs_alpha))
    
    return cov
    
    
def get_cholesky(K, jitter=1e-12):
    N = K.shape[0]
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    return L

def setup_kernel(N, dim):
    # dim = 2
    # N = 100
    N_full = jnp.array([N]*dim)
    Ntot = jnp.prod(N_full)
    x = [jnp.linspace(0, 1, N) for i in range(dim)]
    grid = jnp.meshgrid(*x, indexing='ij')
    X = jnp.zeros((Ntot, dim))
    for i in range(dim):
        X = X.at[:,i].set(grid[i].ravel())
    return X

def generate_sample(key, L):
    N = L.shape[0]
    rand = random.normal(key, (N,))
    sample = jnp.dot(L, rand).T
    return sample


def generate_samples(key, L, Nsamples):
    keys = random.split(key, Nsamples)
    samples = vmap(generate_sample, (0, None))(keys, L)
    return samples

def plot_sample(sample, dim, shape):
    fig = plt.figure()
    if dim == 2:
        grid, U = construct_grid(X, sample, shape=shape)
        X1, X2 = grid
        c = plt.pcolormesh(X1, X2, U, cmap='jet', shading='gouraud', vmin=-2, vmax=2)
        fig.colorbar(c)
        plt.title('GRF Jax')
        plt.axis('square')
    elif dim == 1:
        plt.plot(X, sample)
    plt.show()
    return U

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(0)
    dim = 2
    N = 101
    l = 0.1
    Nk = 25
    Nsamples = 10
    jitter = 1e-12
    shape = jnp.array([N]*dim)
    X = setup_kernel(N, dim)
    print(f"X={X}")
    K = periodic_mattern(X, Nk=Nk, l=l, nu=jnp.inf)
    print(f"K={K}")
    L = get_cholesky(K, jitter)
    print(f"L={L}")
    jax.config.update("jax_enable_x64", False)
    samples = generate_samples(key, L, Nsamples)
    print(samples.shape)
    plt.close('all')
    U = jnp.array([plot_sample(sample, dim, shape) for sample in samples])
