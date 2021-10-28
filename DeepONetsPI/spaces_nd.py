from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax._src.dtypes import dtype

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
import jax.scipy as jsci
from jax.ops import index, index_add, index_update, index_mul
# from jax import random
from pandas.core import indexing
from pathos.pools import ProcessPool
from scipy import linalg, interpolate
import scipy.sparse as ssp
from scipy.special import gamma
import scipy as sci

from sklearn import gaussian_process as gp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import config
# from utils import eig
from IPython.display import display
import time

jax.config.update("jax_enable_x64", True)

def RBF(x1, x2, output_scale, lengthscales):
    # output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

def construct_points(grid):
    N = grid[0].size
    dim = len(grid)
    R = np.zeros((N, dim))
    for i in range(dim):
        R[:,i] = grid[i].ravel()
    return R

def construct_points_jax(grid):
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


def construct_grid_jax(points,values,shape):
    N = points.shape[0]
    dim = points.shape[1]
    grid = tuple(points[:,i].reshape(shape) for i in range(dim))
    vals = values.reshape(shape)
    return grid, vals

class GRF_nd(object):
    def __init__(self, T, dim=1, length_scale=1, N=1000, interp="cubic", kernel="RBF", nu=0.5):
        self.interp = interp
        self.dim = dim
        # if N.isdigit():
        if not isinstance(N, (list, tuple)):
            N = np.array([N]*dim)
        if not isinstance(T, (list, tuple)):
            T = np.array([T]*dim)
        self.N = N
        self.T = T
        Ntot = np.prod(N)
        self.Ntot = Ntot
        x = []
        for i in range(dim):
            x.append(np.linspace(0, T[i], num=N[i]))
        grid = np.meshgrid(*x,indexing='ij')
        R = np.zeros((self.Ntot, self.dim))
        for i in range(dim):
            R[:,i] = grid[i].ravel()
        self.x = R
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=nu)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.Ntot))
        # self.L = np.linalg.cholesky(self.K + 1e-13 * ssp.eye(self.Ntot))
    def random(self, n):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.Ntot, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        # print(f"self.x: {self.x.shape}")
        # print(f"x: {x.shape}")
        # print(f"y: {y.flatten().shape}")
        f = griddata(self.x, y.flatten(), x, method=self.interp)
        # if self.interp == "linear":
        #     return np.interp(x, np.ravel(self.x), y)
        # f = interpolate.interp1d(
        #     np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        # )
        return f

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        # f = np.vstack([griddata(sensors)])
        # if self.interp == "linear":
        #     return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        p = ProcessPool(nodes=config.processes)
        res = p.map(lambda y: griddata(self.x, y, sensors, method=self.interp), ys)
        #  (   lambda y: interpolate.interp1d(
        #         np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        #     )(sensors).T,
        #     ys,
        # )
        return np.vstack(list(res))

class GRF_nd_jax(object):
    # Need to perform GRF as float64 otherwise cholesky fails with matrix not positive definite
    def __init__(self, T, dim=1, kernel="RBF", length_scale=1.0, N=1000, interp="cubic"):
        self.interp = interp
        self.dim = dim
        # if N.isdigit():
        if not isinstance(N, (list, tuple)):
            N = jnp.array([N]*dim)
        if not isinstance(N, (list, tuple)):
            T = jnp.array([T]*dim)
        self.N = N
        self.T = T
        Ntot = jnp.prod(N)
        self.Ntot = Ntot
        x = [jnp.linspace(0, T[i], num=N[i], dtype=jnp.float64) for i in range(dim)]
        # x = []
        # for i in range(dim):
        #     x.append(jnp.linspace(0, T[i], num=N[i]))
        grid = jnp.meshgrid(*x,indexing='ij')
        R = jnp.zeros((self.Ntot, self.dim))
        for i in range(dim):
            # R[:,i] = grid[i].ravel()
            R = index_update(R, index[:, i], grid[i].ravel())
        self.x = R
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = jnp.linalg.cholesky(self.K + 1e-13 * jnp.eye(self.Ntot))
        # space.K + 1e-13 * jnp.eye(space.Ntot)
        
    def random(self, n, key=jax.random.PRNGKey(0)):
        """Generate `n` random feature vectors.
        """
        u = jax.random.normal(key, shape=(self.Ntot, n))
        # u = jnp.random.randn(self.Ntot, n)
        
        return jnp.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        
        points = jnp.float32(self.x)
        point_values = jnp.float32(y.flatten())
        sensors = jnp.float32(x)
        
        f = griddata(points, point_values, sensors, method=self.interp)
        # if self.interp == "linear":
        #     return jnp.interp(x, jnp.ravel(self.x), y)
        # f = interpolate.interp1d(
        #     jnp.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        # )
        return f

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        # f = jnp.vstack([griddata(sensors)])
        # if self.interp == "linear":
        #     return jnp.vstack([jnp.interp(sensors, jnp.ravel(self.x), y).T for y in ys])
        points = jnp.float32(self.x)
        point_values = jnp.float32(ys)
        sensors = jnp.float32(sensors)
        # p = ProcessPool(nodes=config.processes)
        # res = p.map(lambda y: griddata(points, y, sensors, method=self.interp), point_values)
        res = pmap(lambda y: griddata(points, y, sensors, method=self.interp))(point_values)
        #  (   lambda y: interpolate.interp1d(
        #         jnp.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        #     )(sensors).T,
        #     ys,
        # )
        return jnp.vstack(list(res))
    
    
    

if __name__ == "__main__":
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    # space = GRF_KL(1, length_scale=0.2, num_eig=10, N=100, interp="cubic")
    # space_samples(space, 1)
    num = 1
    interp = 'linear' # cubic
    N = 100
    length_scale = 0.2
    
    start = time.time()
    space = GRF_nd(1, dim=2, length_scale=length_scale, N=N, interp=interp)
    sensors = space.x
    features = space.random(num)
    # sensor_values = space.eval_u(features, sensors)
    sensor_values = space.eval_u_one(features, sensors)
    end = time.time()
    
    start_jax = time.time()
    space = GRF_nd_jax(1, dim=2, length_scale=length_scale, N=N, interp=interp)
    sensors = space.x
    features = space.random(num)
    # sensor_values = space.eval_u(features, sensors)
    sensor_values = space.eval_u_one(features, sensors)
    end_jax = time.time()
    
    Time = (end - start) * 1000
    Time_jax = (end_jax - start_jax) * 1000

    print(f"Time: {int(Time)} ms")
    print(f"Time_jax: {int(Time_jax)} ms")
    # space2 = GRF_nd(1, dim=1, length_scale=1, N=100, interp="cubic")
    # W2 = jnp.trace(space1.K + space2.K - 2 * linalg.sqrtm(space1.K @ space2.K)) ** 0.5 / 100 ** 0.5
    # print(W2)
    grid, U = construct_grid(sensors, sensor_values, shape=space.N)
    X,Y = grid
    plt.close('all')
    fig = plt.figure()
    c = plt.pcolormesh(X,Y,U,cmap='jet',shading='gouraud')
    # c = plt.pcolormesh(X,Y,U,cmap='jet', shading='auto')
    fig.colorbar(c)
    # plt.title()
    # plt.xlabel()
    # plt.ylabel()
    # plt.axis('equal')
    plt.axis('square')
    plt.show()