# from flax.linen import activation
from flax.linen import activation
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import random, grad, vmap, jit, hessian
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, elu, softplus, tanh
from jax.config import config
from jax.ops import index_update, index
from jax import lax
from jax.lax import while_loop, scan, cond
from jax.flatten_util import ravel_pytree
import flax
import flax.linen as nn
from typing import Any, Callable, Sequence, Optional
from flax.core import freeze, unfreeze


import itertools
from functools import partial
from torch.utils import data

from scipy.interpolate import griddata


# Define MLP
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = jnp.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = jnp.dot(inputs, W) + b
      return outputs
  return init, apply


class FlaxMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = relu
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, name=f'layers_{i}')(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x
    

# Define modified MLP
def modified_MLP(layers, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = jnp.zeros(d_out)
      return W, b

  def init(rng_key):
      U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return (params, U1, b1, U2, b2) 

  def apply(params, inputs):
      params, U1, b1, U2, b2 = params
      U = activation(jnp.dot(inputs, U1) + b1)
      V = activation(jnp.dot(inputs, U2) + b2)
      for W, b in params[:-1]:
          outputs = activation(jnp.dot(inputs, W) + b)
          inputs = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V) 
      W, b = params[-1]
      outputs = jnp.dot(inputs, W) + b
      return outputs
  return init, apply


class FlaxModifiedMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = relu
    
    @nn.compact
    def __call__(self, inputs):
        U = self.activation(nn.Dense(inputs, name='U'))
        V = self.activation(nn.Dense(inputs, name='V'))
        x = inputs
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, name=f'layers_{i}')(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = jnp.multiply(x, U) + jnp.multiply(1.0 - x, V)
        return x

# Define Fourier feature net
def FF_MLP(layers, freqs=50, activation=relu):
   # Define input encoding function
    def input_encoding(x, w):
        out = jnp.hstack([jnp.sin(jnp.dot(x, w)),
                         jnp.cos(jnp.dot(x, w))])
        return out
    FF = freqs * random.normal(random.PRNGKey(0), (layers[0], layers[1]//2))
    def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[1:-1], layers[2:]))
      return params
    def apply(params, inputs):
        H = input_encoding(inputs, FF)
        for W, b in params[:-1]:
            outputs = jnp.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(H, W) + b
        return outputs
    return init, apply

# Define Fourier feature net using Flax
class FlaskFFMLP(nn.Module):
    layers: Sequence[int]
    freqs: int = 50
    activation: Callable[[jnp.ndarray], jnp.ndarray] = relu
    
    def setup(self):
        self.FF = self.freqs * random.normal(random.PRNGKey(0), (self.layers[0], self.layers[1]//2))

    # def init(rng_key):
    #   def init_layer(key, d_in, d_out):
    #       k1, k2 = random.split(key)
    #       glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
    #       W = glorot_stddev * random.normal(k1, (d_in, d_out))
    #       b = jnp.zeros(d_out)
    #       return W, b
    #   key, *keys = random.split(rng_key, len(self.layers))
    #   params = list(map(init_layer, keys, self.layers[1:-1], self.layers[2:]))
    #   return params
    
    
    def __call__(self, inputs):
        x = self.input_encoding(inputs, self.FF)
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, name=f'layers_{i}')(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        outputs = x
        return outputs
    
    # Define input encoding function
    def input_encoding(self, x, w):
        out = jnp.hstack([jnp.sin(jnp.dot(x, w)),
                         jnp.cos(jnp.dot(x, w))])
        return out