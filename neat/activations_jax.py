"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""

import jax
from jax import numpy as jnp
import types


@jax.jit
def sigmoid_activation(z):
    z = jnp.maximum(-60.0, jnp.minimum(60.0, 5.0 * z))
    return 1.0 / (1.0 + jnp.exp(-z))


@jax.jit
def tanh_activation(z):
    z = jnp.maximum(-60.0, jnp.minimum(60.0, 2.5 * z))
    return jnp.tanh(z)


@jax.jit
def sin_activation(z):
    z = jnp.maximum(-60.0, jnp.minimum(60.0, 5.0 * z))
    return jnp.sin(z)


@jax.jit
def gauss_activation(z):
    z = jnp.maximum(-3.4, jnp.minimum(3.4, z))
    return jnp.exp(-5.0 * z ** 2)


@jax.jit
def relu_activation(z):
    return jnp.maximum(z, 0.0)


@jax.jit
def elu_activation(z):
    return jnp.where(z > 0.0, z, jnp.exp(z) - 1)


@jax.jit
def lelu_activation(z):
    leaky = 0.005
    return jnp.where(z > 0.0, z, leaky * z)


@jax.jit
def selu_activation(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return jnp.where(z > 0.0, lam * z, lam * alpha * (jnp.exp(z) - 1))


@jax.jit
def softplus_activation(z):
    z = jnp.maximum(-60.0, jnp.minimum(60.0, 5.0 * z))
    return 0.2 * jnp.log(1 + jnp.exp(z))


@jax.jit
def identity_activation(z):
    return z


@jax.jit
def clamped_activation(z):
    return jnp.maximum(-1.0, jnp.minimum(1.0, z))


@jax.jit
def inv_activation(z):
    z = 1.0 / z
    return jnp.where(jnp.isfinite(z), z, 0.0)


@jax.jit
def log_activation(z):
    z = jnp.maximum(1e-7, z)
    return jnp.log(z)


@jax.jit
def exp_activation(z):
    z = jnp.maximum(-60.0, jnp.minimum(60.0, z))
    return jnp.exp(z)


@jax.jit
def abs_activation(z):
    return jnp.abs(z)


@jax.jit
def hat_activation(z):
    return jnp.maximum(0.0, 1 - jnp.abs(z))


@jax.jit
def square_activation(z):
    return z ** 2


@jax.jit
def cube_activation(z):
    return z ** 3


class InvalidActivationFunction(TypeError):
    pass


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {} # name (str) to func
        self.lut = [] # lookup table: idx (int) to func
        self.name2idx = {} # name (str) to idx (int)
        self.add('identity', identity_activation)
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)
        self.add('relu', relu_activation)
        self.add('elu', elu_activation)
        self.add('lelu', lelu_activation)
        self.add('selu', selu_activation)
        self.add('softplus', softplus_activation)
        self.add('clamped', clamped_activation)
        self.add('inv', inv_activation)
        self.add('log', log_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)

    def add(self, name, function):
        self.functions[name] = function
        self.name2idx[name] = len(self.lut)
        self.lut.append(function)

    def get(self, name):
        if isinstance(name, int):
            name = self.lut[name]
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions