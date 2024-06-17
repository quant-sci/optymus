import warnings
warnings.filterwarnings('ignore')

import jax.numpy as jnp

def mccormick_function():
  def func(x):
    x = jnp.array(x)
    return x[0]**2 - 3*x[0]*x[1] + 4*x[1]**2 + x[0] - x[1]
  return func

def rastrigin_function():
  def func(x):
    x = jnp.array(x)
    return 20 + x[0]**2 + x[1]**2 - 10*(jnp.cos(2*jnp.pi*x[0]) + jnp.cos(2*jnp.pi*x[1]))
  return func

def ackley_function():
  def func(x):
    x = jnp.array(x)
    return -20*jnp.exp(-0.2*jnp.sqrt(0.5*(x[0]**2 + x[1]**2))) - jnp.exp(0.5*(jnp.cos(2*jnp.pi*x[0]) + jnp.cos(2*jnp.pi*x[1]))) + jnp.exp(1) + 20
  return func

def eggholder_function():
  def func(x):
    x = jnp.array(x)
    return -(x[1] + 47)*jnp.sin(jnp.sqrt(jnp.abs(x[0]/2 + x[1] + 47))) - x[0]*jnp.sin(jnp.sqrt(jnp.abs(x[0] - x[1] - 47)))
  return func

def crossintray_function():
  def func(x):
    x = jnp.array(x)
    return -0.0001*(jnp.abs(jnp.sin(x[0])*jnp.sin(x[1])*jnp.exp(jnp.abs(100 - jnp.sqrt(x[0]**2 + x[1]**2)/jnp.pi))) + 1)**0.1
  return func

def sphere_function():
  def func(x):
    x = jnp.array(x)
    return x[0]**2+x[1]**2
  return func

def rosenbrock_function():
  def func(x):
    x = jnp.array(x)
    return jnp.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
  return func

def beale_function():
  def func(x):
    x = jnp.array(x)
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
  return func

def goldenstein_price_function():
  def func(x):
    x = jnp.array(x)
    return (1 + (x[0] + x[1] + 1)**2*(19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2))*(30 + (2*x[0] - 3*x[1])**2*(18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
  return func

def booth_function():
  def func(x):
    x = jnp.array(x)
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
  return func

def styblinski_tang_function():
  def func(x):
    x = jnp.array(x)
    return 0.5*((x[0]**4 - 16*x[0]**2 + 5*x[0])+(x[1]**4 - 16*x[1]**2 + 5*x[1]))
  return func