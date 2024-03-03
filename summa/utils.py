import jax
import jax.numpy as jnp
from jax._src.typing import Array
from jax._src.nn.initializers import Initializer, KeyArray, DTypeLikeInexact, RealNumeric
from jax._src import core
from jax._src import dtypes

# Activation Functions
def mish(x):
    return jnp.multiply(x, jnp.tanh(jnp.log(1 + jnp.exp(x))))

def relu_n(x, n = 1):
    return jnp.clip(x, 0 ,n)

# Intializers
def shifted_truncated_normal(mean: RealNumeric = 4,
                             stddev: RealNumeric = 0.5,
                             dtype: DTypeLikeInexact = jnp.float_,
                             lower: RealNumeric = -1.0,
                             upper: RealNumeric = 1.0) -> Initializer:

    def init(key: KeyArray,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return jax.random.truncated_normal(key, lower, upper, shape, dtype) * stddev + mean
    return init

