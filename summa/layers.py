import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from summa.utils import shifted_truncated_normal

class ExuLayer(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.truncated_normal(stddev = 0.5)
    bias_init: Callable = shifted_truncated_normal()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init, # Initialization function
                            (inputs.shape[-1], self.features))  # shape info.
        bias = self.param('bias', self.bias_init, (1,))
        y = inputs - bias
        y = jnp.exp(kernel) * y
        return y