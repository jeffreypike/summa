import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence
from jax._src.typing import Array
from summa.layers import ExuLayer
from summa.utils import mish

class FeatureNet(nn.Module):
    hidden_units: int
    shallow: bool = True
        
    @nn.compact
    def __call__(self, 
                 inputs: Array, 
                 hyperparams: Optional[dict] = None):
        '''Evaluates the feature network'''
        
        dropout_rate = 0. if hyperparams is None or 'dropout_rate' not in hyperparams else hyperparams['dropout_rate']
        x = ExuLayer(self.hidden_units)(inputs)
        x = mish(x)
        x = nn.Dropout(rate = dropout_rate, deterministic = False)(x)
        if not self.shallow:
            x = nn.Dense(64, kernel_init = nn.initializers.glorot_uniform())(x)
            x = nn.relu(x)
            x = nn.Dense(32, kernel_init = nn.initializers.glorot_uniform())(x)
            x = nn.relu(x)
        x = nn.Dense(1, kernel_init = nn.initializers.glorot_uniform())(x)
        return x

class NAM(nn.Module):
    hidden_units: Sequence[int]
    rng_collection: str = 'feature_dropout'
    
    def setup(self):
        self.subnets = [FeatureNet(units) for units in self.hidden_units]
    
    def __call__(self, 
                 X: Array, 
                 hyperparams: Optional[dict] = None):
        '''Evaluates the NAM'''
        feature_dropout_rate = 0. if hyperparams is None or 'feature_dropout_rate' not in hyperparams else hyperparams['feature_dropout_rate']
        output = 0
        for i, subnet in enumerate(self.subnets):
            xi = X[:, i].reshape(X.shape[0], 1)
            
            if not feature_dropout_rate:
                output += subnet(xi, hyperparams)
            
            else:
                random_key = self.make_rng(self.rng_collection)
                rng = jax.random.uniform(random_key)

                output += jax.lax.select((rng > feature_dropout_rate), 
                                         subnet(xi, hyperparams),
                                         jnp.zeros_like(xi))
        return output