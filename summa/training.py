import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional
from jax._src.typing import Array
from summa.models import FeatureNet, NAM


def get_optimal_params(model,
                       params: optax.Params,
                       X_train: Array,
                       y_train: Array,
                       loss_fn: Callable,
                       optimizer: optax.GradientTransformation,
                       epochs: int,
                       rngs: dict,
                       hyperparams: Optional[dict] = None,
                       X_test: Optional[Array] = None,
                       y_test: Optional[Array] = None) -> optax.Params:
    '''Train a model'''
    def l2_loss(X, weight_decay):
        return weight_decay * (X ** 2).mean()

    weight_decay = 0. if hyperparams is None or 'weight_decay' not in hyperparams else hyperparams['weight_decay']
    output_penalty = 0. if hyperparams is None or 'output_penalty' not in hyperparams else hyperparams['output_penalty']
    
    def loss(params, X_batch, y_batch):
        pred = model.apply(params, 
                           X_batch, 
                           hyperparams,
                           rngs = rngs)
        loss = jnp.mean(loss_fn(pred, y_batch))
        loss += sum([l2_loss(weight, weight_decay) for weight in jax.tree_util.tree_leaves(params)])
        if type(model) == NAM:
            for j in range(len(model.hidden_units)):
                subnet = list(params['params'].keys())[j]
                params_subnet = {'params': params['params'][subnet]}
                y = FeatureNet(nam.hidden_units[j]).apply(params_subnet, X_batch[:, j].reshape(100,1))
                loss += output_penalty * jnp.sum(y**2)
        return loss
    
    @jax.jit
    def step(params, opt_state, X_batch, y_batch):
        loss_value, grads = jax.value_and_grad(loss)(params, X_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # initialize values
    opt_state = optimizer.init(params)
    batch_size = X_train.shape[0] if hyperparams is None or 'batch_size' not in hyperparams else hyperparams['batch_size']
    batch_key = None if rngs is None or 'batching' not in rngs else rngs['batching']
    history = []
    best_loss = jnp.inf
    best_params = params
    val_loss = None
    val_history = []
    
    for j in range(epochs):
        permutation = jnp.arange(X_train.shape[0])
        if batch_key is not None:
            batch_key = jax.random.split(batch_key, 1)[0]
            permutation = jax.random.permutation(batch_key, permutation)
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_indices = permutation[i: i + batch_size]
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]
            params, opt_state, loss_value = step(params, opt_state, X_batch, y_batch)
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = params
            if X_test is not None:
                val_loss = loss(params, X_test, y_test)
        if j % 100 == 0:
            print(f'step {j}, loss: {loss_value}')
        history.append(loss_value)
    return best_params, {'train_loss': history, 'best_loss': best_loss, 'validation_loss': val_history}