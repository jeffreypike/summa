import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from jax._src.typing import Array
from jax._src.nn.initializers import Initializer, KeyArray, DTypeLikeInexact, RealNumeric
from jax._src import core
from jax._src import dtypes
from typing import Optional

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

# Visualization
colour_scheme = ['#4C78A8',
                 '#F58518',
                 '#E45756',
                 '#72B7B2',
                 '#54A24B',
                 '#EECA3B',
                 '#B279A2',
                 '#FF9DA6',
                 '#9D755D',
                 '#BAB0AC']

def format_fig(figure: go.Figure,
               colour_scheme: Optional[list] = colour_scheme) -> go.Figure:
    fig = go.Figure(figure)

    # remove plotly blue background, centre title
    fig.update_layout(paper_bgcolor = "#ffffff",
                      plot_bgcolor = "#ffffff",
                      title_x = 0.5)
    
    # add axes
    fig.update_xaxes(showline = True, linewidth = 1, linecolor ="black")
    fig.update_yaxes(showline = True, linewidth = 1, linecolor ="black")

    # update colour scheme
    if colour_scheme is not None:
        for j, trace in enumerate(fig.data):
            trace.update(marker_color = colour_scheme[j])
    
    return fig