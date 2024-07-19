import jax.numpy as jnp
import flax.linen as nn
import optax
import plotly.graph_objects as go
from jax._src.typing import Array
from typing import Optional
from summa.models import NAM

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
            if type(trace) == go.Contour:
                continue
            trace.update(marker_color = colour_scheme[j])
    
    return fig

def plot_decision_boundary(nam: NAM,
                           params: optax.Params,
                           X: Array,
                           step_size: Optional[float] = 0.01,
                           colour_scheme: Optional[list] = colour_scheme[:2],
                           opacity: Optional[float] = 0.4) -> go.Contour:
    '''Plots the decision boundary for a 2D classifier'''
    # set min and max values and create some padding
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    delta_x = (x_max - x_min) / 10
    delta_y = (y_max - y_min) / 10

    # generate a grid of points with distance h between them
    xx, yy = jnp.meshgrid(jnp.arange(x_min - delta_x, x_max + delta_x, step_size), 
                          jnp.arange(y_min - delta_y, y_max + delta_y, step_size))

    # predict the function value for the whole gid
    Z = nn.sigmoid(nam.apply(params, jnp.c_[xx.ravel(), yy.ravel()]))

    # generate the contour trace
    trace = go.Contour(x = xx.ravel(),
                       y = yy.ravel(),
                       z = Z,
                       colorscale = colour_scheme,
                       opacity = opacity)
    
    return trace