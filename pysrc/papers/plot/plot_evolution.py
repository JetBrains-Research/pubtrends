import logging

import holoviews as hv
from holoviews import dim

from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor

logger = logging.getLogger(__name__)


def plot_topics_evolution(df, evolution_df, evolution_kwds, width, height):
    """
    Sankey diagram of topic evolution
    :return:
        if evolution_df is None: None, as no evolution can be observed in 1 step
        Sankey diagram + table with keywords
    """
    logger.debug('Processing topic_evolution')
    # Topic evolution analysis failed, one step is not enough to analyze evolution
    if evolution_df is None or not evolution_kwds:
        logger.debug(f'Topic evolution failure, '
                     f'evolution_df is None: {evolution_df is None}, '
                     f'evolution_kwds is None: {evolution_kwds is None}')
        return None

    n_steps = len(evolution_df.columns) - 2

    edges, nodes_data = PlotPreprocessor.topic_evolution_data(
        evolution_df, evolution_kwds, n_steps
    )

    value_dim = hv.Dimension('Papers', unit=None)
    nodes_ds = hv.Dataset(nodes_data, 'index', 'label')
    topic_evolution = hv.Sankey((edges, nodes_ds), ['From', 'To'], vdims=value_dim)
    topic_evolution.opts(labels='label',
                         width=width, height=max(height, len(set(df['comp'])) * 30),
                         show_values=False, cmap='tab20',
                         edge_color=dim('To').str(), node_color=dim('index').str())

    p = hv.render(topic_evolution, backend='bokeh')
    p.sizing_mode = 'stretch_width'
    kwds_data = PlotPreprocessor.topic_evolution_keywords_data(
        evolution_kwds
    )
    return p, kwds_data
