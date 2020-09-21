import holoviews as hv
from bokeh.embed import components
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from holoviews import dim

from pysrc.papers.plot.plot_preprocessor_experimental import ExperimentalPlotPreprocessor
from pysrc.papers.plot.plotter import Plotter, PLOT_WIDTH, TALL_PLOT_HEIGHT, visualize_analysis


def visualize_experimental_analysis(analyzer):
    result = visualize_analysis(analyzer)
    result['experimental'] = True  # Mark as experimental
    if analyzer.similarity_graph.nodes():
        evolution_result = ExperimentalPlotter(analyzer).topic_evolution()
        if evolution_result is not None:
            evolution_data, keywords_data = evolution_result
            result['topic_evolution'] = [components(evolution_data)]
            if keywords_data:
                result['topic_evolution_keywords'] = keywords_data
        return result


class ExperimentalPlotter(Plotter):
    def __init__(self, analyzer=None):
        super().__init__(analyzer)

    def topic_evolution(self):
        """
        Sankey diagram of topic evolution
        :return:
            if self.analyzer.evolution_df is None: None, as no evolution can be observed in 1 step
            if number of steps < 3: Sankey diagram
            else: Sankey diagram + table with keywords
        """
        # Topic evolution analysis failed, one step is not enough to analyze evolution
        if self.analyzer.evolution_df is None or not self.analyzer.evolution_kwds:
            return None

        n_steps = len(self.analyzer.evolution_df.columns) - 2

        edges, nodes_data = ExperimentalPlotPreprocessor.topic_evolution_data(
            self.analyzer.evolution_df, self.analyzer.evolution_kwds, n_steps
        )

        value_dim = hv.Dimension('Amount', unit=None)
        nodes_ds = hv.Dataset(nodes_data, 'index', 'label')
        topic_evolution = hv.Sankey((edges, nodes_ds), ['From', 'To'], vdims=value_dim)
        topic_evolution.opts(labels='label', width=PLOT_WIDTH, height=TALL_PLOT_HEIGHT,
                             show_values=False, cmap='tab20',
                             edge_color=dim('To').str(), node_color=dim('index').str())

        if n_steps > 3:
            kwds_data = ExperimentalPlotPreprocessor.topic_evolution_keywords_data(
                self.analyzer.evolution_kwds
            )
            return hv.render(topic_evolution, backend='bokeh'), kwds_data

        return hv.render(topic_evolution, backend='bokeh'), None
