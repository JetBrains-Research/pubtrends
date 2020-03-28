import holoviews as hv
from bokeh.embed import components
from bokeh.layouts import column
# Tools used: hover,pan,tap,wheel_zoom,box_zoom,reset,save
from bokeh.models.widgets.tables import DataTable
from holoviews import dim

from models.keypaper.plot_preprocessor_experimental import ExperimentalPlotPreprocessor
from models.keypaper.plotter import Plotter, PLOT_WIDTH, TALL_PLOT_HEIGHT, visualize_analysis


def visualize_experimental_analysis(analyzer):
    result = visualize_analysis(analyzer)
    result['experimental'] = True  # Mark as experimental
    if analyzer.paper_relations_graph.nodes():
        subtopic_evolution = ExperimentalPlotter(analyzer).subtopic_evolution()
        # Pass subtopic evolution only if not None
        if subtopic_evolution:
            result['subtopic_evolution'] = [components(subtopic_evolution)]
    return result


class ExperimentalPlotter(Plotter):
    def __init__(self, analyzer=None):
        super().__init__(analyzer)

    def subtopic_evolution(self):
        """
        Sankey diagram of subtopic evolution
        :return:
            if self.analyzer.evolution_df is None: None, as no evolution can be observed in 1 step
            if number of steps < 3: Sankey diagram
            else: Sankey diagram + table with keywords
        """
        # Subtopic evolution analysis failed, one step is not enough to analyze evolution
        if self.analyzer.evolution_df is None or not self.analyzer.evolution_kwds:
            return None

        n_steps = len(self.analyzer.evolution_df.columns) - 2

        edges, nodes_data = ExperimentalPlotPreprocessor.subtopic_evolution_data(
            self.analyzer.evolution_df, self.analyzer.evolution_kwds, n_steps
        )

        value_dim = hv.Dimension('Amount', unit=None)
        nodes_ds = hv.Dataset(nodes_data, 'index', 'label')
        topic_evolution = hv.Sankey((edges, nodes_ds), ['From', 'To'], vdims=value_dim)
        topic_evolution.opts(labels='label', width=PLOT_WIDTH, height=TALL_PLOT_HEIGHT,
                             show_values=False, cmap='tab20',
                             edge_color=dim('To').str(), node_color=dim('index').str())

        if n_steps > 3:
            columns, source = ExperimentalPlotPreprocessor.subtopic_evolution_keywords_data(
                self.analyzer.evolution_kwds
            )
            subtopic_keywords = DataTable(source=source, columns=columns, width=PLOT_WIDTH, index_position=None)

            return column(hv.render(topic_evolution, backend='bokeh'), subtopic_keywords)

        return hv.render(topic_evolution, backend='bokeh')
