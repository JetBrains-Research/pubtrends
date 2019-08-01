import math

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource


class PlotPreprocessor:

    @staticmethod
    def chord_diagram_data_source(cocitation_graph, df, comps, comp_other, palette):
        # Co-citation graph nodes and weighted edges
        nodes = list(cocitation_graph.nodes())
        edges = list(cocitation_graph.edges(data=True))

        # Using merge left keeps order
        gdf = pd.merge(pd.Series(nodes, dtype=object).reset_index().rename(
            columns={0: 'id'}),
            df[['id', 'title', 'authors', 'year', 'total', 'comp']],
            how='left'
        ).sort_values(by=['comp', 'total'], ascending=[True, False])

        sorted_nodes = list(gdf['id'].values)

        gdf['index'] = gdf['id']
        gdf['colors'] = [palette[comps[n]] for n in sorted_nodes]
        gdf['year'] = gdf['year'].replace(np.nan, "Undefined")
        gdf['total'] = gdf['total']
        log_total = np.log(gdf['total'])
        gdf['size'] = (log_total / np.max(log_total)) * 5 + 5
        gdf['topic'] = [f'#{comps[n]}{" OTHER" if comps[n] == comp_other else ""}' for n in sorted_nodes]

        edge_starts = []
        edge_ends = []
        edge_weights = []
        for start, end, data in edges:
            edge_starts.append(start)
            edge_ends.append(end)
            edge_weights.append(min(data['weight'], 20))

        edge_colors = []
        edge_alphas = []
        for edge_start, edge_end in zip(edge_starts, edge_ends):
            if comps[edge_start] == comps[edge_end]:
                edge_colors.append(palette[comps[edge_start]])
                edge_alphas.append(0.1)
            else:
                edge_colors.append('grey')
                edge_alphas.append(0.05)

        layout, xs, ys = PlotPreprocessor.chord_diagram_layout(sorted_nodes, edges)

        node_data_source = ColumnDataSource(gdf)

        # xs and ys contain data for bezier paths
        edge_data_source = ColumnDataSource(
            dict(start=edge_starts, end=edge_ends, edge_weights=edge_weights,
                 xs=xs, ys=ys, edge_colors=edge_colors, edge_alphas=edge_alphas)
        )

        return layout, node_data_source, edge_data_source

    @staticmethod
    def chord_diagram_layout(nodes, edges):
        # Draw quadratic bezier paths
        def bezier(start, end, steps=10, c=1.5):
            return [(1 - s) ** c * start + s ** c * end for s in np.linspace(0, 1, steps)]

        # Node positions
        circ = [i * 2 * math.pi / len(nodes) for i in range(len(nodes))]
        x = [math.cos(i) for i in circ]
        y = [math.sin(i) for i in circ]
        layout = dict(zip(list(nodes), zip(x, y)))

        # Edge paths
        xs, ys = [], []
        for edge_start, edge_end, _ in edges:
            sx, sy = layout[edge_start]
            ex, ey = layout[edge_end]
            xs.append(bezier(sx, ex))
            ys.append(bezier(sy, ey))

        return layout, xs, ys
