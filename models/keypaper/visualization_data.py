import math
from itertools import product as cart_product

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, TableColumn

from .utils import cut_authors_list


class PlotPreprocessor:

    @staticmethod
    def hex2rgb(color):
        return [int(color[pos:pos + 2], 16) for pos in range(1, 7, 2)]

    @staticmethod
    def chord_diagram_data(cocitation_graph, df, comps, comp_other, palette):
        # Co-citation graph nodes and weighted edges
        nodes = list(cocitation_graph.nodes())
        edges = list(cocitation_graph.edges())
        weighted_edges = list(cocitation_graph.edges(data=True))

        # Using merge left keeps order
        gdf = pd.merge(pd.Series(nodes, dtype=object).reset_index().rename(
            columns={0: 'id'}),
            df[['id', 'title', 'authors', 'year', 'total', 'comp']],
            how='left'
        ).sort_values(by=['comp', 'total'], ascending=[True, False])

        sorted_nodes = list(gdf['id'].values)

        # Index column is required by GraphRenderer, see corresponding Plotter function
        gdf['index'] = gdf['id']
        gdf['colors'] = [palette[comps[n]] for n in sorted_nodes]
        gdf['year'] = gdf['year'].replace(np.nan, "Undefined")
        log_total = np.log(gdf['total'])
        gdf['size'] = (log_total / np.max(log_total)) * 5 + 5
        gdf['topic'] = [f'#{comps[n]}{" OTHER" if comps[n] == comp_other else ""}' for n in sorted_nodes]
        gdf['authors'] = gdf['authors'].apply(lambda authors: cut_authors_list(authors))

        edge_starts = []
        edge_ends = []
        edge_weights = []
        for start, end, data in weighted_edges:
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
        for edge_start, edge_end in edges:
            sx, sy = layout[edge_start]
            ex, ey = layout[edge_end]
            xs.append(bezier(sx, ex))
            ys.append(bezier(sy, ey))

        return layout, xs, ys

    @staticmethod
    def heatmap_clusters_data(cocitation_graph, df, comp_sizes):
        clusters = list(map(str, comp_sizes.keys()))
        n_comps = len(clusters)

        # Load edge data to DataFrame
        links = pd.DataFrame(cocitation_graph.edges(data=True), columns=['source', 'target', 'value'])
        links['value'] = links['value'].apply(lambda data: data['weight'])

        # Map each node to corresponding component
        cluster_edges = links.merge(df[['id', 'comp']], how='left', left_on='source', right_on='id') \
                             .merge(df[['id', 'comp']], how='left', left_on='target', right_on='id')

        # Calculate connectivity matrix for components
        cluster_edges = cluster_edges.groupby(['comp_x', 'comp_y'])['value'].sum().reset_index()
        connectivity_matrix = [[0] * n_comps for _ in range(n_comps)]
        for index, row in cluster_edges.iterrows():
            cx, cy = row['comp_x'], row['comp_y']
            connectivity_matrix[cx][cy] += row['value']
            if cx != cy:
                connectivity_matrix[cy][cx] += row['value']
        cluster_edges = pd.DataFrame([{'comp_x': i, 'comp_y': j, 'value': connectivity_matrix[i][j]}
                                      for i, j in cart_product(range(n_comps), range(n_comps))])

        # Density = number of co-citations between subtopics / (size of subtopic 1 * size of subtopic 2)
        def get_density(row):
            return row['value'] / (comp_sizes[row['comp_x']] * comp_sizes[row['comp_y']])

        cluster_edges['density'] = cluster_edges.apply(lambda row: get_density(row), axis=1)
        cluster_edges['comp_x'] = cluster_edges['comp_x'].astype(str)
        cluster_edges['comp_y'] = cluster_edges['comp_y'].astype(str)
        return cluster_edges, clusters

    @staticmethod
    def component_ratio_data(df, palette):
        assigned_comps = df[df['comp'] >= 0]
        comp_size = dict(assigned_comps.groupby('comp')['id'].count())
        total_papers = sum(assigned_comps['comp'] >= 0)

        # comps are reversed to display in descending order
        comps = list(reversed(list(map(str, comp_size.keys()))))
        ratios = [100 * comp_size[int(c)] / total_papers for c in comps]
        colors = [palette[int(c)] for c in comps]
        source = ColumnDataSource(data=dict(comps=comps, ratios=ratios, colors=colors))
        return comps, source

    @staticmethod
    def component_size_summary_data(df, comps, min_year, max_year):
        n_comps = len(comps)
        components = [str(i) for i in range(n_comps)]
        years = list(range(min_year, max_year + 1))
        data = {'years': years}
        for c in range(n_comps):
            data[str(c)] = [len(df[np.logical_and(df['comp'] == c, df['year'] == y)])
                            for y in range(min_year, max_year + 1)]
        return components, data

    @staticmethod
    def subtopic_evolution_data(df, kwds, n_steps):
        def sort_nodes_key(node):
            y, c = node[0].split(' ')
            return int(y), -int(c)

        cols = df.columns[2:]
        pairs = list(zip(cols, cols[1:]))
        nodes = set()
        edges = []
        for now, then in pairs:
            nodes_now = [f'{now} {c}' for c in df[now].unique()]
            nodes_then = [f'{then} {c}' for c in df[then].unique()]

            inner = {node: 0 for node in nodes_then}
            changes = {node: inner.copy() for node in nodes_now}
            for pmid, comp in df.iterrows():
                c_now, c_then = comp[now], comp[then]
                changes[f'{now} {c_now}'][f'{then} {c_then}'] += 1

            for v in nodes_now:
                for u in nodes_then:
                    if changes[v][u] > 0:
                        edges.append((v, u, changes[v][u]))
                        nodes.add(v)
                        nodes.add(u)

        nodes_data = []
        for node in nodes:
            year, c = node.split(' ')
            if int(c) >= 0:
                if n_steps < 4:
                    label = f"{year} {', '.join(kwds[int(year)][int(c)][:5])}"
                else:
                    label = node
            else:
                label = f"Published after {year}"
            nodes_data.append((node, label))
        nodes_data = sorted(nodes_data, key=sort_nodes_key, reverse=True)

        return edges, nodes_data

    @staticmethod
    def subtopic_evolution_keywords_data(kwds):
        years = []
        subtopics = []
        keywords = []
        for year, comps in kwds.items():
            for c, kwd in comps.items():
                if c >= 0:
                    years.append(year)
                    subtopics.append(c)
                    keywords.append(', '.join(kwd))
        data = dict(
            years=years,
            subtopics=subtopics,
            keywords=keywords
        )
        source = ColumnDataSource(data)
        source.add(pd.RangeIndex(start=1, stop=len(keywords), step=1), 'index')
        columns = [
            TableColumn(field="index", title="#", width=20),
            TableColumn(field="years", title="Year", width=50),
            TableColumn(field="subtopics", title="Subtopic", width=50),
            TableColumn(field="keywords", title="Keywords", width=800),
        ]
        return columns, source

    @staticmethod
    def article_view_data_source(df, min_year, max_year, width=760):
        df_local = df[['id', 'title', 'year', 'total', 'authors', 'comp']].copy()

        # Size is based on the citations number, at least 1
        df_local['size'] = 1 + np.log(df['total'] + 1)

        # Calculate max size of circles to avoid overlapping along x-axis
        max_radius_screen_units = width / (max_year - min_year + 1)
        size_scaling_coefficient = max_radius_screen_units / df_local['size'].max()
        df_local['size'] = df_local['size'] * size_scaling_coefficient

        # Replace NaN values with Undefined for tooltips
        df_local['year'] = df_local['year'].replace(np.nan, "Undefined")

        df_local['authors'] = df_local['authors'].apply(lambda authors: cut_authors_list(authors))

        df_local['comp'] = df_local['comp']

        return ColumnDataSource(df_local)

    @staticmethod
    def papers_statistics_data(df):
        cols = ['year', 'id', 'title', 'authors']
        df_stats = df[cols].groupby(['year']).size().reset_index(name='counts')
        return ColumnDataSource(df_stats)
