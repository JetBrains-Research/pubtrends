import logging

import networkx as nx
import numpy as np
import pandas as pd
from itertools import product as cart_product

from models.keypaper.analyzer import KeyPaperAnalyzer
from models.keypaper.utils import cut_authors_list

logger = logging.getLogger(__name__)


class PlotPreprocessor:

    @staticmethod
    def heatmap_clusters_data(similarity_graph, df, comp_sizes):
        logger.debug('Computing heatmap clusters_data')
        # c + 1 is used to start numbering with 1
        clusters = list(map(str, [c + 1 for c in comp_sizes.keys()]))
        n_comps = len(clusters)

        logger.debug('Load edge data to DataFrame')
        edges = similarity_graph.size()
        sources = [None] * edges
        targets = [None] * edges
        values = [0.0] * edges
        i = 0
        for u, v, data in similarity_graph.edges(data=True):
            sources[i] = u
            targets[i] = v
            values[i] = data.get('cocitation', 0) * KeyPaperAnalyzer.SIMILARITY_COCITATION + \
                data.get('bibcoupling', 0) * KeyPaperAnalyzer.SIMILARITY_BIBLIOGRAPHIC_COUPLING + \
                data.get('citation', 0) * KeyPaperAnalyzer.SIMILARITY_CITATION
            i += 1
        links_df = pd.DataFrame(data={'source': sources, 'target': targets, 'value': values})

        logger.debug('Map each node to corresponding component')
        cluster_edges = links_df.merge(df[['id', 'comp']], how='left', left_on='source', right_on='id') \
            .merge(df[['id', 'comp']], how='left', left_on='target', right_on='id')

        logger.debug('Calculate connectivity matrix for components')
        cluster_edges = cluster_edges.groupby(['comp_x', 'comp_y'])['value'].sum().reset_index()
        connectivity_matrix = [[0] * n_comps for _ in range(n_comps)]
        for index, row in cluster_edges.iterrows():
            cx, cy = int(row['comp_x']), int(row['comp_y'])
            connectivity_matrix[cx][cy] += row['value']
            if cx != cy:
                connectivity_matrix[cy][cx] += row['value']
        cluster_edges = pd.DataFrame([{'comp_x': i, 'comp_y': j, 'value': connectivity_matrix[i][j]}
                                      for i, j in cart_product(range(n_comps), range(n_comps))])

        logger.debug('Density = value between subtopics / (size of subtopic 1 * size of subtopic 2)')

        def get_density(row):
            return row['value'] / (comp_sizes[row['comp_x']] * comp_sizes[row['comp_y']])

        cluster_edges['density'] = cluster_edges.apply(lambda row: get_density(row), axis=1)
        cluster_edges['comp_x'] = cluster_edges['comp_x'].apply(lambda x: x + 1).astype(str)
        cluster_edges['comp_y'] = cluster_edges['comp_y'].apply(lambda x: x + 1).astype(str)
        return cluster_edges, clusters

    @staticmethod
    def component_ratio_data(df):
        assigned_comps = df[df['comp'] >= 0]
        comp_size = dict(assigned_comps.groupby('comp')['id'].count())
        total_papers = sum(assigned_comps['comp'] >= 0)

        # comps are reversed to display in descending order
        comps = list(reversed(list(comp_size.keys())))
        ratios = [100 * comp_size[c] / total_papers for c in comps]

        # c + 1 is used to start numbering from 1
        comps = list(map(str, [c + 1 for c in comps]))
        return comps, ratios

    @staticmethod
    def component_size_summary_data(df, comps, min_year, max_year):
        n_comps = len(comps)
        components = [str(i + 1) for i in range(n_comps)]
        years = list(range(min_year, max_year + 1))
        data = {'years': years}
        for c in range(n_comps):
            data[str(c + 1)] = [len(df[np.logical_and(df['comp'] == c, df['year'] == y)])
                                for y in range(min_year, max_year + 1)]
        return components, data

    @staticmethod
    def article_view_data_source(df, min_year, max_year, components_split, width=760):
        columns = ['id', 'title', 'year', 'type', 'total', 'authors', 'comp']
        df_local = df[columns].copy()

        # Replace NaN values with Undefined for tooltips
        df_local['year'] = df_local['year'].replace(np.nan, "Undefined")

        # Correction of same year / total
        df_local['count'] = 1  # Temporarily column to count clashes
        if components_split:
            df_counts = df_local[['comp', 'year', 'total', 'count']].groupby(['comp', 'year', 'total']).sum()
        else:
            df_counts = df_local[['year', 'total', 'count']].groupby(['year', 'total']).sum()
        df_counts['delta'] = 0
        dft = pd.DataFrame(columns=columns + ['y'])
        for _, r in df_local.iterrows():
            id, title, year, type, total, authors, comp, _ = r  # Ignore count
            if components_split:
                cd = df_counts.loc[(comp, year, total)]
            else:
                cd = df_counts.loc[(year, total)]
            c, d = cd['count'], cd['delta']
            # Make papers with same year and citations have different y values
            dft.loc[len(dft)] = (id, title, year, type, total, authors, comp,
                                 # Fix to show not cited papers on log axis
                                 max(1, total) + (d - int(c / 2)) / float(c))
            cd['delta'] += 1  # Increase delta
        df_local = dft

        # Size is based on the citations number, at least 1
        # Use list here to correctly process single element df
        df_local['size'] = 1 + np.log(list(df_local['total'] + 1))

        # Calculate max size of circles to avoid overlapping along x-axis
        max_radius_screen_units = width / max(max_year - min_year + 1, 30)
        size_scaling_coefficient = max_radius_screen_units / df_local['size'].max()
        df_local['size'] = df_local['size'] * size_scaling_coefficient

        # Split authors
        df_local['authors'] = df_local['authors'].apply(lambda authors: cut_authors_list(authors))

        return df_local

    @staticmethod
    def papers_statistics_data(df):
        cols = ['year', 'id', 'title', 'authors']
        df_stats = df[cols].groupby(['year']).size().reset_index(name='counts')
        years_df = pd.DataFrame({'year': np.arange(np.min(df_stats['year']), np.max(df_stats['year']) + 1, 1)})
        return pd.merge(left=years_df, left_on='year', right=df_stats, right_on='year', how='left').fillna(0)

    @staticmethod
    def article_citation_dynamics_data(df, pid):
        sel = df[df['id'] == pid]
        year = int(sel['year'].values[0])

        x = [col for col in df.columns if isinstance(col, (int, float)) and col >= year]
        y = list(sel[x].values[0])
        return dict(x=x, y=y)

    @staticmethod
    def dump_citations_graph_cytoscape(df, citations_graph):
        logger.debug('Mapping citations graph to cytoscape JS')
        graph = citations_graph.copy()

        logger.debug('Collect attributes for nodes')
        attrs = {}
        for node in df['id']:
            if not graph.has_node(node):
                graph.add_node(node)

            sel = df[df['id'] == node]
            comp = int(sel['comp'].values[0])
            attrs[node] = {
                'title': sel['title'].values[0],
                'abstract': sel['abstract'].values[0],
                'authors': cut_authors_list(sel['authors'].values[0]),
                'year': int(sel['year'].values[0]),
                'cited': int(sel['total'].values[0]),
                'comp': comp + 1,  # For visualization consistency
            }
        nx.set_node_attributes(graph, attrs)

        logger.debug('Group not connected nodes in groups by cluster')
        comp_groups = set()
        cytoscape_data = nx.cytoscape_data(graph)["elements"]
        for node_cs in cytoscape_data['nodes']:
            nid = node_cs['data']['id']
            if graph.degree(nid) == 0:
                comp = node_cs['data']['comp']
                if comp not in comp_groups:
                    comp_group = {
                        'group': 'nodes',
                        'data': {
                            'id': f'subtopic_{comp}',
                            'comp': comp,
                        },
                        'classes': 'group'
                    }
                    comp_groups.add(comp)
                    cytoscape_data['nodes'].append(comp_group)
                node_cs['data']['parent'] = f'subtopic_{comp}'

        logger.debug('Done citations graph in cytoscape JS')
        return cytoscape_data

    @staticmethod
    def dump_structure_graph_cytoscape(df, structure_graph):
        logger.info('Mapping structure graph to cytoscape JS')
        graph = structure_graph.copy()

        logger.info('Collect attributes for nodes')
        attrs = {}
        for edge in df['id']:
            sel = df[df['id'] == edge]
            comp = int(sel['comp'].values[0])
            attrs[edge] = {
                'title': sel['title'].values[0],
                'abstract': sel['abstract'].values[0],
                'authors': cut_authors_list(sel['authors'].values[0]),
                'year': int(sel['year'].values[0]),
                'cited': int(sel['total'].values[0]),
                'comp': comp + 1,  # For visualization consistency
            }
        nx.set_node_attributes(graph, attrs)
        cytoscape_data = nx.cytoscape_data(graph)["elements"]

        logger.info('Done structure graph to cytoscape JS')
        return cytoscape_data