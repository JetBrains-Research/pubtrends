import logging
from itertools import product as cart_product

import networkx as nx
import numpy as np
import pandas as pd

from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.utils import cut_authors_list

logger = logging.getLogger(__name__)


class PlotPreprocessor:

    @staticmethod
    def topics_similarity_data(similarity_graph, df, comp_sizes):
        logger.debug('Computing mean similarity of all edges between topics')
        # c + 1 is used to start numbering with 1
        components = list(map(str, [c + 1 for c in comp_sizes.keys()]))
        n_comps = len(components)

        logger.debug('Load edge data to DataFrame')
        edges = len(similarity_graph.edges)
        sources = [None] * edges
        targets = [None] * edges
        similarities = [0.0] * edges
        i = 0
        for u, v, data in similarity_graph.edges(data=True):
            sources[i] = u
            targets[i] = v
            similarities[i] = KeyPaperAnalyzer.get_similarity(data)
            i += 1
        similarity_df = pd.DataFrame(data={'source': sources, 'target': targets, 'similarity': similarities})

        logger.debug('Assign each paper with corresponding component / topic')
        similarity_topics_df = similarity_df.merge(df[['id', 'comp']], how='left', left_on='source', right_on='id') \
            .merge(df[['id', 'comp']], how='left', left_on='target', right_on='id')

        logger.debug('Calculate mean similarity between for topics')
        mean_similarity_topics_df = \
            similarity_topics_df.groupby(['comp_x', 'comp_y'])['similarity'].mean().reset_index()
        similarity_matrix = [[0] * n_comps for _ in range(n_comps)]
        for index, row in mean_similarity_topics_df.iterrows():
            cx, cy = int(row['comp_x']), int(row['comp_y'])
            similarity_matrix[cx][cy] = similarity_matrix[cy][cx] = row['similarity']

        similarity_topics_df = pd.DataFrame([
            {'comp_x': i, 'comp_y': j, 'similarity': similarity_matrix[i][j]}
            for i, j in cart_product(range(n_comps), range(n_comps))
        ])
        similarity_topics_df['comp_x'] = similarity_topics_df['comp_x'].apply(lambda x: x + 1).astype(str)
        similarity_topics_df['comp_y'] = similarity_topics_df['comp_y'].apply(lambda x: x + 1).astype(str)
        return similarity_topics_df, components

    @staticmethod
    def component_ratio_data(df):
        assigned_comps = df[df['comp'] >= 0]
        comp_size = dict(assigned_comps.groupby('comp')['id'].count())
        total_papers = sum(assigned_comps['comp'] >= 0)
        comps = list(comp_size.keys())
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
        years_df = pd.DataFrame({'year': np.arange(df_stats['year'].min(), df_stats['year'].max() + 1, 1)})
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
        # Put all the nodes into the graph
        for node in df['id']:
            if not graph.has_node(node):
                graph.add_node(node)
        cytoscape_graph = PlotPreprocessor.dump_to_cytoscape(df, graph)
        logger.debug('Set pagerank for citations graph')
        pids = set(df['id'])
        for node_cs in cytoscape_graph['nodes']:
            nid = node_cs['data']['id']
            if nid in pids:
                node_cs['data']['pagerank'] = float(df.loc[df['id'] == nid]['pagerank'].values[0])
        return cytoscape_graph

    @staticmethod
    def dump_structure_graph_cytoscape(df, structure_graph):
        logger.debug('Mapping structure graph to cytoscape JS')
        cytoscape_graph = PlotPreprocessor.dump_to_cytoscape(df, structure_graph.copy())
        logger.debug('Set centrality for structure graph')
        centrality = nx.algorithms.centrality.degree_centrality(structure_graph)
        pids = set(df['id'])
        for node_cs in cytoscape_graph['nodes']:
            nid = node_cs['data']['id']
            if nid in pids:
                node_cs['data']['centrality'] = centrality[nid]
        return cytoscape_graph

    @staticmethod
    def dump_to_cytoscape(df, graph):
        logger.debug('Collect attributes for nodes')
        attrs = {}
        for node in df['id']:
            sel = df[df['id'] == node]
            topic = int(sel['comp'].values[0])
            attrs[node] = {
                'title': sel['title'].values[0],
                'abstract': sel['abstract'].values[0],
                'authors': cut_authors_list(sel['authors'].values[0]),
                'year': int(sel['year'].values[0]),
                'cited': int(sel['total'].values[0]),
                'topic': topic
            }
        nx.set_node_attributes(graph, attrs)

        logger.debug('Group not connected nodes into groups')
        topic_groups = set()
        cytoscape_data = nx.cytoscape_data(graph)['elements']
        for node_cs in cytoscape_data['nodes']:
            nid = node_cs['data']['id']
            if graph.degree(nid) == 0:
                topic = node_cs['data']['topic']
                if topic not in topic_groups:
                    topic_group = {
                        'group': 'nodes',
                        'data': {
                            'id': f'topic_{topic}',
                            'topic': topic,
                        },
                        'classes': 'group'
                    }
                    topic_groups.add(topic)
                    cytoscape_data['nodes'].append(topic_group)
                node_cs['data']['parent'] = f'topic_{topic}'
        logger.debug('Done dumping graph to cytoscape JS')
        return cytoscape_data

    @staticmethod
    def topic_evolution_data(df, kwds, n_steps):
        def sort_nodes_key(node):
            y, c = node[0].split(' ')
            return int(y), 1 if c == 'NPY' else -int(c)

        cols = df.columns[2:]
        pairs = list(zip(cols, cols[1:]))
        nodes = set()
        edges = []
        for now, then in pairs:
            nodes_now = [f'{now} {PlotPreprocessor.evolution_topic_name(c)}' for c in df[now].unique()]
            nodes_then = [f'{then} {PlotPreprocessor.evolution_topic_name(c)}' for c in df[then].unique()]

            inner = {node: 0 for node in nodes_then}
            changes = {node: inner.copy() for node in nodes_now}
            for pmid, comp in df.iterrows():
                c_now, c_then = comp[now], comp[then]
                changes[f'{now} {PlotPreprocessor.evolution_topic_name(c_now)}'][
                    f'{then} {PlotPreprocessor.evolution_topic_name(c_then)}'
                ] += 1

            for v in nodes_now:
                for u in nodes_then:
                    if changes[v][u] > 0:
                        edges.append((v, u, changes[v][u]))
                        nodes.add(v)
                        nodes.add(u)

        nodes_data = []
        for node in nodes:
            year, c = node.split(' ')
            if c == 'NPY':
                label = c
            else:
                if n_steps < 4:
                    label = f"{year} {', '.join(kwds[int(year)][int(c) - 1][:5])}"
                else:
                    label = f"{year} {c}"
            nodes_data.append((node, label))
        nodes_data = sorted(nodes_data, key=sort_nodes_key, reverse=True)

        return edges, nodes_data

    @staticmethod
    def evolution_topic_name(comp):
        if comp == -1:
            result = 'NPY'
        else:
            # Fix topic numbering to start with 1
            result = comp + 1
        return result

    @staticmethod
    def topic_evolution_keywords_data(kwds):
        kwds_data = []
        for year, comps in kwds.items():
            for comp, kwd in comps.items():
                if comp >= 0:
                    kwds_data.append((year, comp + 1, ', '.join(kwd)))
        return kwds_data
