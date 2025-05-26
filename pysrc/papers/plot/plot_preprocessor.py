import json
import logging

import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from pysrc.papers.utils import cut_authors_list, rgb2hex

logger = logging.getLogger(__name__)


class PlotPreprocessor:

    @staticmethod
    def component_sizes(df):
        logger.debug('Processing component_sizes')
        assigned_comps = df[df['comp'] >= 0]
        d = dict(assigned_comps.groupby('comp')['id'].count())
        return [int(d[k]) for k in range(len(d))]

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
    def compute_kwds(topics_description, n):
        kwds = [(comp, ','.join([f'{t}:{v:.3f}' for t, v in vs[:n]]))
                for comp, vs in topics_description.items()]
        return pd.DataFrame(kwds, columns=['comp', 'kwd']).sort_values(by='comp')

    @staticmethod
    def article_view_data_source(df, min_year, max_year, components_split, width=760):
        columns = ['id', 'title', 'year', 'type', 'total', 'authors', 'journal', 'comp']
        df_local = df[columns].copy()

        # Correction of same year / total
        df_local['count'] = 1  # Temporarily column to count clashes
        if components_split:
            df_counts = df_local[['comp', 'year', 'total', 'count']].groupby(['comp', 'year', 'total']).sum()
        else:
            df_counts = df_local[['year', 'total', 'count']].groupby(['year', 'total']).sum()
        df_counts['delta'] = 0
        dft = pd.DataFrame(columns=columns + ['y'], dtype=object)
        for _, r in df_local.iterrows():
            pid, title, year, type, total, authors, journal, comp, _ = r
            if components_split:
                cd = df_counts.loc[(comp, year, total)]
                c, d = cd['count'], cd['delta']
                df_counts.loc[(comp, year, total), 'delta'] += 1  # Increase delta for better layout
            else:
                cd = df_counts.loc[(year, total)]
                c, d = cd['count'], cd['delta']
                df_counts.loc[(year, total), 'delta'] += 1  # Increase delta for better layout
            # Make papers with same year and citations have different y values
            dft.loc[len(dft)] = (pid, title, year, type, total, authors, journal, comp,
                                 # Fix to show not cited papers on log axis
                                 max(1, total) + (d - int(c / 2)) / float(c))
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
    def dump_similarity_graph_cytoscape(df, papers_graph):
        logger.debug('Mapping structure graph to cytoscape JS')
        papers_graph = papers_graph.copy()
        cytoscape_graph = PlotPreprocessor.dump_to_cytoscape(df, papers_graph)
        maxy = df['y'].max()
        for node_cs in cytoscape_graph['nodes']:
            nid = node_cs['data']['id']
            sel = df.loc[df['id'] == nid]
            # Adjust vertical axis with bokeh graph
            node_cs['position'] = dict(x=int(sel['x'].values[0] * 10), y=int((maxy - sel['y'].values[0]) * 6))
        return cytoscape_graph

    @staticmethod
    def dump_to_cytoscape(df, graph):
        logger.debug('Collect attributes for nodes')
        attrs = {}
        for node in df['id']:
            sel = df[df['id'] == node]
            attrs[node] = dict(
                title=sel['title'].values[0],
                authors=cut_authors_list(sel['authors'].values[0]),
                journal=sel['journal'].values[0],
                year=int(sel['year'].values[0]),
                cited=int(sel['total'].values[0]),
                topic=int(sel['comp'].values[0]),
                connections=len(list(graph.neighbors(node))),
                # These can be heavy
                abstract=sel['abstract'].values[0],
                mesh=sel['mesh'].values[0],
                keywords=sel['keywords'].values[0],
            )
        nx.set_node_attributes(graph, attrs)
        return nx.cytoscape_data(graph)['elements']

    @staticmethod
    def topics_words(kwd_df, n):
        words2show = {}
        for _, row in kwd_df.iterrows():
            comp, kwds = row[0], row[1]
            if kwds != '':  # Correctly process empty freq_kwds encoding
                words2show[comp] = [p.split(':')[0] for p in kwds.split(',')[:n]]
            else:
                words2show[comp] = []
        return words2show

    @staticmethod
    def frequent_keywords_data(freq_kwds, df, corpus_terms, corpus_counts, n):
        logger.debug('Computing frequencies of terms')
        keywords = [t for t, _ in list(freq_kwds.items())[:n]]

        logger.debug('Grouping papers by year')
        t = df[['year']].copy()
        t['i'] = range(len(t))
        papers_by_year = t[['year', 'i']].groupby('year')['i'].apply(list).to_dict()

        logger.debug('Collecting numbers of papers with term per year')
        binary_counts = corpus_counts.copy()
        binary_counts[binary_counts.nonzero()] = 1
        numbers_per_year = np.zeros(shape=(len(papers_by_year), len(corpus_terms)))
        for i, (year, iss) in enumerate(papers_by_year.items()):
            numbers_per_year[i, :] = binary_counts[iss].sum(axis=0)[0, :]

        logger.debug('Collecting top keywords with maximum sum of numbers over years')
        top_keyword_idxs = [corpus_terms.index(t) for t in keywords if t in corpus_terms]

        logger.debug('Collecting dataframe with smoothed numbers for keywords with local weighted regression')
        years = [year for year, _ in papers_by_year.items()]
        keyword_dfs = []
        for idx in top_keyword_idxs:
            keyword = corpus_terms[idx]
            numbers = numbers_per_year[:, idx]
            numbers = lowess(numbers, years, frac=0.2, return_sorted=False)
            # Some values may become negative because of smoothing
            numbers = numbers.clip(min=0.0)
            keyword_df = pd.DataFrame(data=numbers.astype(int), columns=['number'])
            keyword_df['keyword'] = keyword
            keyword_df['year'] = years
            keyword_dfs.append(keyword_df)
        keywords_df = pd.concat(keyword_dfs, axis=0).reset_index(drop=True)
        return keywords_df, years

    @staticmethod
    def get_topic_word_cloud_data(topics_description, comp, n):
        kwds = {}
        for k, v in topics_description[comp][:n]:
            for word in k.split(' '):
                kwds[word] = kwds.get(word, 0) + v
        return kwds

    @staticmethod
    def word_cloud_prepare(wc):
        word_records = [
            (word, int(position[0]), int(position[1]), int(font_size), orientation is not None, rgb2hex(color))
            for (word, count), font_size, position, orientation, color in wc.layout_
            if not word.startswith('#')  # Skip technical records for scale
        ]
        return json.dumps(dict(word_records=word_records, width=wc.width, height=wc.height))

    @staticmethod
    def get_top_papers_id_title_year_cited_topic(papers, df, n=50):
        top_papers = map(lambda v: (df[df['id'] == v], df[df['id'] == v]['total'].values[0]), papers)
        return [(el[0]['id'].values[0],
                 el[0]['title'].values[0],
                 el[0]['year'].values[0],
                 el[0]['total'].values[0],
                 el[0]['comp'].values[0])
                for el in sorted(top_papers, key=lambda x: x[1], reverse=True)[:n]]
