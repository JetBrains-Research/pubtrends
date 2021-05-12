import logging
import numpy as np
import pandas as pd
import networkx as nx
import community
import itertools

logger = logging.getLogger(__name__)


def popular_journals(df, n):
    """
    Computes journals with the highest summary number of citations of papers
    :param df: Papers dataframe
    :param n: Number of journal to return
    :return:
    """
    journal_stats = df.groupby(['journal', 'comp']).size().reset_index(name='counts')
    journal_stats['journal'].replace('', np.nan, inplace=True)
    journal_stats.dropna(subset=['journal'], inplace=True)

    journal_stats.sort_values(by=['journal', 'counts'], ascending=False, inplace=True)

    journal_stats = journal_stats.groupby('journal').agg(
        {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

    journal_stats.columns = journal_stats.columns.droplevel(level=1)
    journal_stats.columns = ['journal', 'comp', 'counts', 'sum']

    journal_stats = journal_stats.sort_values(by=['sum'], ascending=False)
    return journal_stats.head(n=n)


def popular_authors(df, n):
    """
    Computes authors of the papers with highest summary number of citations
    :param df: Papers dataframe
    :param n: Number of authors to return
    :return:
    """
    author_stats = df[['authors', 'comp']].copy()
    author_stats['authors'].replace({'': np.nan, -1: np.nan}, inplace=True)
    author_stats.dropna(subset=['authors'], inplace=True)

    author_stats = split_df_list(author_stats, target_column='authors', separator=', ')
    author_stats.rename(columns={'authors': 'author'}, inplace=True)

    author_stats = author_stats.groupby(['author', 'comp']).size().reset_index(name='counts')
    author_stats.sort_values(by=['author', 'counts'], ascending=False, inplace=True)

    author_stats = author_stats.groupby('author').agg(
        {'comp': lambda x: list(x), 'counts': [lambda x: list(x), 'sum']}).reset_index()

    author_stats.columns = author_stats.columns.droplevel(level=1)
    author_stats.columns = ['author', 'comp', 'counts', 'sum']
    author_stats = author_stats.sort_values(by=['sum'], ascending=False)
    return author_stats.head(n=n)


def split_df_list(df, target_column, separator):
    """
    :param df: dataframe to split
    :param target_column: the column containing the values to split
    :param separator:  the symbol used to perform the split
    :return: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    """

    def split_list_to_rows(row, row_accumulator, target_column, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def build_authors_similarity_graph(
        df,
        texts_similarity,
        citations_graph,
        cocit_grouped_df,
        bibliographic_coupling_df,
        check_author_func):
    logger.debug('Processing papers')
    result = nx.Graph()
    for _, row in df[['authors']].iterrows():
        authors = row[0].split(', ')
        authors = authors if len(authors) <= 2 else [authors[0], authors[-1]]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a1 = authors[i]
                a2 = authors[j]
                if check_author_func(a1) and check_author_func(a2):
                    update_edge(result, a1, a2, 'authorship', 1)

    logger.debug('Processing co-citations')
    for el in cocit_grouped_df[['cited_1', 'cited_2', 'total']].values:
        start, end, cocitation = str(el[0]), str(el[1]), float(el[2])
        authors1 = df.loc[df['id'] == start]['authors'].values[0].split(', ')
        authors2 = df.loc[df['id'] == end]['authors'].values[0].split(', ')
        authors1 = authors1 if len(authors1) <= 2 else [authors1[0], authors1[-1]]
        authors2 = authors2 if len(authors2) <= 2 else [authors2[0], authors2[-1]]
        for a1, a2 in itertools.product(authors1, authors2):
            if check_author_func(a1) and check_author_func(a2):
                update_edge(result, a1, a2, 'cocitation', cocitation)

    logger.debug('Bibliographic coupling')
    if len(bibliographic_coupling_df) > 0:
        for el in bibliographic_coupling_df[['citing_1', 'citing_2', 'total']].values:
            start, end, bibcoupling = str(el[0]), str(el[1]), float(el[2])
            authors1 = df.loc[df['id'] == start]['authors'].values[0].split(', ')
            authors2 = df.loc[df['id'] == end]['authors'].values[0].split(', ')
            authors1 = authors1 if len(authors1) <= 2 else [authors1[0], authors1[-1]]
            authors2 = authors2 if len(authors2) <= 2 else [authors2[0], authors2[-1]]
            for a1, a2 in itertools.product(authors1, authors2):
                if check_author_func(a1) and check_author_func(a2):
                    update_edge(result, a1, a2, 'bibcoupling', bibcoupling)

    logger.debug('Text similarity')
    pids = list(df['id'])
    if len(df) >= 2:
        for i, pid1 in enumerate(df['id']):
            similarity_queue = texts_similarity[i]
            while not similarity_queue.empty():
                similarity, j = similarity_queue.get()
                pid2 = pids[j]
                authors1 = df.loc[df['id'] == pid1]['authors'].values[0].split(', ')
                authors2 = df.loc[df['id'] == pid2]['authors'].values[0].split(', ')
                authors1 = authors1 if len(authors1) <= 2 else [authors1[0], authors1[-1]]
                authors2 = authors2 if len(authors2) <= 2 else [authors2[0], authors2[-1]]
                for a1, a2 in itertools.product(authors1, authors2):
                    if check_author_func(a1) and check_author_func(a2):
                        update_edge(result, a1, a2, 'text', similarity)

    logger.debug('Citations')
    for u, v in citations_graph.edges:
        authors1 = df.loc[df['id'] == u]['authors'].values[0].split(', ')
        authors2 = df.loc[df['id'] == v]['authors'].values[0].split(', ')
        authors1 = authors1 if len(authors1) <= 2 else [authors1[0], authors1[-1]]
        authors2 = authors2 if len(authors2) <= 2 else [authors2[0], authors2[-1]]
        for a1, a2 in itertools.product(authors1, authors2):
            if check_author_func(a1) and check_author_func(a2):
                update_edge(result, a1, a2, 'citation', 1)

    return result


def update_edge(graph, a1, a2, name, value):
    if a1 == a2:
        return
    if a1 > a2:
        a1, a2 = a2, a1
    if not graph.has_edge(a1, a2):
        graph.add_edge(a1, a2)
    edge = graph[a1][a2]
    edge[name] = edge.get(name, 0) + value


def compute_authors_citations_and_papers(df):
    logger.debug('Compute author citations')
    author_citations = {}
    for i, row in df[['authors', 'total']].iterrows():
        authors = row['authors'].split(', ')
        #     authors = authors if len(authors) <= 2 else [authors[0], authors[-1]]
        for a in authors:
            author_citations[a] = author_citations.get(a, 0) + row['total']

    logger.debug('Compute number of papers per author')
    author_papers = {}
    for i, row in df[['title', 'authors']].iterrows():
        authors = row['authors'].split(', ')
        #     authors = authors if len(authors) <= 2 else [authors[0], authors[-1]]
        for a in authors:
            author_papers[a] = author_papers.get(a, 0) + 1

    return author_citations, author_papers


def cluster_authors(authors_graph, similarity_func, coauthorship=1000):
    connected_components = nx.number_connected_components(authors_graph)
    logger.debug(f'Authors graph has {connected_components} connected components')

    logger.debug('Compute aggregated similarity using co-authorship')
    for _, _, d in authors_graph.edges(data=True):
        d['similarity'] = coauthorship * d.get('authorship', 0) + similarity_func(d)

    logger.debug('Graph clustering via Louvain community algorithm')
    partition_louvain = community.best_partition(
        authors_graph, weight='similarity', random_state=42
    )
    logger.debug(f'Best partition {len(set(partition_louvain.values()))} components')
    components = set(partition_louvain.values())
    comp_sizes = {c: sum([partition_louvain[node] == c for node in partition_louvain.keys()]) for c in components}
    logger.debug(f'Clusters: {comp_sizes}')
    return partition_louvain
