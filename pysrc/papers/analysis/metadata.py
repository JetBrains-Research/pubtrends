import logging

import numpy as np
import pandas as pd

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


def update_edge(graph, a1, a2, name, value):
    if a1 == a2:
        return
    if a1 > a2:
        a1, a2 = a2, a1
    if not graph.has_edge(a1, a2):
        graph.add_edge(a1, a2)
    edge = graph[a1][a2]
    edge[name] = edge.get(name, 0) + value
