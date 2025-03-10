import logging

import numpy as np
import pandas as pd

from pysrc.papers.utils import reorder_publications

logger = logging.getLogger(__name__)


def find_top_cited_papers(df, n_papers):
    papers_to_show = min(n_papers, len(df))
    top_cited_df = df.sort_values(by='total', ascending=False).iloc[:papers_to_show, :]
    return list(top_cited_df['id']), top_cited_df


def find_max_gain_papers(df, citation_years):
    max_gain_data = []
    for year in citation_years:
        max_gain = df[year].astype(int).max()
        if max_gain > 0:
            sel = df[df[year] == max_gain]
            max_gain_data.append([year, str(sel['id'].values[0]),
                                  sel['title'].values[0],
                                  sel['journal'].values[0],
                                  sel['authors'].values[0],
                                  sel['year'].values[0], max_gain])

    max_gain_df = pd.DataFrame(max_gain_data,
                               columns=['year', 'id', 'title', 'journal', 'authors', 'paper_year', 'count'],
                               dtype=object)
    return list(max_gain_df['id'].values), max_gain_df


def find_max_relative_gain_papers(df, citation_years):
    current_sum = pd.Series(np.zeros(len(df), ))
    df_rel = df.loc[:, ['id', 'title', 'journal', 'authors', 'year']]
    for year in citation_years:
        df_rel[year] = df[year] / (current_sum + (current_sum == 0))
        current_sum += df[year]

    max_rel_gain_data = []
    for year in citation_years:
        max_rel_gain = df_rel[year].max()
        # Ignore less than 1 percent relative gain
        if max_rel_gain >= 0.01:
            sel = df_rel[df_rel[year] == max_rel_gain]
            max_rel_gain_data.append([year, str(sel['id'].values[0]),
                                      sel['title'].values[0],
                                      sel['journal'].values[0],
                                      sel['authors'].values[0],
                                      sel['year'].values[0], max_rel_gain])

    max_rel_gain_df = pd.DataFrame(max_rel_gain_data,
                                   columns=['year', 'id', 'title', 'journal', 'authors', 'paper_year', 'rel_gain'],
                                   dtype=object)
    return list(max_rel_gain_df['id']), max_rel_gain_df


def build_cit_stats_df(cits_by_year_df, n_papers):
    # Get citation stats with columns 'id', year_1, ..., year_N and fill NaN with 0
    df = cits_by_year_df.pivot(index='id', columns='year', values='count').reset_index().fillna(0)

    # Fix column names from float 'YYYY.0' to int 'YYYY'
    df = df.rename({col: int(col) for col in df.columns if col != 'id'})

    df['total'] = df.iloc[:, 1:].sum(axis=1)
    df = df.sort_values(by='total', ascending=False)
    logger.debug(f'Loaded citation stats for {len(df)} of {n_papers} papers')

    return df


def build_cocit_grouped_df(cocit_df):
    logger.debug('Aggregating co-citations')
    if len(cocit_df) == 0:
        return pd.DataFrame(data=[], columns=['cited_1', 'cited_2', 'year', 'total'], dtype=object)
    cocit_grouped_df = cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
    cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                    columns=['year'], values=['citing']).reset_index()
    cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
    cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1).astype(int)
    cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)
    # Cleanup columns after grouping
    cocit_grouped_df.reset_index(inplace=True)
    cocit_grouped_df.columns = [a if b == '' else b for a, b in cocit_grouped_df.columns.values.tolist()]
    cocit_grouped_df.drop('index', axis='columns', inplace=True)
    for c in cocit_grouped_df.columns:
        if c not in ['index', 'cited_1', 'cited_2', 'total']:
            cocit_grouped_df[c] = cocit_grouped_df[c].astype(int)

    return cocit_grouped_df


def merge_citation_stats(ids, pub_df, cit_df):
    df = pd.merge(pub_df, cit_df, on='id', how='outer')
    # restore original publications order, important when analysing single paper
    df = reorder_publications(ids, df)
    # Fill only new columns to preserve year NaN values
    df[cit_df.columns] = df[cit_df.columns].fillna(0)
    df.authors = df.authors.fillna('')

    # Publication and citation year range
    citation_years = [int(col) for col in list(df.columns) if isinstance(col, (int, float))]
    return df, citation_years