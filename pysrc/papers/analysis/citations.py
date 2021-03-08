import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def find_top_cited_papers(df, n_papers):
    papers_to_show = min(n_papers, len(df))
    top_cited_df = df.sort_values(by='total', ascending=False).iloc[:papers_to_show, :]
    top_cited_papers = set(top_cited_df['id'].values)
    return top_cited_papers, top_cited_df


def find_max_gain_papers(df, citation_years):
    max_gain_data = []
    for year in citation_years:
        max_gain = df[year].astype(int).max()
        if max_gain > 0:
            sel = df[df[year] == max_gain]
            max_gain_data.append([year, str(sel['id'].values[0]),
                                  sel['title'].values[0],
                                  sel['authors'].values[0],
                                  sel['year'].values[0], max_gain])

    max_gain_df = pd.DataFrame(max_gain_data,
                               columns=['year', 'id', 'title', 'authors',
                                        'paper_year', 'count'])
    max_gain_papers = set(max_gain_df['id'].values)
    return max_gain_papers, max_gain_df


def find_max_relative_gain_papers(df, citation_years):
    current_sum = pd.Series(np.zeros(len(df), ))
    df_rel = df.loc[:, ['id', 'title', 'authors', 'year']]
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
                                      sel['authors'].values[0],
                                      sel['year'].values[0], max_rel_gain])

    max_rel_gain_df = pd.DataFrame(max_rel_gain_data,
                                   columns=['year', 'id', 'title', 'authors',
                                            'paper_year', 'rel_gain'])
    return list(max_rel_gain_df['id'].values), max_rel_gain_df


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
    cocit_grouped_df = cocit_df.groupby(['cited_1', 'cited_2', 'year']).count().reset_index()
    cocit_grouped_df = cocit_grouped_df.pivot_table(index=['cited_1', 'cited_2'],
                                                    columns=['year'], values=['citing']).reset_index()
    cocit_grouped_df = cocit_grouped_df.replace(np.nan, 0)
    cocit_grouped_df['total'] = cocit_grouped_df.iloc[:, 2:].sum(axis=1).astype(int)
    cocit_grouped_df = cocit_grouped_df.sort_values(by='total', ascending=False)

    return cocit_grouped_df


def merge_citation_stats(pub_df, cit_df):
    df = pd.merge(pub_df, cit_df, on='id', how='outer')

    # Fill only new columns to preserve year NaN values
    df[cit_df.columns] = df[cit_df.columns].fillna(0)
    df.authors = df.authors.fillna('')

    # Publication and citation year range
    citation_years = [int(col) for col in list(df.columns) if isinstance(col, (int, float))]
    min_year, max_year = int(df['year'].min()), int(df['year'].max())

    return df, min_year, max_year, citation_years