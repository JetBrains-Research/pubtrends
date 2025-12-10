import logging
import re

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from pysrc.papers.db.search_error import SearchError

logger = logging.getLogger(__name__)


def ints_to_vals(xs):
    return ','.join(f'{i}' for i in xs)


def strs_to_vals(xs):
    return ','.join(f'(\'{i}\')' for i in xs)


def preprocess_quotes(value):
    return re.sub("'{2,}", "'", value.strip().strip("'"))


def preprocess_search_query_for_postgres(query):
    """
    return a string that can be used in Postgres full text search query together with phrasal check queries
    """
    query = query.lower()
    # Fix apostrophes 's
    query = re.sub("'s$", '', query)
    query = re.sub("'s\\s", ' ', query)
    # Escaping
    query = re.sub("[\^&'|<>=]+", '', query)
    # Whitespaces normalization, see #215
    query = re.sub('\s{2,}', ' ', query.strip())
    query = query.strip(', ')  # Strip trailing spaces and commas

    if len(query) == 0:
        raise SearchError('Empty query')

    or_queries = []
    or_phrases = []
    if ',' in query:
        for p in query.split(','):
            query, phrase_filter = preprocess_search_query_for_postgres(p.strip())
            if len(query) > 0:
                or_queries.append(query)
            if len(phrase_filter) > 0:
                or_phrases.append(phrase_filter)
        return ' | '.join(or_queries), ' OR '.join(or_phrases)

    if len(re.findall('"', query)) % 2 == 1:
        raise SearchError(f'Illegal search query, please use search terms or '
                          f'all the query wrapped in "" for phrasal search. Query: {query}')

    # Looking for complete phrases
    phrases = []
    phrases_manuals = []
    for phrase in list(re.findall('"[^"]*"', query)):
        phrase_strip = phrase.strip('"').strip()
        if ' ' not in phrase_strip:
            raise SearchError(f'Illegal search query, please use search terms or '
                              f'all the query wrapped in "" for phrasal search. Query: {query}')
        phrases.append('<->'.join(phrase_strip.split(' ')))
        # Disable stemming-based lookup for phrases
        # see: https://github.com/JetBrains-Research/pubtrends/issues/242
        phrase_query = "'(\\m" + phrase_strip.replace(' ', '\\s+') + "\\M)'"
        phrases_manuals.append(
            f"(P.title IS NOT NULL AND P.title ~* {phrase_query} OR "
            f"P.abstract IS NOT NULL AND P.abstract ~* {phrase_query})"
        )
        # Cut phrase within the query
        query = query.replace(phrase, "").strip()

    phrases_result = ' & '.join(phrases) if phrases else ''
    phrases_manuals_result = ' AND '.join(phrases_manuals) if phrases_manuals else ''

    # Processing words
    rest_words = re.sub('\s{2,}', ' ', query).strip()
    words = rest_words.split(' ')
    words_result = ' & '.join(words) if words else ''

    if phrases_result != '':
        if words_result == '':
            return phrases_result, phrases_manuals_result
        else:
            return f'{phrases_result} & {words_result}', phrases_manuals_result
    elif words_result != '':
        return words_result, ''
    else:
        raise SearchError(f'Illegal search query, please use search terms or '
                          f'all the query wrapped in "" for phrasal search. Query: {query}')


def process_cocitations_postgres(cursor):
    data = []
    lines = 0
    for row in cursor:
        lines += 1
        if lines % 1000 == 1:
            logger.debug(f'Processed {lines} lines of cocitations')
        citing, year, cited_list = row
        cited_list.sort()
        for i in range(len(cited_list)):
            for j in range(i + 1, len(cited_list)):
                data.append((str(citing), str(cited_list[i]), str(cited_list[j]), year))
    df = pd.DataFrame(data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)
    logger.debug(f'Total {lines} lines of cocitations info')
    logger.debug(f'Total df size {len(df)}')

    # Hack for missing year in SS, see https://github.com/JetBrains-Research/pubtrends/issues/258
    df['year'] = df['year'].fillna(1970)
    df['year'] = df['year'].astype(int)
    return df


def process_bibliographic_coupling_postgres(ids, cursor):
    lines = 0
    bm = lil_matrix((len(ids), len(ids)), dtype=np.int8)
    indx = {pid: i for i, pid in enumerate(ids)}
    for row in cursor:
        lines += 1
        if lines % 1000 == 1:
            logger.debug(f'Processed {lines} lines of bibliographic coupling')
        _, citing_list = row
        citing_list.sort()
        for i in range(len(citing_list)):
            for j in range(i + 1, len(citing_list)):
                bm[indx[str(citing_list[i])], indx[str(citing_list[j])]] += 1
    logger.debug(f'Total {lines} lines of bibliographic coupling info')
    if lines > 0:
        bmc = bm.tocoo(copy=False)
        df = pd.DataFrame(dict(citing_1=[ids[i] for i in bmc.row],
                               citing_2=[ids[j] for j in bmc.col],
                               total=list(bmc.data)))
    else:
        df = pd.DataFrame(columns=['citing_1', 'citing_2', 'total'], dtype=object)
    logger.debug(f'Df size {len(df)}')
    return df
