import logging
import re

import numpy as np
import pandas as pd
from nltk import SnowballStemmer
from scipy.sparse import lil_matrix

from pysrc.papers.db.search_error import SearchError

logger = logging.getLogger(__name__)


def preprocess_search_query_for_postgres(query, min_search_words):
    logger.debug(f'Preprocess search string for Postgres full text lookup query: {query}')
    query = query.strip(', ')  # Strip trailing spaces and commas
    if ',' in query:
        qor = ''
        for p in query.split(','):
            if len(qor) > 0:
                qor += ' | '
            pp = preprocess_search_query_for_postgres(p.strip(), min_search_words)
            qor += pp
        return qor

    # Whitespaces normalization, see #215
    processed = re.sub('[ ]{2,}|\t+', ' ', query.strip())
    if len(processed) == 0:
        raise SearchError('Empty query')

    if len(processed) == 0:
        raise SearchError('Empty query')

    # Check query complexity
    if len(re.split('[ -]', processed)) < min_search_words:
        raise SearchError(f'Please use more specific query with >= {min_search_words} words. Query: {query}')

    # Check query complexity for similar words
    stemmer = SnowballStemmer('english')
    stems = set([stemmer.stem(word) for word in [re.sub("[\"-]|('s$)", '', w) for w in processed.split(' ')]])
    if len(stems) + len(processed.split('-')) - 1 < min_search_words:
        raise SearchError(f'Please use query with >= {min_search_words} different words. Query: {query}')

    if len(re.findall('"', processed)) % 2 == 1:
        raise SearchError(f'Illegal search query, please use search terms or '
                          f'all the query wrapped in "" for phrasal search. Query: {query}')

    # Looking for complete phrases
    phrases = []
    for phrase in re.findall('"[^"]*"', processed):
        phrase_strip = phrase.strip('"').strip()
        if ' ' not in phrase_strip:
            raise SearchError(f'Illegal search query, please use search terms or '
                              f'all the query wrapped in "" for phrasal search. Query: {query}')
        phrases.append('<->'.join(phrase_strip.split(' ')))
        processed = processed.replace(phrase, "").strip()
    phrases_result = ' & '.join(phrases) if phrases else ''

    # Processing words
    rest_words = re.sub('[ ]{2,}', ' ', processed).strip()
    words = [re.sub("'s$", '', w) for w in rest_words.split(' ')]  # Fix apostrophes
    words_result = ' & '.join(words) if words else ''

    if phrases_result != '':
        return phrases_result if words_result == '' else f'{phrases_result} & {words_result}'
    elif words_result != '':
        return words_result
    else:
        raise SearchError(f'Illegal search query, please use search terms or '
                          f'all the query wrapped in "" for phrasal search. Query: {query}')


def no_stemming_filter_for_phrases(query_str):
    ors = []
    for q in query_str.lower().split('|'):
        ands = []
        for w in re.split('&', q.strip()):
            if '<->' in w.strip():  # Process only phrases
                ands.append(
                    '(P.title IS NOT NULL AND P.title ~* \'(\\m' + w.strip().replace("<->", "\\s+") + '\\M)\'' +
                    ' OR ' +
                    'P.abstract IS NOT NULL AND P.abstract ~* \'(\\m' + w.strip().replace("<->", "\\s+") + '\\M)\')'
                )
        if len(ands) > 0:
            ors.append(' AND '.join(ands))
    if len(ors) > 0:
        return ' AND (' + ' OR '.join(ors) + ')'
    else:
        return ''


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
    df['year'].fillna(1970, inplace=True)
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
