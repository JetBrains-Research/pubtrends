import re

import pandas as pd
from nltk import SnowballStemmer

from pysrc.papers.db.search_error import SearchError


def preprocess_search_query_for_postgres(query, min_search_words):
    """ Preprocess search string for Postgres full text lookup """
    if ',' in query:
        qor = ''
        for p in query.split(','):
            if len(qor) > 0:
                qor += ' | '
            pp = preprocess_search_query_for_postgres(p.strip(), min_search_words)
            qor += pp
        return qor
    processed = re.sub('[ ]{2,}', ' ', query.strip())  # Whitespaces normalization, see #215
    if len(processed) == 0:
        raise SearchError('Empty query')
    processed = re.sub('[^0-9a-zA-Z\'"\\-\\.+ ]', '', processed)  # Remove unknown symbols
    if len(processed) == 0:
        raise SearchError('Illegal character(s), only English letters, numbers, '
                          f'and +- signs are supported. Query: {query}')
    if len(re.split('[ -]', processed)) < min_search_words:
        raise SearchError(f'Please use more specific query with >= {min_search_words} words. Query: {query}')
    # Looking for complete phrase
    if re.match('^"[^"]+"$', processed):
        return '<->'.join(re.sub('[\'"]', '', processed).split(' '))
    elif re.match('^[^"]+$', processed):
        words = [re.sub("'s$", '', w) for w in processed.split(' ')]  # Fix apostrophes
        stemmer = SnowballStemmer('english')
        stems = set([stemmer.stem(word) for word in words])  # Avoid similar words
        if len(stems) + len(processed.split('-')) - 1 < min_search_words:
            raise SearchError(f'Please use query with >= {min_search_words} different words. Query: {query}')
        return ' & '.join(words)
    raise SearchError(f'Illegal search query, please use search terms or '
                      f'all the query wrapped in "" for phrasal search. Query: {query}')


def process_cocitations_postgres(cursor):
    data = []
    lines = 0
    for row in cursor:
        lines += 1
        citing, year, cited_list = row
        cited_list.sort()
        for i in range(len(cited_list)):
            for j in range(i + 1, len(cited_list)):
                data.append((str(citing), str(cited_list[i]), str(cited_list[j]), year))
    df = pd.DataFrame(data, columns=['citing', 'cited_1', 'cited_2', 'year'], dtype=object)
    df['year'] = df['year'].astype(int)
    return df, lines


def process_bibliographic_coupling_postgres(cursor):
    data = []
    lines = 0
    for row in cursor:
        lines += 1
        _, citing_list = row
        citing_list.sort()
        for i in range(len(citing_list)):
            for j in range(i + 1, len(citing_list)):
                data.append((str(citing_list[i]), str(citing_list[j]), 1))
    df = pd.DataFrame(data, columns=['citing_1', 'citing_2', 'total'], dtype=object)
    df = df.groupby(['citing_1', 'citing_2']).sum().reset_index()
    return df, lines
