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

    # Whitespaces normalization, see #215
    processed = re.sub('[ ]{2,}', ' ', query.strip())
    if len(processed) == 0:
        raise SearchError('Empty query')

    processed = re.sub('[^0-9a-zA-Z\'"\\-\\.+ ]', '', processed)  # Remove unknown symbols
    if len(processed) == 0:
        raise SearchError('Illegal character(s), only English letters, numbers, '
                          f'and +- signs are supported. Query: {query}')

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


def no_stemming_filter(query_str):
    return ' AND (' + \
           ' OR '.join(
               ' AND '.join(f"position('{w.strip()}' in LOWER(P.title))>0" for w in re.split(r'(?:&|<->)+', q)) +
               ' OR ' +
               ' AND '.join(f"position('{w.strip()}' in LOWER(P.abstract))>0" for w in re.split(r'(?:&|<->)+', q))
               for q in query_str.lower().split('|')) + \
           ')'


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
    df = pd.DataFrame(data, columns=['citing', 'cited_1', 'cited_2', 'year'])
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
    df = pd.DataFrame(data, columns=['citing_1', 'citing_2', 'total'])
    if lines > 0:
        df = df.groupby(['citing_1', 'citing_2']).sum().reset_index()
    return df, lines
