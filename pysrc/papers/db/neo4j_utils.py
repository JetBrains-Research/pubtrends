import re

from nltk import SnowballStemmer


def preprocess_search_query_for_neo4j(query, min_search_words):
    """ Preprocess search string for Neo4j full text lookup """
    if ',' in query:
        qor = ''
        for p in query.split(','):
            if len(qor) > 0:
                qor += ' OR '
            pp = preprocess_search_query_for_neo4j(p.strip(), min_search_words)
            if ' AND ' in pp:
                qor += f'({pp})'
            else:
                qor += pp
        return qor
    processed = re.sub('[ ]{2,}', ' ', query.strip())  # Whitespaces normalization, see #215
    if len(processed) == 0:
        raise Exception('Empty query')
    processed = re.sub('[^0-9a-zA-Z\'"\\-\\.+ ]', '', processed)  # Remove unknown symbols
    if len(processed) == 0:
        raise Exception('Illegal character(s), only English letters, numbers, '
                        f'and +- signs are supported. Query: {query}')
    if len(re.split('[ -]', processed)) < min_search_words:
        raise Exception(f'Please use more specific query with >= {min_search_words} words. Query: {query}')
    # Looking for complete phrase
    if re.match('^"[^"]+"$', processed):
        return '"' + re.sub('"', '', processed) + '"'
    elif re.match('^[^"]+$', processed):
        words = [re.sub("'s$", '', w) for w in processed.split(' ')]  # Fix apostrophes
        stemmer = SnowballStemmer('english')
        stems = set([stemmer.stem(word) for word in words])  # Avoid similar words
        if len(stems) + len(processed.split('-')) - 1 < min_search_words:
            raise Exception(f'Please use query with >= {min_search_words} different words. Query: {query}')
        return ' AND '.join([w if '-' not in w else f'"{w}"' for w in words])  # Dashed terms should be quoted
    raise Exception(f'Illegal search query, please use search terms or '
                    f'all the query wrapped in "" for phrasal search. Query: {query}')
