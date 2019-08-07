import logging
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

PUBMED_ARTICLE_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='
SEMANTIC_SCHOLAR_BASE_URL = 'https://www.semanticscholar.org/paper/'


def tokenize(text, terms=None):
    is_noun_or_adj = lambda pos: (pos[:2] == 'NN' or pos == 'JJ')
    special_symbols_regex = re.compile(r'[^a-zA-Z0-9\- ]*')
    text = text.lower()

    # Filter out search terms
    if terms is not None:
        for term in terms:
            text = text.replace(term.lower(), '')

    tokenized = word_tokenize(re.sub(special_symbols_regex, '', text))
    stop_words = set(stopwords.words('english'))
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if
             is_noun_or_adj(pos) and word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = list(filter(lambda t: len(t) >= 3, [lemmatizer.lemmatize(n) for n in nouns]))
    return tokens


def get_ngrams(text, n=1):
    """1/2/3-grams computation for string"""
    tokens = tokenize(text)
    ngrams = list(tokens)
    if n > 1:
        for t1, t2 in zip(tokens[:-1], tokens[1:]):
            ngrams.append(t1 + ' ' + t2)
    if n > 2:
        for t1, t2, t3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
            ngrams.append(t1 + ' ' + t2 + ' ' + t3)
    return ngrams


def get_most_common_ngrams(titles, abstracts, number=500):
    """
    :param titles: list of titles for articles in a component
    :param abstracts: list of abstracts
    :param number: amount of most common ngrams
    :return: dictionary {ngram : frequency}
    """
    ngrams_counter = Counter()
    for text in titles:
        if isinstance(text, str):
            for ngram in get_ngrams(text):
                # More weight for titles
                ngrams_counter[ngram] += 1.5
    for text in abstracts:
        if isinstance(text, str):
            for ngram in get_ngrams(text):
                ngrams_counter[ngram] += 1
    most_common = {}
    for ngram, cnt in ngrams_counter.most_common(number):
        most_common[ngram] = cnt / len(ngrams_counter)
    return most_common


def get_subtopic_descriptions(df, comps, size=100):
    """
    Create TF-IDF based description on n-grams
    :param comps: dictionary {component : [list of ids]}
    """
    logging.info('Computing most common n-grams')
    n_comps = len(set(comps.keys()))
    most_common = [None] * n_comps
    for idx, comp in comps.items():
        df_comp = df[df['id'].isin(comp)]
        most_common[idx] = get_most_common_ngrams(df_comp['title'], df_comp['abstract'])

    logging.info('Compute Augmented Term Frequency - Inverse Document Frequency')
    # The tf–idf is the product of two statistics, term frequency and inverse document frequency.
    # This provides greater weight to values that occur in fewer documents.
    idfs = {}
    kwd = {}
    for idx in range(n_comps):
        max_cnt = max(most_common[idx].values())
        idfs[idx] = {k: (0.5 + 0.5 * v / max_cnt) *  # augmented frequency to avoid document length bias
                     np.log(n_comps / sum([k in mcoc for mcoc in most_common])) \
                     for k, v in most_common[idx].items()}
        kwd[idx] = ','.join([f'{k}:{(max(most_common[idx][k], 1e-3)):.3f}'
                             for k, _v in list(sorted(idfs[idx].items(),
                                                      key=lambda kv: kv[1],
                                                      reverse=True))[:size]])
    return kwd


def get_word_cloud_data(df_kwd, c):
    """Parse TF-IDF based ngramms from text"""
    kwds = {}
    for pair in list(df_kwd[df_kwd['comp'] == c]['kwd'])[0].split(','):
        ngram, count = pair.split(':')
        for word in ngram.split(' '):
            kwds[word] = float(count) + kwds.get(word, 0)
    return kwds


def get_tfidf_words(df, comps, terms, size=5):
    corpus = []

    for comp, article_ids in comps.items():
        comp_corpus = ''
        for article_id in article_ids:
            sel = df[df['id'] == article_id]
            if len(sel) > 0:
                title = sel['title'].astype(str).values[0]
                abstract = sel['abstract'].astype(str).values[0]
                comp_corpus += f'{title} {abstract}'
            else:
                raise ValueError('Empty selection by id')
        corpus.append(comp_corpus)

    vectorizer = TfidfVectorizer(tokenizer=lambda text: tokenize(text, terms=terms), stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names()
    kwd = {}
    for i in comps.keys():
        # It might be faster to use np.argpartition instead of np.argsort
        # Sort indices by tfidf value
        ind = np.argsort(tfidf[i, :].toarray(), axis=1)

        # Take size indices with the largest tfidf
        kwd[i] = list(map(lambda idx: words[idx], ind[0, -size:]))

    return kwd


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


def cut_authors_list(authors, limit=10):
    before_separator = limit - 1
    separator = ',...,'
    author_list = authors.split(', ')
    if len(author_list) > limit:
        return ', '.join(author_list[:before_separator]) + separator + author_list[-1]
    return authors
