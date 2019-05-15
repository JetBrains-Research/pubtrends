import logging
import nltk
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

PUBMED_ARTICLE_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='


def get_ngrams(string):
    "1/2/3-gramms computation for string"
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(re.sub('[^a-zA-Z0-9\- ]*', '', string.lower()))
    stop_words = set(stopwords.words('english'))
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos) and word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(n) for n in nouns]
    ngrams = list(tokens)
    for t1, t2 in zip(tokens[:-1], tokens[1:]):
        ngrams.append(t1 + ' ' + t2)
    for t1, t2, t3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
        ngrams.append(t1 + ' ' + t2 + ' ' + t3)
    return ngrams


def get_most_common_ngrams(titles, number=50):
    ngrams = []
    for title in titles:
        ngrams.extend(get_ngrams(title))
    most_common = {}
    for ngram, cnt in Counter(ngrams).most_common(number):
        most_common[ngram] = cnt / len(ngrams)
    return most_common


def get_subtopic_descriptions(df, size=5):
    logging.info('Compute most common n-gramms')
    n_comps = df['comp'].nunique()
    most_common = [None] * n_comps
    for c in range(n_comps):
        most_common[c] = dict(get_most_common_ngrams(df[df['comp'] == c]['title'].values))

    logging.info('Compute Augmented Term Frequency - Inverse Document Frequency')
    # The tf–idf is the product of two statistics, term frequency and inverse document frequency.
    # This provides greater weight to values that occur in fewer documents.
    idfs = {}
    kwd = {}
    for c in range(n_comps):
        max_cnt = max(most_common[c].values())
        idfs[c] = {k: (0.5 + 0.5 * v / max_cnt) *  # augmented frequency to avoid document length bias
                   np.log(n_comps / sum([k in mcoc for mcoc in most_common])) \
                   for k,v in most_common[c].items()}
        kwd[c] = ', '.join([f'{k} ({most_common[c][k]:.2f})' \
                           for k,_v in list(sorted(idfs[c].items(),
                                                   key=lambda kv: kv[1],
                                                   reverse=True))[:size]])
    return kwd