import binascii
import html
import logging
import re
import sys
from collections import Counter
from string import Template

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

LOCAL_BASE_URL = Template('/paper?source=$source&id=')
PUBMED_ARTICLE_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='
SEMANTIC_SCHOLAR_BASE_URL = 'https://www.semanticscholar.org/paper/'

# IMPORTANT
# KeyPaperAnalyzer.launch() performs "zoom - 1" expand operations if id_list is given.
# This allows to perform zoom out in case of regular zoom out action and double in case of single paper analysis.
ZOOM_IN = 0
ZOOM_OUT = 1
PAPER_ANALYSIS = 2

ZOOM_IN_TITLE = 'detailed'
ZOOM_OUT_TITLE = 'expanded'
PAPER_ANALYSIS_TITLE = 'paper-analysis'

SORT_MOST_CITED = 'Most Cited'
SORT_MOST_RELEVANT = 'Most Relevant'
SORT_MOST_RECENT = 'Mose Recent'


def zoom_name(zoom):
    if int(zoom) == ZOOM_IN:
        return ZOOM_IN_TITLE
    elif int(zoom) == ZOOM_OUT:
        return ZOOM_OUT_TITLE
    elif int(zoom) == PAPER_ANALYSIS:
        return PAPER_ANALYSIS_TITLE
    raise ValueError(f'Illegal zoom key value: {zoom}')


def get_wordnet_pos(treebank_tag):
    """Convert pos_tag output to WordNetLemmatizer tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def is_noun_or_adj(pos):
    return pos[:2] == 'NN' or pos == 'JJ'


def tokenize(text, query=None):
    special_symbols_regex = re.compile(r'[^a-zA-Z0-9\- ]*')
    text = text.lower()

    # Filter out search terms
    if query is not None:
        for term in query.split(' '):
            text = text.replace(term.lower(), '')

    tokenized = word_tokenize(re.sub(special_symbols_regex, '', text))
    stop_words = set(stopwords.words('english'))
    words_of_interest = [(word, pos) for word, pos in nltk.pos_tag(tokenized) if
                         word not in stop_words and is_noun_or_adj(pos)]

    lemmatizer = WordNetLemmatizer()
    lemmatized = list(filter(lambda t: len(t) >= 3, [lemmatizer.lemmatize(w, pos=get_wordnet_pos(pos))
                                                     for w, pos in words_of_interest]))

    stemmer = SnowballStemmer('english')
    stemmed = [(stemmer.stem(word), word) for word in lemmatized]

    # Substitute each stem with the shortest similar word
    stems_mapping = {}
    for stem, word in stemmed:
        if stem in stems_mapping:
            if len(stems_mapping[stem]) > len(word):
                stems_mapping[stem] = word
        else:
            stems_mapping[stem] = word

    return [stems_mapping[stem] for stem, _ in stemmed]


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
        # augmented frequency to avoid document length bias
        idfs[idx] = {k: (0.5 + 0.5 * v / max_cnt) *
                     np.log(n_comps / sum([k in mcoc for mcoc in most_common]))
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


def get_tfidf_words(df, comps, query, size=5):
    corpus = []

    for comp, article_ids in comps.items():
        # Generate descriptions only for meaningful components, avoid -1
        if comp >= 0:
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

    vectorizer = TfidfVectorizer(tokenizer=lambda text: tokenize(text, query=query), stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names()
    kwd = {}
    for i in comps.keys():
        # Generate no keywords for '-1' component
        if i < 0:
            kwd[i] = ''
            continue

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
    # handle empty string and float('nan') cases
    if not authors or pd.isnull(authors):
        return "No authors listed"

    before_separator = limit - 1
    separator = ',...,'
    author_list = authors.split(', ')
    if len(author_list) > limit:
        return ', '.join(author_list[:before_separator]) + separator + author_list[-1]
    return authors


def extract_authors(authors_list):
    if not authors_list:
        return ''

    return ', '.join(filter(None, map(lambda authors: html.unescape(authors['name']), authors_list)))


def crc32(hex_string):
    n = binascii.crc32(bytes.fromhex(hex_string))
    return to_32_bit_int(n)


def to_32_bit_int(n):
    if n >= (1 << 31):
        return -(1 << 32) + n
    return n


def build_corpus(df):
    logging.info(f'Building corpus from {len(df)} articles')
    corpus = [f'{title} {abstract}'
              for title, abstract in zip(df['title'].values, df['abstract'].values)]
    logging.info(f'Corpus size: {sys.getsizeof(corpus)} bytes')
    return corpus


def vectorize(corpus, terms=None, n_words=1000):
    logging.info(f'Counting word usage in the corpus, using only {n_words} most frequent words')
    vectorizer = CountVectorizer(tokenizer=lambda t: tokenize(t, terms), max_features=n_words)
    counts = vectorizer.fit_transform(corpus)
    logging.info(f'Output shape: {counts.shape}')
    return counts, vectorizer


def lda_subtopics(counts, n_topics=10):
    logging.info(f'Performing LDA subtopic analysis')
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    topics = lda.fit_transform(counts)

    logging.info('Done')
    return topics, lda


def explain_lda_subtopics(lda, vectorizer, n_top_words=20):
    feature_names = vectorizer.get_feature_names()
    explanations = {}
    for i, topic in enumerate(lda.components_):
        explanations[i] = [(topic[i], feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]

    return explanations
