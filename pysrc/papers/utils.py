import binascii
import html
import logging
import re
import sys
from collections import Counter
from string import Template
from threading import Lock

import nltk
import numpy as np
import pandas as pd
from matplotlib import colors
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

nltk.download('averaged_perceptron_tagger')  # required for nltk.pos_tag
nltk.download('punkt')  # required for word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
STOP_WORDS_SET = set(stopwords.words('english'))

# Lock to support multithreading for NLTK
# See https://github.com/nltk/nltk/issues/1576
NLTK_LOCK = Lock()

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
SORT_MOST_RECENT = 'Most Recent'


log = logging.getLogger(__name__)


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
    NLTK_LOCK.acquire()
    if treebank_tag.startswith('J'):
        result = wordnet.ADJ
    elif treebank_tag.startswith('V'):
        result = wordnet.VERB
    elif treebank_tag.startswith('N'):
        result = wordnet.NOUN
    elif treebank_tag.startswith('R'):
        result = wordnet.ADV
    else:
        result = ''
    NLTK_LOCK.release()
    return result


def is_noun_or_adj(pos):
    return pos[:2] == 'NN' or pos == 'JJ'


def preprocess_text(text):
    text = text.lower()
    # Replace non-ascii with space
    text = re.sub('[^a-zA-Z0-9 ]+', ' ', text)
    # Whitespaces normalization, see #215
    text = re.sub('[ ]{2,}', ' ', text.strip())
    return text


def tokens_stems(text, query=None, min_token_length=3):
    # Filter out search query
    if query is not None:
        for term in preprocess_text(query).split(' '):
            text = text.replace(term, '')

    tokenized = word_tokenize(text)

    words_of_interest = [(word, pos) for word, pos in nltk.pos_tag(tokenized) if
                         word not in STOP_WORDS_SET and is_noun_or_adj(pos)]

    lemmatizer = WordNetLemmatizer()
    lemmatized = filter(lambda t: len(t) >= min_token_length,
                        [lemmatizer.lemmatize(w, pos=get_wordnet_pos(pos)) for w, pos in words_of_interest])

    stemmer = SnowballStemmer('english')
    return [(stemmer.stem(word), word) for word in lemmatized]


def tokenize(text, query=None, min_token_length=3):
    if text is None:
        return []
    text = preprocess_text(text)

    stemmed = tokens_stems(text, query, min_token_length)
    # Substitute each stem with the shortest similar word
    stems_mapping = {}
    for stem, word in stemmed:
        if stem in stems_mapping:
            if len(stems_mapping[stem]) > len(word):
                stems_mapping[stem] = word
        else:
            stems_mapping[stem] = word

    return [stems_mapping[stem] for stem, _ in stemmed]


def get_frequent_tokens(df, query, fraction=0.1, min_tokens=20):
    """
    Compute tokens weighted frequencies
    :param query: search query to exclude
    :param fraction: fraction of most common tokens
    :param min_tokens: minimal number of tokens to return
    :return: dictionary {token: frequency}
    """
    counter = Counter()
    for text in df['title'] + ' ' + df['abstract']:
        for token in tokenize(text, query):
            counter[token] += 1
    result = {}
    tokens = len(counter)
    for token, cnt in counter.most_common(max(min_tokens, int(tokens * fraction))):
        result[token] = cnt / tokens
    return result


def get_topic_word_cloud_data(df_kwd, comp):
    """Parse TF-IDF based tokens from text"""
    kwds = {}
    for pair in list(df_kwd[df_kwd['comp'] == comp]['kwd'])[0].split(','):
        token, value = pair.split(':')
        for word in token.split(' '):
            kwds[word] = float(value) + kwds.get(word, 0)
    return kwds


def get_topics_description(df, comps, corpus_ngrams, corpus_counts, query, n_words):
    if len(comps) == 1:
        most_frequent = get_frequent_tokens(df, query)
        return {0: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words]}

    tfidf = compute_comps_tfidf(df, comps, corpus_counts)
    result = {}
    for comp in comps.keys():
        # Generate no keywords for '-1' component
        if comp == -1:
            result[comp] = ''
            continue

        # Take size indices with the largest tfidf first
        counter = Counter()
        for i, n in enumerate(corpus_ngrams):
            for w in n.split(' '):
                counter[w] += tfidf[comp, i]
        result[comp] = counter.most_common(n_words)
    return result


def get_evolution_topics_description(df, comps, corpus_ngrams, corpus_counts, size):
    # -1 is Not Published Yet
    tfidf = compute_comps_tfidf(df, comps, corpus_counts, ignore_comp=-1)
    kwd = {}
    for comp in comps.keys():
        # Generate no keywords for '-1' component
        if comp == -1:
            kwd[comp] = ''
            continue

        # Sort indices by tfidf value
        # It might be faster to use np.argpartition instead of np.argsort
        ind = np.argsort(tfidf[comp, :].toarray(), axis=1)

        # Take tokens with the largest tfidf
        kwd[comp] = [corpus_ngrams[idx] for idx in ind[0, -size:]]
    return kwd


def compute_comps_tfidf(df, comps, corpus_counts, ignore_comp=None):
    log.debug(f'Creating corpus for comps {len(comps)}')
    comps_to_process = dict(enumerate([c for c in comps if c != ignore_comp]))
    comp_counts = np.zeros(shape=(len(comps_to_process), corpus_counts.shape[1]), dtype=np.short)
    for c, article_ids in comps.items():
        if c in comps_to_process:
            comp_counts[comps_to_process[c], :] = \
                np.sum(corpus_counts[np.flatnonzero(df['id'].isin(article_ids)), :], axis=0)
    return compute_tfidf(comp_counts)


def compute_tfidf(counts):
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(counts)
    log.debug(f'TFIDF shape {tfidf.shape}')
    return tfidf


def vectorize_corpus(df, max_features, n_gram):
    corpus = build_corpus(df)
    vectorizer = CountVectorizer(
        min_df=0.01,
        max_df=0.5 if len(df) > 1 else 1.0,  # For tests
        ngram_range=(1, n_gram),
        max_features=max_features,
        tokenizer=lambda t: tokenize(t)
    )
    log.debug('Vectorizing')
    counts = vectorizer.fit_transform(corpus)
    log.debug(f'Vectorized corpus size {counts.shape}')
    return vectorizer.get_feature_names(), counts


def cosine_similarity(x):
    """Modified version of sklearn.metrics.pairwise.cosine_similarity - no copying"""
    normalize(x, copy=False)
    return x @ x.T


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
    log.info(f'Building corpus from {len(df)} papers')
    corpus = [preprocess_text(f'{title} {abstract} {keywords} {mesh}')
              for title, abstract, mesh, keywords in
              zip(df['title'], df['abstract'], df['keywords'], df['mesh'])]
    log.info(f'Corpus size: {sys.getsizeof(corpus)} bytes')
    return corpus


def vectorize(corpus, query=None, n_words=1000):
    log.info(f'Counting word usage in the corpus, using only {n_words} most frequent words')
    vectorizer = CountVectorizer(tokenizer=lambda t: tokenize(t, query), max_features=n_words)
    counts = vectorizer.fit_transform(corpus)
    log.info(f'Output shape: {counts.shape}')
    return counts, vectorizer


def lda_topics(counts, n_topics=10):
    log.info('Performing LDA topic analysis')
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    topics = lda.fit_transform(counts)

    log.info('Done')
    return topics, lda


def explain_lda_topics(lda, vectorizer, n_top_words=20):
    feature_names = vectorizer.get_feature_names()
    explanations = {}
    for i, topic in enumerate(lda.components_):
        explanations[i] = [(topic[i], feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]

    return explanations


def trim(string, max_length):
    return f'{string[:max_length]}...' if len(string) > max_length else string


def preprocess_doi(line):
    """
    Removes doi.org prefix if full URL was pasted, then strips unnecessary slashes
    """
    (_, _, doi) = line.rpartition('doi.org')
    return doi.strip('/')


def preprocess_search_title(line):
    """
    Title processing similar to PubmedXMLParser - special characters removal
    """
    return line.strip('.[]')


def rgb2hex(color):
    if isinstance(color, str):
        match = re.match('rgb\\((\\d+), (\\d+), (\\d+)\\)', color)
        if match:
            r, g, b = match.group(1), match.group(2), match.group(3)
        else:
            r, g, b, _ = colors.to_rgba(color)
            r, g, b = r * 255, g * 255, b * 255
    else:
        r, g, b = color
    return "#{0:02x}{1:02x}{2:02x}".format(int(r), int(g), int(b))
