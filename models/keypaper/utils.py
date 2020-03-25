import binascii
import html
import itertools
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from threading import Lock

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
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace('-', ' ')
    text = re.sub('[^a-zA-Z0-9 ]*', '', text)
    # Whitespaces normalization, see #215
    text = re.sub('[ ]{2,}', ' ', text.strip())
    return text


def tokenize(text, query=None, min_token_length=3):
    if text is None:
        return []
    text = preprocess_text(text)

    # Filter out search query
    if query is not None:
        for term in preprocess_text(query).split(' '):
            text = text.replace(term.lower(), '')

    tokenized = word_tokenize(text)

    words_of_interest = [(word, pos) for word, pos in nltk.pos_tag(tokenized) if
                         word not in STOP_WORDS_SET and is_noun_or_adj(pos)]

    lemmatizer = WordNetLemmatizer()
    lemmatized = filter(lambda t: len(t) >= min_token_length,
                        [lemmatizer.lemmatize(w, pos=get_wordnet_pos(pos)) for w, pos in words_of_interest])

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


def get_topics_description(df, comps, query, n_words):
    if len(comps) == 1:
        most_frequent = get_frequent_tokens(df, query)
        return {0: list(sorted(most_frequent.items(), key=lambda kv: kv[1], reverse=True))[:n_words]}

    ngrams, tfidf = compute_tfidf(df, comps, query, n_words, n_gram=2)
    result = {}
    for comp in comps.keys():
        # Generate no keywords for '-1' component
        if comp == -1:
            result[comp] = ''
            continue

        # Sort indices by tfidf value
        # It might be faster to use np.argpartition instead of np.argsort
        ind = np.argsort(tfidf[comp, :].toarray(), axis=1)

        # Take size indices with the largest tfidf
        result[comp] = list(itertools.chain.from_iterable(
            [[(t, tfidf[comp, idx]) for t in ngrams[idx].split(' ')] for idx in ind[0, ::-1]])
        )
    return result


def get_tfidf_words(df, comps, query, size, n_words):
    tokens, tfidf = compute_tfidf(df, comps, query, n_words, other_comp=-1, ignore_other=True)
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
        kwd[comp] = [tokens[idx] for idx in ind[0, -size:]]
    return kwd


def compute_tfidf(df, comps, query, n_words, n_gram=1, other_comp=None, ignore_other=False):
    corpus = []
    for comp, article_ids in comps.items():
        # -1 is OTHER component, not meaningful
        if not (ignore_other and comp == other_comp):
            df_comp = df[df['id'].isin(article_ids)]
            corpus.append(' '.join([f'{t} {a}' for t, a in zip(df_comp['title'], df_comp['abstract'])]))
    vectorizer = CountVectorizer(min_df=0.01, max_df=0.8, ngram_range=(1, n_gram),
                                 max_features=n_words * len(comps),
                                 tokenizer=lambda t: tokenize(t, query))
    counts = vectorizer.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(counts)
    ngrams = vectorizer.get_feature_names()
    return ngrams, tfidf


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
    log.info(f'Building corpus from {len(df)} articles')
    corpus = [f'{title} {abstract}'
              for title, abstract in zip(df['title'].values, df['abstract'].values)]
    log.info(f'Corpus size: {sys.getsizeof(corpus)} bytes')
    return corpus


def vectorize(corpus, query=None, n_words=1000):
    log.info(f'Counting word usage in the corpus, using only {n_words} most frequent words')
    vectorizer = CountVectorizer(tokenizer=lambda t: tokenize(t, query), max_features=n_words)
    counts = vectorizer.fit_transform(corpus)
    log.info(f'Output shape: {counts.shape}')
    return counts, vectorizer


def lda_subtopics(counts, n_topics=10):
    log.info(f'Performing LDA subtopic analysis')
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    topics = lda.fit_transform(counts)

    log.info('Done')
    return topics, lda


def explain_lda_subtopics(lda, vectorizer, n_top_words=20):
    feature_names = vectorizer.get_feature_names()
    explanations = {}
    for i, topic in enumerate(lda.components_):
        explanations[i] = [(topic[i], feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]

    return explanations


def trim(string, max_length):
    return f'{string[:max_length]}...' if len(string) > max_length else string


def preprocess_search_query(query, min_search_words):
    """ Preprocess search string for Neo4j full text lookup """
    if ',' in query:
        qor = ''
        for p in query.split(','):
            if len(qor) > 0:
                qor += ' OR '
            pp = preprocess_search_query(p.strip(), min_search_words)
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
        stems = set([stemmer.stem(word) for word in words])
        if len(stems) + len(processed.split('-')) - 1 < min_search_words:
            raise Exception(f'Please use query with >= {min_search_words} different words. Query: {query}')
        return ' AND '.join([w if '-' not in w else f'"{w}"' for w in words])  # Dashed terms should be quoted
    raise Exception(f'Illegal search query, please use search terms or '
                    f'all the query wrapped in "" for phrasal search. Query: {query}')
