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
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Lock to support multithreading for NLTK
# See https://github.com/nltk/nltk/issues/1576
stopwords_lock = Lock()
wordnet_lock = Lock()

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

TOKENIZE_SPEC_SYMBOLS = re.compile(r'[^a-zA-Z0-9\- ]*')

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
    stopwords_lock.acquire()
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
    stopwords_lock.release()
    return result


def is_noun_or_adj(pos):
    return pos[:2] == 'NN' or pos == 'JJ'


def tokenize(text, query=None):
    text = text.lower()

    # Filter out search terms
    if query is not None:
        for term in query.split(' '):
            text = text.replace(term.lower(), '')

    tokenized = word_tokenize(re.sub(TOKENIZE_SPEC_SYMBOLS, '', text))

    stopwords_lock.acquire()
    stop_words = set(stopwords.words('english'))
    stopwords_lock.release()

    words_of_interest = [(word, pos) for word, pos in nltk.pos_tag(tokenized) if
                         word not in stop_words and is_noun_or_adj(pos)]

    lemmatizer = WordNetLemmatizer()
    lemmatized = filter(lambda t: len(t) >= 3,
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


def get_most_common_tokens(texts, fraction=0.1):
    """
    :param texts: list of texts for articles in a component
    :param fraction: fraction of most common tokens
    :return: dictionary {token: frequency}
    """
    counter = Counter()
    for text in texts:
        if isinstance(text, str):
            for token in set(tokenize(text)):
                counter[token] += 1
    most_common = {}
    tokens = len(counter)
    for token, cnt in counter.most_common(tokens if tokens <= 200 else int(tokens * fraction)):
        most_common[token] = cnt / tokens
    return most_common


def get_subtopic_descriptions(df, most_cited_papers_per_comp):
    """
    Create TF-IDF based description on tokens
    :param most_cited_papers_per_comp: dictionary {component: description}
    """
    log.info('Computing most common terms')
    n_comps = len(set(most_cited_papers_per_comp.keys()))
    most_common = [None] * n_comps
    for comp, comp_papers in most_cited_papers_per_comp.items():
        df_comp = df[df['id'].isin(comp_papers)]
        most_common[comp] = get_most_common_tokens(df_comp['title'] + ' ' + df_comp['abstract'])

    log.info('Compute Augmented Term Frequency - Inverse Document Frequency')
    # The tf–idf is the product of two statistics, term frequency and inverse document frequency.
    # This provides greater weight to values that occur in fewer documents.
    aug_tf_idf = {}
    kwds = {}
    for comp in range(n_comps):
        max_cnt = max(most_common[comp].values())
        # Augmented frequency to avoid document length bias
        aug_tf_idf[comp] = {k: (0.5 + 0.5 * v / max_cnt) * np.log(n_comps / sum([k in mcoc for mcoc in most_common]))
                            for k, v in most_common[comp].items()}
        kwds[comp] = [(k, most_common[comp][k]) for k, _v in
                      list(sorted(aug_tf_idf[comp].items(), key=lambda kv: kv[1], reverse=True))]
    return kwds


def get_topic_word_cloud_data(df_kwd, c):
    """Parse TF-IDF based tokens from text"""
    kwds = {}
    for pair in list(df_kwd[df_kwd['comp'] == c]['kwd'])[0].split(','):
        token, count = pair.split(':')
        for word in token.split(' '):
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
    log.info(f'Building corpus from {len(df)} articles')
    corpus = [f'{title} {abstract}'
              for title, abstract in zip(df['title'].values, df['abstract'].values)]
    log.info(f'Corpus size: {sys.getsizeof(corpus)} bytes')
    return corpus


def vectorize(corpus, terms=None, n_words=1000):
    log.info(f'Counting word usage in the corpus, using only {n_words} most frequent words')
    vectorizer = CountVectorizer(tokenizer=lambda t: tokenize(t, terms), max_features=n_words)
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
    ''' Preprocess searh string for Neo4j full text lookup '''
    if len(query) == 0:
        return None
    terms_str = re.sub('[^0-9a-zA-Z"\\-\\.+, ]', '', query.strip())  # Remove unknown symbols
    if len(terms_str) == 0:
        raise Exception(f'Illegal character(s), only English letters, numbers, '
                        f'and +- signs are supported')
    if len(query.split(' ')) < min_search_words:
        raise Exception(f'Please use more specific query with >= {min_search_words} words')
    # Looking for complete phrase
    if re.match('^"[^"]+"$', terms_str):
        return '\'"' + re.sub('"', '', terms_str) + '"\''
    elif re.match('^[^"]+$', terms_str):
        return '"' + ' AND '.join([f"'{w}'" for w in terms_str.split(' ')]) + '"'
    raise Exception(f'Illegal search string, please use search terms or '
                    f'all the query wrapped in "" for phrasal search')


def preprocess_doi(line):
    # Remove doi.org prefix if full URL was pasted, then strip unnecessary slashes
    (_, _, doi) = line.partition('doi.org')
    return doi.strip('/')
