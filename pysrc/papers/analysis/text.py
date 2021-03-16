import logging
import re
from threading import Lock

import nltk
import numpy as np
from nltk import word_tokenize, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Ensure that modules are downloaded in advance
# nltk averaged_perceptron_tagger required for nltk.pos_tag
# nltk punkt  required for word_tokenize
# nltk stopwords
# nltk wordnet
STOP_WORDS_SET = set(stopwords.words('english'))

# Lock to support multithreading for NLTK
# See https://github.com/nltk/nltk/issues/1576
NLTK_LOCK = Lock()


def vectorize_corpus(df, max_features, min_df, max_df):
    """
    Create vectorization for papers in df.
    :param df: papers dataframe
    :param max_features: Maximum vocabulary size
    :param min_df: Ignore terms with frequency lower than given threshold
    :param max_df: Ignore terms with frequency higher than given threshold
    :return: Return vocabulary and term counts.
    """
    corpus = build_corpus(df)
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df if len(df) > 1 else 1.0,  # For tests
        max_features=max_features,
        tokenizer=lambda t: tokenize(t)
    )
    logger.debug(f'Vectorization min_df={min_df} max_df={max_df} max_features={max_features}')
    counts = vectorizer.fit_transform(corpus)
    logger.debug(f'Vectorized corpus size {counts.shape}')
    terms_counts = np.asarray(np.sum(counts, axis=0)).reshape(-1)
    terms_freqs = terms_counts / len(df)
    logger.debug(f'Terms frequencies min={terms_freqs.min()}, max={terms_freqs.max()}, '
                 f'mean={terms_freqs.mean()}, std={terms_freqs.std()}')
    return vectorizer.get_feature_names(), counts


def analyze_texts_similarity(df, corpus_vectors, min_threshold):
    """
    Computes texts similarities based on cosine distance between texts vectors
    :param df: Papers dataframe
    :param corpus_vectors: Vectorized papers matrix
    :param min_threshold: Min similarity threshold to add it to result
    :return: List[(similarity, index)] - list of similar papers
    """
    cos_similarities = cosine_similarity(corpus_vectors)
    similarities = []
    # Adding text citations
    for i, pid1 in enumerate(df['id']):
        paper_similarities = []
        similarities.append(paper_similarities)
        for j in range(i + 1, len(df)):
            similarity = cos_similarities[i, j]
            if np.isfinite(similarity) and similarity >= min_threshold:
                paper_similarities.append((similarity, j))
    return similarities


def compute_comps_tfidf(df, comps, corpus_counts, ignore_comp=None):
    """
    Compute TFIDF for components based on average counts
    :param df: Papers dataframe
    :param comps: Dict of component to all papers
    :param corpus_counts: Vectorization for all papers
    :param ignore_comp: None or number of component to ignore
    :return: TFIDF matrix of size (components x new_vocabulary_size) and new_vocabulary
    """
    logger.debug('Compute average terms counts per components')
    # Since some of the components may be skipped, use this dict for continuous indexes
    comp_idx = dict(enumerate([c for c in comps if c != ignore_comp]))
    terms_freqs_per_comp = np.zeros(shape=(len(comp_idx), corpus_counts.shape[1]), dtype=np.short)
    for comp, comp_pids in comps.items():
        if comp in comp_idx:  # Not ignored
            terms_freqs_per_comp[comp_idx[comp], :] = \
                np.sum(corpus_counts[np.flatnonzero(df['id'].isin(comp_pids)), :], axis=0) / len(comp_pids)

    return compute_tfidf(terms_freqs_per_comp)


def get_topic_word_cloud_data(kwd_df, comp):
    """
    Parse TF-IDF based tokens from text for given comp
    :param kwd_df:
    :param comp:
    :return:
    """
    kwds = {}
    for pair in list(kwd_df[kwd_df['comp'] == comp]['kwd'])[0].split(','):
        if pair != '':  # Correctly process empty kwds encoding
            token, value = pair.split(':')
            for word in token.split(' '):
                kwds[word] = float(value) + kwds.get(word, 0)
    return kwds


def get_frequent_tokens(df, query, fraction=0.1, min_tokens=20):
    """
    Compute tokens weighted frequencies
    :param df: papers dataframe
    :param query: search query to exclude
    :param fraction: fraction of most common tokens
    :param min_tokens: minimal number of tokens to return
    :return: dictionary {token: frequency}
    """
    counter = nltk.Counter()
    for title, abstract, mesh, keywords in zip(df['title'], df['abstract'], df['keywords'], df['mesh']):
        for token in tokenize(f'{title} {abstract} {keywords} {mesh}', query):
            counter[token] += 1
    result = {}
    tokens = len(counter)
    for token, cnt in counter.most_common(max(min_tokens, int(tokens * fraction))):
        result[token] = cnt / tokens
    return result


def is_noun_or_adj(pos):
    return pos[:2] == 'NN' or pos == 'JJ'


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


def preprocess_text(text):
    text = text.lower()
    # Replace non-ascii with space
    text = re.sub('[^a-zA-Z0-9 ]+', ' ', text)
    # Whitespaces normalization, see #215
    text = re.sub('[ ]{2,}', ' ', text.strip())
    return text


def build_corpus(df):
    logger.info(f'Building corpus from {len(df)} papers')
    return [preprocess_text(f'{title} {abstract} {keywords} {mesh}')
            for title, abstract, mesh, keywords in
            zip(df['title'], df['abstract'], df['keywords'], df['mesh'])]


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


def compute_tfidf(counts):
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(counts)
    logger.debug(f'TFIDF shape {tfidf.shape}')
    return tfidf
