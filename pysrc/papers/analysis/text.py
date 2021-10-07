import logging
import re
from queue import PriorityQueue
from threading import Lock

import nltk
import numpy as np
from nltk import word_tokenize, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Ensure that modules are downloaded in advance
# nltk averaged_perceptron_tagger required for nltk.pos_tag
# nltk punkt required for word_tokenize
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
    :return: Return vocabulary, term counts and Stems to token map.
    """
    corpus, stems_map = build_corpus(df)
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df if len(df) > 1 else 1.0,  # For tests
        max_features=max_features,
        tokenizer=lambda t: tokenize(t, stems_map)
    )
    logger.debug(f'Vectorization min_df={min_df} max_df={max_df} max_features={max_features}')
    counts = vectorizer.fit_transform(corpus)
    logger.debug(f'Vectorized corpus size {counts.shape}')
    terms_counts = np.asarray(np.sum(counts, axis=0)).reshape(-1)
    terms_freqs = terms_counts / len(df)
    logger.debug(f'Terms frequencies min={terms_freqs.min()}, max={terms_freqs.max()}, '
                 f'mean={terms_freqs.mean()}, std={terms_freqs.std()}')
    return vectorizer.get_feature_names(), counts, stems_map


def analyze_texts_similarity(df, corpus_vectors, min_threshold, max_similar):
    """
    Computes texts similarities based on cosine distance between texts vectors
    :param df: Papers dataframe
    :param corpus_vectors: Vectorized papers matrix
    :param min_threshold: Min similarity threshold to add it to result
    :param max_similar: Max similar papers
    :return: PriorityQueue[(similarity, index)] - queue of similar papers
    """
    cos_similarities = cosine_similarity(corpus_vectors)
    similarity_queues = [PriorityQueue(maxsize=max_similar) for _ in range(len(df))]
    # Adding text citations
    for i, pid1 in enumerate(df['id']):
        queue_i = similarity_queues[i]
        for j in range(i + 1, len(df)):
            similarity = cos_similarities[i, j]
            if np.isfinite(similarity) and similarity >= min_threshold:
                if queue_i.full():
                    queue_i.get()  # Removes the element with lowest similarity
                queue_i.put((similarity, j))
    return similarity_queues


def get_frequent_tokens(df, stems_map, fraction=0.1, min_tokens=20):
    """
    Compute tokens weighted frequencies
    :param df: papers dataframe
    :param stems_map Mapping from stems to words
    :param fraction: fraction of most common tokens
    :param min_tokens: minimal number of tokens to return
    :return: dictionary {token: frequency}
    """
    counter = nltk.Counter()
    for title, abstract, mesh, keywords in zip(df['title'], df['abstract'], df['keywords'], df['mesh']):
        for token in tokenize(f'{title} {abstract} {keywords} {mesh}', stems_map):
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
    result = ''
    try:
        NLTK_LOCK.acquire()
        if treebank_tag.startswith('J'):
            result = wordnet.ADJ
        elif treebank_tag.startswith('V'):
            result = wordnet.VERB
        elif treebank_tag.startswith('N'):
            result = wordnet.NOUN
        elif treebank_tag.startswith('R'):
            result = wordnet.ADV
    finally:
        NLTK_LOCK.release()
        return result


def stemmed_tokens(text, min_token_length=3):
    tokenized = word_tokenize(text)
    interesting_tokens = [(token, pos) for token, pos in nltk.pos_tag(tokenized) if
                          token not in STOP_WORDS_SET and is_noun_or_adj(pos)]
    # Apply lemmatizer to fix plurals, etc
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = filter(lambda t: len(t) >= min_token_length,
                               [lemmatizer.lemmatize(w, pos=get_wordnet_pos(pos)) for w, pos in interesting_tokens])

    # Apply stemming to reduce word length
    stemmer = SnowballStemmer('english')
    return [(stemmer.stem(token), token) for token in lemmatized_tokens]


def preprocess_text(text):
    text = text.lower()
    # Replace non-ascii with space
    text = re.sub('[^a-zA-Z0-9 ]+', ' ', text)
    # Whitespaces normalization, see #215
    text = re.sub('[ ]{2,}', ' ', text.strip())
    return text


def build_corpus(df):
    logger.info(f'Building corpus from {len(df)} papers')
    logger.info(f'Processing lemmas and stemming for all papers')
    df_stems_and_tokens = [
        stemmed_tokens(preprocess_text(f'{title} {abstract} {keywords} {mesh}'), min_token_length=3)
        for title, abstract, mesh, keywords in
        zip(df['title'], df['abstract'], df['keywords'], df['mesh'])
    ]
    logger.info('Creating global shortest stemming to tokens map')
    stems_map = build_stemming_map(flatten(df_stems_and_tokens))
    logger.info('Creating stemmed corpus')
    return [' '.join([stems_map[s] for s, _ in stemmed]) for stemmed in df_stems_and_tokens], stems_map


def flatten(t):
    return [item for sublist in t for item in sublist]


def tokenize(text, stems_map=None, min_token_length=3):
    stems_and_tokens = stemmed_tokens(preprocess_text(text), min_token_length)
    if stems_map is None:
        return [t for _, t in stems_and_tokens]
    else:
        return [stems_map[s] for s, _ in stems_and_tokens if s in stems_map]


def build_stemming_map(stems_and_tokens):
    """ Substitute each stem with the shortest similar word """
    stems_map = {}
    for stem, token in stems_and_tokens:
        if stem in stems_map:
            if len(stems_map[stem]) > len(token):
                stems_map[stem] = token
        else:
            stems_map[stem] = token
    return stems_map
