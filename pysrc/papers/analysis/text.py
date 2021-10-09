import logging
import re
from threading import Lock

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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
    :param min_df: Ignore tokens with frequency lower than given threshold
    :param max_df: Ignore tokens with frequency higher than given threshold
    :return: Return vocabulary, term counts and Stems to token map.
    """
    papers_corpus, stems_tokens_map = build_stemmed_corpus(df)
    logger.debug(f'Vectorize corpus')
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df if len(df) > 1 else 1.0,  # For tests
        max_features=max_features,
        preprocessor=lambda t: t,
        tokenizer=lambda t: t
    )
    counts = vectorizer.fit_transform(papers_corpus)
    logger.debug(f'Vectorized corpus size {counts.shape}')
    tokens_counts = np.asarray(np.sum(counts, axis=0)).reshape(-1)
    tokens_freqs = tokens_counts / len(df)
    logger.debug(f'Tokens frequencies min={tokens_freqs.min()}, max={tokens_freqs.max()}, '
                 f'mean={tokens_freqs.mean()}, std={tokens_freqs.std()}')
    corpus_tokens = vectorizer.get_feature_names()
    return corpus_tokens, counts, stems_tokens_map


def get_frequent_tokens(df, stems_tokens_map, fraction=0.1, min_tokens=20):
    """
    Compute tokens weighted frequencies
    :param df: papers dataframe
    :param stems_tokens_map Mapping from stems to tokens
    :param fraction: fraction of most common tokens
    :param min_tokens: minimal number of tokens to return
    :return: dictionary {token: frequency}
    """
    counter = nltk.Counter()
    for title, abstract, mesh, keywords in zip(df['title'], df['abstract'], df['mesh'], df['keywords']):
        for token in tokenize(f'{title} {abstract} {mesh} {keywords}', stems_tokens_map):
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
    # Ignore stop words, take into accounts nouns and adjectives, fix plural forms
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos))
              for token, pos in nltk.pos_tag(tokenized)
              if len(token) > min_token_length and token not in STOP_WORDS_SET and is_noun_or_adj(pos)]

    # Apply stemming to reduce word length
    stemmer = SnowballStemmer('english')
    return [(stemmer.stem(token), token) for token in lemmas]


def preprocess_text(text):
    text = text.lower()
    # Replace non-ascii with space
    text = re.sub('[^a-zA-Z0-9 ]+', ' ', text)
    # Whitespaces normalization, see #215
    text = re.sub('[ ]{2,}', ' ', text.strip())
    return text


def build_stemmed_corpus(df):
    logger.info(f'Building corpus from {len(df)} papers')
    logger.info(f'Processing lemmas and stemming for all papers')
    papers_stems_and_tokens = [
        stemmed_tokens(preprocess_text(f'{title} {abstract} {mesh} {keywords}'))
        for title, abstract, mesh, keywords in zip(df['title'], df['abstract'], df['mesh'], df['keywords'])
    ]
    logger.info('Creating global shortest stemming to tokens map')
    stems_tokens_map = build_stems_to_tokens_map(flatten(papers_stems_and_tokens))
    logger.info('Creating stemmed corpus')
    return [[stems_tokens_map[s] for s, _ in stemmed] for stemmed in papers_stems_and_tokens], stems_tokens_map


def tokenize(text, stems_tokens_map=None, min_token_length=3):
    stems_and_tokens = stemmed_tokens(preprocess_text(text), min_token_length)
    if stems_tokens_map is None:
        return [t for _, t in stems_and_tokens]
    else:
        return [stems_tokens_map[s] for s, _ in stems_and_tokens if s in stems_tokens_map]


def build_stems_to_tokens_map(stems_and_tokens):
    """ Substitute each stem with the shortest similar word """
    stems_tokens_map = {}
    for stem, token in stems_and_tokens:
        if stem in stems_tokens_map:
            if len(stems_tokens_map[stem]) > len(token):
                stems_tokens_map[stem] = token
        else:
            stems_tokens_map[stem] = token
    return stems_tokens_map


def word2vec_tokens(df, corpus_tokens, stems_tokens_map, vector_size=32):
    logger.debug(f'Compute words embeddings with word2vec')
    corpus_tokens_set = set(corpus_tokens)
    logger.debug('Collecting sentences across dataset')
    sentences = []
    for _, row in df.iterrows():
        for field in ['title', 'abstract', 'mesh', 'keywords']:
            sentences.extend(
                flatten([stems_tokens_map[s] for s, _ in
                         stemmed_tokens(preprocess_text(sentence.strip()))
                         if s in stems_tokens_map and stems_tokens_map[s] in corpus_tokens_set
                         ] for sentence in row[field].split('.') if len(sentence.strip()) > 0))
    logger.debug('Training word2vec model')
    w2v = Word2Vec(sentences, vector_size=vector_size, min_count=0, workers=1, epochs=10, seed=42)
    logger.debug('Retrieve word embeddings, corresponding subjects and reorder according to corpus_terms')
    ids, embeddings = w2v.wv.index_to_key, w2v.wv.vectors
    indx = {t: i for i, t in enumerate(ids)}
    return np.array([
        embeddings[indx[t]] if t in indx else np.zeros(embeddings.shape[1])  # Process missing
        for t in corpus_tokens
    ])


def texts_embeddings(corpus_counts, tokens_w2v_embeddings):
    """
    Computes texts embeddings as weighted average of word2vec words embeddings.
    :param corpus_counts: Vectorized papers matrix
    :param tokens_w2v_embeddings: Tokens word2vec embeddings
    :return: numpy array [publications x embeddings]
    """
    logger.debug('Compute text embeddings as weighted average of word2vec tokens embeddings')
    texts_embeddings = np.array([
        np.mean([np.multiply(tokens_w2v_embeddings[t], corpus_counts[pid, t])
                 for t in range(corpus_counts.shape[1])], axis=0)
        for pid in range(corpus_counts.shape[0])
    ])
    logger.debug(f'Texts embeddings shape: {texts_embeddings.shape}')
    return texts_embeddings


def flatten(t):
    return [item for sublist in t for item in sublist]
