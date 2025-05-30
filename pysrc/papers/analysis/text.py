import logging
import nltk
import numpy as np
import re
import spacy
from gensim.models import Word2Vec
from itertools import chain
from nltk import word_tokenize, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from threading import Lock
import pandas as pd
from more_itertools import sliced
from math import ceil
import concurrent
import multiprocessing
import concurrent.futures

from pysrc.config import WORD2VEC_EMBEDDINGS_LENGTH, WORD2VEC_WINDOW, WORD2VEC_EPOCHS, PubtrendsConfig
from pysrc.services.embeddings_service import is_embeddings_service_ready, is_texts_embeddings_available, \
    fetch_texts_embedding, fetch_tokens_embeddings

NLP = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

# Ensure that modules are downloaded in advance
# nltk averaged_perceptron_tagger required for nltk.pos_tag
# nltk punkt required for word_tokenize
# nltk stopwords
# nltk wordnet
NLTK_STOP_WORDS_SET = set(stopwords.words('english'))

# Lock to support multithreading for NLTK
# See https://github.com/nltk/nltk/issues/1576
NLTK_LOCK = Lock()


def vectorize_corpus(df, max_features, min_df, max_df, test=False):
    """
    Create vectorization for papers in df.
    :param df: papers dataframe
    :param max_features: Maximum vocabulary size
    :param min_df: Ignore tokens with frequency lower than given threshold
    :param max_df: Ignore tokens with frequency higher than given threshold
    :param test:
    :return: Return list of list of sentences for each paper, tokens, and counts matrix
    """
    papers_sentences_corpus = build_stemmed_corpus(df)
    logger.debug(f'Vectorize corpus of {len(df)} papers')
    counts = None
    while counts is None:
        try:
            vectorizer = CountVectorizer(
                min_df=min_df,
                max_df=max_df if not test else 1.0,
                max_features=max_features,
                preprocessor=lambda t: t,
                tokenizer=lambda t: t
            )
            counts = vectorizer.fit_transform([list(chain(*sentences)) for sentences in papers_sentences_corpus])
        except:
            # Workaround for exception After pruning, no terms remain.
            logger.debug(f'Failed to build counts for vector for min_df={min_df}, max_df={max_df}, adjusting')
            min_df = max(0.0, min_df - 0.1)
            max_df = min(1.0, max_df + 0.1)
    logger.debug(f'Vectorized corpus size {counts.shape}')
    tokens_counts = np.asarray(np.sum(counts, axis=0)).reshape(-1)
    tokens_freqs = tokens_counts / len(df)
    logger.debug(f'Tokens frequencies min={tokens_freqs.min()}, max={tokens_freqs.max()}, '
                 f'mean={tokens_freqs.mean()}, std={tokens_freqs.std()}')
    corpus_tokens = vectorizer.get_feature_names_out().tolist()
    corpus_tokens_set = set(corpus_tokens)
    # Filter tokens left after vectorization
    filtered_corpus = [
        [[t for t in sentence if t in corpus_tokens_set] for sentence in paper_sentences]
        for paper_sentences in papers_sentences_corpus
    ]
    return filtered_corpus, corpus_tokens, counts


def get_frequent_tokens(tokens, fraction=0.1, min_tokens=20):
    """
    Compute tokens weighted frequencies
    :param tokens List of tokens
    :param fraction: fraction of most common tokens
    :param min_tokens: minimal number of tokens to return
    :return: dictionary {token: frequency}
    """
    counter = FreqDist(tokens)
    result = {}
    tokens = len(counter)
    for token, cnt in counter.most_common(max(min_tokens, int(tokens * fraction))):
        result[token] = cnt / tokens
    return result


# Convert pos_tag output to WordNetLemmatizer tags
try:
    NLTK_LOCK.acquire()
    NLTK_POS_TAG_TO_WORDNET = dict(JJ=wordnet.ADJ, NN=wordnet.NOUN, VB=wordnet.VERB, RB=wordnet.ADV)
finally:
    NLTK_LOCK.release()


def stemmed_tokens(text, min_token_length=4):
    text = text.lower()
    # Replace non-ascii or punctuation with space
    text = re.sub('[^a-zA-Z0-9-+.,:!?]+', ' ', text)
    # Whitespaces normalization, see #215
    text = re.sub('\s{2,}', ' ', text.strip())
    # Tokenize text
    tokens = word_tokenize(text)
    # Let tokens start with letters only
    tokens = [re.sub('^[^a-zA-Z]+', '', t) for t in tokens]
    # Ignore e-XX tokens
    tokens = [t for t in tokens if not re.match('e-?[0-9]+', t)]
    # Ignore stop words, take into accounts nouns and adjectives, fix plural forms
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token, pos=NLTK_POS_TAG_TO_WORDNET[pos[:2]])
              for token, pos in nltk.pos_tag(tokens)
              if len(token) >= min_token_length
              and token not in NLTK_STOP_WORDS_SET
              and pos[:2] in NLTK_POS_TAG_TO_WORDNET]

    # Apply stemming to reduce word length,
    # later shortest word will be used as actual word
    stemmer = SnowballStemmer('english')
    return [(stemmer.stem(token), token) for token in lemmas]


def build_stemmed_corpus(df):
    """ Tokenization is done in several steps
    1. Lemmatization:  Ignore stop words, take into accounts nouns and adjectives, fix plural forms
    2. Stemming: reducing words
    3. Matching stems to a shortest existing lemma in texts
    """
    logger.info(f'Building corpus from {len(df)} papers')
    logger.info(f'Processing stemming for all papers')
    papers_stemmed_sentences = []
    # NOTE: we split mesh and keywords by commas into separate sentences
    for i, (title, abstract, mesh, keywords) in enumerate(zip(df['title'],
                                                              df['abstract'],
                                                              df['mesh'].replace(',', '.'),
                                                              df['keywords'].replace(',', '.'))):
        if i % 100 == 1:
            logger.debug(f'Processed {i} papers')
        papers_stemmed_sentences.append([
            stemmed_tokens(sentence)
            for sentence in f'{title}.{abstract}.{mesh}.{keywords}'.split('.')
            if len(sentence.strip()) > 0
        ])
    logger.debug(f'Done processing stemming for {len(df)} papers')
    logger.info('Creating global shortest stem to word map')
    stems_tokens_map = _build_stems_to_tokens_map(chain(*chain(*papers_stemmed_sentences)))
    logger.info('Creating stemmed corpus')
    return [[[stems_tokens_map.get(s, s) for s, _ in stemmed] for stemmed in sentence]
            for sentence in papers_stemmed_sentences]


def _build_stems_to_tokens_map(stems_and_tokens):
    """ Build a map to substitute each stem with the shortest word if word is different """
    stems_tokens_map = {}
    for stem, token in stems_and_tokens:
        if stem != token:  # Ignore tokens similar to stems
            if stem in stems_tokens_map:
                if len(stems_tokens_map[stem]) > len(token):
                    stems_tokens_map[stem] = token
            else:
                stems_tokens_map[stem] = token
    return stems_tokens_map


def _train_word2vec(corpus, corpus_tokens, vector_size=WORD2VEC_EMBEDDINGS_LENGTH, test=False):
    logger.debug('Collecting sentences across dataset')
    sentences = list(filter(
        lambda l: test or len(l) >= WORD2VEC_WINDOW,  # Ignore short sentences, less than a window
        chain.from_iterable(corpus)))
    logger.debug(f'Total {len(sentences)} sentences')
    logger.debug('Training word2vec model')
    w2v = Word2Vec(
        sentences, vector_size=vector_size,
        window=WORD2VEC_WINDOW, min_count=0, workers=1, epochs=WORD2VEC_EPOCHS, seed=42
    )
    logger.debug('Retrieve word embeddings, corresponding subjects and reorder according to corpus_terms')
    ids, embeddings = w2v.wv.index_to_key, w2v.wv.vectors
    indx = {t: i for i, t in enumerate(ids)}
    return np.array([
        embeddings[indx[t]] if t in indx else np.zeros(embeddings.shape[1])  # Process missing embeddings
        for t in corpus_tokens
    ])


def embeddings(df, corpus, corpus_tokens, corpus_counts, test=False):
    if not test and is_embeddings_service_ready():
        # Start with text embeddings
        if is_texts_embeddings_available():
            chunks, chunks_idx = parallel_collect_chunks(
                df['id'],
                [f'{title}. {abstract}' for title, abstract in zip(df['title'], df['abstract'])],
                256, 1)
            return fetch_texts_embedding(chunks), chunks_idx

        # Fallback to tokens embeddings
        tokens_embs = fetch_tokens_embeddings(corpus_tokens)
        if tokens_embs is not None:
            return _texts_embeddings(corpus_counts, tokens_embs), None

    logger.debug('Use to in-house word2vec')
    tokens_embs = _train_word2vec(corpus, corpus_tokens, test=test)
    return _texts_embeddings(corpus_counts, tokens_embs), None

def chunks_to_text_embeddings(df, chunks_embeddings, chunks_idx):
    if chunks_idx is None:
        return chunks_embeddings
    text_embeddings = np.ndarray((len(df), chunks_embeddings.shape[1]))
    chunks_df = pd.DataFrame(chunks_idx, columns=['pid', 'cid']).groupby('pid').agg('count')
    ci = 0
    for i, pid in enumerate(df['id']):
        chunks = chunks_df.loc[pid, 'cid']
        text_embeddings[i] = np.mean(chunks_embeddings[ci:ci + chunks], axis=0)
        ci += chunks
    return text_embeddings


def _texts_embeddings(corpus_counts, tokens_embeddings):
    """
    Computes texts embeddings as TF-IDF weighted average of words embeddings.
    :param corpus_counts: Vectorized papers matrix
    :param tokens_embeddings: Tokens word2vec embeddings
    :return: numpy array [publications x embeddings]
    """
    logger.debug('Compute TF-IDF on tokens counts')
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(corpus_counts)
    logger.debug(f'TFIDF shape {tfidf.shape}')

    logger.debug('Compute text embeddings as TF-IDF weighted average of tokens embeddings')
    embeddings = np.array([
        np.mean((tokens_embeddings.T * tfidf[i, :].T).T, axis=0) for i in range(tfidf.shape[0])
    ])
    logger.debug(f'Texts embeddings shape: {embeddings.shape}')
    return embeddings


def get_chunks(text, max_tokens=128, overlap_sentences=1):
    """
    Split text into a list of overlapping chunks.

    Args:
        text (str): The text to split into chunks
        max_tokens (int): Maximum number of tokens per chunk
        overlap_sentences (int): Number of sentences to overlap between chunks

    Returns:
        list: List of text chunks
    """

    # Get all sentences
    sentences = list(NLP(text).sents)

    if not sentences:
        return [text]

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        # If adding this sentence exceeds max_tokens, create a new chunk
        if current_token_count + len(sentence) > max_tokens and current_chunk_sentences:
            # Join the sentences in the current chunk
            chunk_text = ' '.join([s.text for s in current_chunk_sentences])
            chunks.append(chunk_text)

            # Keep the overlapping sentences for the next chunk
            if overlap_sentences > 0:
                overlap_size = min(overlap_sentences, len(current_chunk_sentences))
                current_chunk_sentences = current_chunk_sentences[-overlap_size:]
                current_token_count = sum(len(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences = []
                current_token_count = 0

        # Add the current sentence to the chunk
        current_chunk_sentences.append(sentence)
        current_token_count += len(sentence)

    # Add the last chunk if there are any sentences left
    if current_chunk_sentences:
        chunk_text = ' '.join([s.text for s in current_chunk_sentences])
        chunks.append(chunk_text)
    return chunks


def collect_papers_chunks(args):
    batch, max_tokens, overlap_sentences = args
    chunks = []
    chunk_idx = []
    for pid, text in batch:
        for i, chunk in enumerate(get_chunks(text, max_tokens, overlap_sentences)):
            chunk_idx.append((pid, i))
            chunks.append(chunk)
    return chunks, chunk_idx

def parallel_collect_chunks(
        pids,
        texts,
        max_tokens,
        overlap_sentences=1,
        max_workers=multiprocessing.cpu_count()
):
    chunks = []
    chunk_idx = []

    parallel_batches = [(b, max_tokens, overlap_sentences)
                        for b in sliced(list(zip(pids, texts)), int(ceil(len(texts) / max_workers)))]

    # Process texts in parallel processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and wait for results
        results = list(executor.map(collect_papers_chunks, parallel_batches))

    # Combine results
    for text_chunks, text_chunk_idx in results:
        chunks.extend(text_chunks)
        chunk_idx.extend(text_chunk_idx)
    assert len(chunks) == len(chunk_idx)
    return chunks, chunk_idx
