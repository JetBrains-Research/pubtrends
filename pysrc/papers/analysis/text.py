import concurrent
import concurrent.futures
import logging
import multiprocessing
from itertools import chain
from math import ceil
from threading import Lock

import nltk
import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from more_itertools import sliced
from nltk import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from pysrc.config import WORD2VEC_EMBEDDINGS_LENGTH, WORD2VEC_WINDOW, WORD2VEC_EPOCHS, EMBEDDINGS_CHUNK_SIZE, \
    EMBEDDINGS_SENTENCE_OVERLAP
from pysrc.services.embeddings_service import is_embeddings_db_available
from pysrc.services.embeddings_service import is_texts_embeddings_available, fetch_texts_embedding, \
    fetch_tokens_embeddings, load_embeddings_from_df

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


def stemmed_tokens(sentence, min_token_length=3):
    # Tokenize text
    tokens = [t.text.lower().strip() for t in sentence]
    # Filter by length
    tokens = [t for t in tokens if len(t) >= min_token_length]
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
    3. Matching stems to the shortest existing lemma in texts
    """
    logger.info(f'Building corpus from {len(df)} papers')
    logger.info(f'Processing stemming for all papers')
    papers_stemmed_sentences = []
    # NOTE: we split mesh and keywords by commas into separate sentences
    for i, (title, abstract) in enumerate(zip(df['title'], df['abstract'])):
        papers_stemmed_sentences.append([
            stemmed_tokens(s) for s in NLP(f'{title}. {abstract}').sents
        ])
        if i % 100 == 1:
            logger.debug(f'Processed {i} papers')
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
    logger.debug('Training word2vec model from sentences')
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
    if not test:
        if is_texts_embeddings_available():

            # Fetching text embeddings from database
            if is_embeddings_db_available():
                return fetch_embeddings_from_db(df)

            # Compute text embeddings from texts
            return embeddings_from_service(df)

        # Fallback to tokens embeddings
        tokens_embs = fetch_tokens_embeddings(corpus_tokens)
        if tokens_embs is not None:
            return _texts_embeddings(corpus_counts, tokens_embs), None

    # Use in-house word2vec on tokens
    tokens_embs = _train_word2vec(corpus, corpus_tokens, test=test)
    return _texts_embeddings(corpus_counts, tokens_embs), None


def embeddings_from_service(df):
    logger.debug("Fetching embeddings from embeddings service")
    data = [(pid, f'{title}. {abstract}')
            for pid, title, abstract in zip(df['id'], df['title'], df['abstract'])]
    logger.debug('Collecting chunks for embeddings')
    chunks, chunks_idx = collect_papers_chunks(
        (data, EMBEDDINGS_CHUNK_SIZE, EMBEDDINGS_SENTENCE_OVERLAP))
    logger.debug(f'Done collecting chunks for embeddings: {len(chunks)}')
    return fetch_texts_embedding(chunks), chunks_idx


def fetch_embeddings_from_db(df):
    logger.debug(f'Fetching embeddings from DB')
    db_embeddings, db_index = load_embeddings_from_df(df['id'])
    logger.debug(f'Fetched {len(db_index)} embeddings from DB')
    pids_in_db = set([p for p, _ in db_index])
    pids_not_in_db = [pid for pid in df['id'] if pid not in pids_in_db]
    if len(pids_not_in_db) == 0:
        logger.debug('All the pids found in embeddings DB')
        return db_embeddings, db_index
    logger.debug(f'Not all the pids found in embeddings DB: {len(pids_not_in_db)}')
    not_in_db_df = df[df['id'].isin(pids_not_in_db)]
    logger.debug('Collecting chunks for embeddings not in DB')
    not_in_db_embeddings, not_in_db_index = embeddings_from_service(not_in_db_df)
    all_embeddings = np.concatenate([db_embeddings, not_in_db_embeddings])
    all_index = db_index + not_in_db_index
    logger.debug(f'Concatenated embeddings: {all_embeddings.shape}, index len: {len(all_index)}')
    return all_embeddings, all_index


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
    for i, (pid, text) in enumerate(batch):
        for chunk_id, chunk in enumerate(get_chunks(text, max_tokens, overlap_sentences)):
            chunk_idx.append((pid, chunk_id))
            chunks.append(chunk)
        if i % 100 == 1:
            logger.debug(f'Processed {i} papers')
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
