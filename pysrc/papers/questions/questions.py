import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pysrc.config import MAX_AUTHOR_LENGTH, EMBEDDINGS_QUESTIONS_CHUNK_SIZE, EMBEDDINGS_QUESTIONS_SENTENCE_OVERLAP
from pysrc.papers.analysis.text import get_chunks
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.utils import trim, topics_palette_rgb, rgb2hex
from pysrc.services.embeddings_service import fetch_texts_embedding


def get_relevant_papers(
        search_query,
        question_text,
        data,
        questions_threshold,
        questions_top_n
):
    # Get chunks and embeddings from data
    chunks_embeddings = data.chunks_embeddings
    chunks_idx = data.chunks_idx

    # Compute embeddings for the question
    contextual_question_embedding = fetch_texts_embedding([search_query, question_text])
    if contextual_question_embedding is None:
        raise Exception('Failed to compute embeddings for question')
    # Use average of search query and question for similarities
    contextual_question_embedding = np.mean(contextual_question_embedding, axis=0).reshape(1, -1)

    # Find the closest chunks
    similarities = cosine_similarity(contextual_question_embedding, chunks_embeddings)[0]

    # Filter indices with similarity >= questions_threshold
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= questions_threshold]

    # If no chunks meet the threshold, return empty list
    if not filtered_indices:
        return []

    # Sort filtered ones
    top_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)

    # Get the corresponding paper IDs
    paper_ids = []
    answer_chunks = []
    for idx in top_indices:
        pid, chunk_id = chunks_idx[idx]
        if pid not in paper_ids:
            paper_ids.append(pid)
            sel = data.df[data.df['id'] == pid]
            for chunk in get_chunks(f'{sel["title"].values[0]}. {sel["abstract"].values[0]}',
                                    EMBEDDINGS_QUESTIONS_CHUNK_SIZE,
                                    EMBEDDINGS_QUESTIONS_SENTENCE_OVERLAP):
                answer_chunks.append((pid, chunk))
        if len(paper_ids) == questions_top_n:
            break

    answer_chunk_embeddings = fetch_texts_embedding([t for _, t in answer_chunks])
    if answer_chunk_embeddings is None:
        raise Exception('Failed to compute embeddings for answer papers')

    question_embedding = fetch_texts_embedding([question_text])
    if question_embedding is None:
        raise Exception('Failed to compute embeddings for question')


    # Find the closest chunks to the question itself without context
    answer_similarities = cosine_similarity(question_embedding, answer_chunk_embeddings)[0]
    # Sort filtered ones
    answers_top_indices = sorted(list(range(len(answer_chunks))), key=lambda i: answer_similarities[i], reverse=True)
    result_chunks = []
    result_pids = []
    for idx in answers_top_indices:
        pid, chunk_text = answer_chunks[idx]
        if pid not in result_pids:
            result_pids.append(pid)
            result_chunks.append(chunk_text)

    # Filter and reorder by similarity
    df = data.df[data.df['id'].isin(result_pids)].copy()
    df = df.set_index('id').loc[result_pids].reset_index()
    df['chunk'] = result_chunks

    comp_colors = topics_palette_rgb(data.df)

    # Get paper details
    url_prefix = Loaders.get_url_prefix(data.source)
    papers = []
    for _, row in df.iterrows():
        pid, title, chunk, comp, authors, journal, year = \
            row['id'], row['title'], row['chunk'], row['comp'], row['authors'], row['journal'], row['year']
        # Don't trim or cut anything here, because this information can be exported
        papers.append(dict(pid=pid, title=title, chunk=chunk,
                           topic=comp + 1, color = rgb2hex(str(comp_colors[comp])),
                           authors=trim(authors, MAX_AUTHOR_LENGTH), journal=journal, year=year,
                           url=url_prefix + pid if url_prefix else None))
    return papers
