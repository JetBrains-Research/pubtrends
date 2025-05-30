import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pysrc.config import MAX_AUTHOR_LENGTH
from pysrc.papers.analysis.embeddings_service import fetch_texts_embedding
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.utils import trim, topics_palette_rgb, rgb2hex


def get_relevant_papers(question_text, data, questions_threshold, questions_top_n):
    # Get chunks and embeddings from data
    chunks_embeddings = data.chunks_embeddings
    chunks_idx = data.chunks_idx

    # Compute embeddings for the question
    question_embedding = fetch_texts_embedding([question_text])
    if question_embedding is None:
        raise Exception('Failed to compute embeddings for question')

    # Find the closest chunks
    similarities = cosine_similarity(question_embedding, chunks_embeddings)[0]

    # Filter indices with similarity >= questions_threshold
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= questions_threshold]

    # If no chunks meet the threshold, return empty list
    if not filtered_indices:
        return []

    # Sort filtered ones
    top_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)

    # Get the corresponding paper IDs
    paper_ids = set()
    for idx in top_indices:
        paper_ids.add(chunks_idx[idx][0])  # The first element is paper ID
        if len(paper_ids) == questions_top_n:
            break
    df = data.df[data.df['id'].isin(paper_ids)]

    comp_colors = topics_palette_rgb(data.df)

    # Get paper details
    url_prefix = Loaders.get_url_prefix(data.source)
    papers = []
    for _, row in df.iterrows():
        pid, title, comp, authors, journal, year = \
            row['id'], row['title'], row['comp'], row['authors'], row['journal'], row['year']
        # Don't trim or cut anything here, because this information can be exported
        papers.append(dict(pid=pid, title=title,
                           topic=comp + 1, color = rgb2hex(str(comp_colors[comp])),
                           authors=trim(authors, MAX_AUTHOR_LENGTH), journal=journal, year=year,
                           url=url_prefix + pid if url_prefix else None))
    return papers
