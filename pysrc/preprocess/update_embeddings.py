from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from more_itertools import sliced
from tqdm.auto import tqdm

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector
from pysrc.preprocess.embeddings.work_manager import WorkManager
from pysrc.papers.analysis.text import paper_text
from pysrc.config import PubtrendsConfig

if TYPE_CHECKING:
    from pysrc.faiss.faiss_connector import FaissConnector
    from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 2_000
FAISS_CHUNK_SIZE = 10_000
# Number of papers between Faiss index saves, each save writes the whole index of several GB
FAISS_SAVE_EVERY = 1_000_000


def update_year(
        df: pd.DataFrame,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
):
    wm = WorkManager(
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=None,  # Faiss index is updated from embeddings DB by update_faiss
    )
    print('Storing embeddings into DB')
    for index_slice in tqdm(list(sliced(range(len(df)), CHUNK_SIZE))):
        print(f'\rProcessing chunks {index_slice[0]}-{index_slice[-1]}          ', end='')
        chunk_df = df.iloc[index_slice]
        texts = [paper_text(title, abstract) for title, abstract in zip(chunk_df['title'], chunk_df['abstract'])]
        wm.process_compute_and_store_embeddings_work(
            list(chunk_df['pmid']), texts, embeddings_model_connector.device == 'cpu'
        )
    wm.finish_compute_and_store_embeddings_work(embeddings_model_connector.device == 'cpu')
    print('\rDone                           ')


def train_faiss(embeddings_db_connector: EmbeddingsDBConnector, faiss_connector: FaissConnector):
    """Returns False if there are not enough embeddings in DB to train the index yet"""
    min_size, size = faiss_connector.training_size()
    print(f'Sampling {size} embeddings for Faiss index training')
    embeddings = embeddings_db_connector.sample_embeddings(size)
    if len(embeddings) < min_size:
        print(f'Not enough embeddings to train Faiss index: {len(embeddings)} < {min_size}')
        return False
    print(f'Training Faiss index on {len(embeddings)} embeddings')
    faiss_connector.train(embeddings)
    return True


def update_faiss_index(
        df: pd.DataFrame,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector,
        publications_db_connector: PublicationsDBConnector | None = None,
        embeddings_model_connector: EmbeddingsModelConnector | None = None,
):
    print('Storing embeddings into Faiss')
    wm = WorkManager(
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector,
        publications_db_connector=publications_db_connector,
    )
    for index_slice in tqdm(list(sliced(range(len(df)), FAISS_CHUNK_SIZE))):
        print(f'\rProcessing chunks {index_slice[0]}-{index_slice[-1]}          ', end='')
        chunk_df = df.iloc[index_slice]
        wm.process_load_and_store_embeddings_to_faiss_work(
            list(chunk_df['pmid']),
        )
    wm.finish_load_and_store_embeddings_to_faiss_work()
    print('\rDone                           ')


def update_embeddings(
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        max_year,
        min_year,
        test_limit_per_year=None
):
    """Computes missing embeddings and stores them into embeddings DB, Faiss index is updated by update_faiss"""
    for year in range(max_year, min_year - 1, -1):
        print(f'Processing year {year}')
        df = publications_db_connector.load_publications_year(year)
        # Papers without any text can't be embedded
        df = df[[paper_text(t, a) != '' for t, a in zip(df['title'], df['abstract'])]]
        pids_to_process = set(embeddings_db_connector.collect_ids_without_embeddings(df['pmid']))
        print(f'To process {len(pids_to_process)}')
        df = df[df['pmid'].isin(pids_to_process)]

        if test_limit_per_year is not None:
            print(f'Limit to {test_limit_per_year}')
            df = df.head(test_limit_per_year)

        update_year(
            df,
            embeddings_model_connector=embeddings_model_connector,
            embeddings_db_connector=embeddings_db_connector,
        )


def update_faiss(
        publications_db_connector: PublicationsDBConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector,
        max_year,
        min_year,
        test_limit_per_year=None,
        embeddings_model_connector: EmbeddingsModelConnector | None = None,
):
    """Adds embeddings from embeddings DB missing in Faiss index, trains and rebuilds untrained index"""
    faiss_index, pids_idx = faiss_connector.create_or_load_faiss()
    if faiss_index.ntotal != len(pids_idx):
        raise RuntimeError(
            f'Index/FAISS size mismatch: {len(pids_idx)} pids vs {faiss_index.ntotal} vectors; '
            'refusing to update an inconsistent index'
        )
    if not faiss_index.is_trained:
        print('Untrained Faiss index, rebuilding from embeddings DB')
        if not train_faiss(embeddings_db_connector, faiss_connector):
            raise RuntimeError('Not enough embeddings in DB to train Faiss index')
        # Marker makes an interrupted rebuild resume, as intermediate saves store a trained index
        open(faiss_connector.rebuild_marker_file, 'w').close()
    rebuild = os.path.exists(faiss_connector.rebuild_marker_file)
    if rebuild:
        min_year = publications_db_connector.load_min_year()
        print(f'Rebuilding Faiss index for years {min_year}-{max_year}')

    indexed_pids = np.unique(pids_idx['pmid'].to_numpy())
    cumulative_updates = 0
    for year in range(max_year, min_year - 1, -1):
        print(f'Processing year {year}')
        pids_year = np.array(publications_db_connector.collect_pids_types_year(year), dtype=np.int64)
        pids_year = pids_year[~np.isin(pids_year, indexed_pids)].tolist()

        if test_limit_per_year is not None:
            print(f'Limit to {test_limit_per_year}')
            pids_year = pids_year[:test_limit_per_year]

        print(f'To process {len(pids_year)}')
        if len(pids_year) == 0:
            continue

        cumulative_updates += len(pids_year)

        update_faiss_index(
            df=pd.DataFrame(dict(pmid=pids_year)),
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=embeddings_model_connector,
        )
        if cumulative_updates >= FAISS_SAVE_EVERY:
            faiss_connector.save()
            cumulative_updates = 0  # Reset counter after save
    # Always save at the end if there are pending updates
    if cumulative_updates > 0 or rebuild:
        faiss_connector.save()
    if rebuild:
        os.remove(faiss_connector.rebuild_marker_file)
        print('Faiss index rebuild completed')


if __name__ == '__main__':
    # Disable tokenizers parallelism to avoid issues with multiprocessing
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description='Update embeddings')
    # Next year is included for papers published ahead of print
    parser.add_argument('--max-year', type=int, default=current_year + 1, help='Maximum year to process (inclusive)')
    parser.add_argument('--min-year', type=int, default=current_year - 1, help='Minimum year to process (inclusive)')
    parser.add_argument('--host', type=str, default='', help='Embeddings database host')
    parser.add_argument('--port', type=str, default='', help='Embeddings database port')
    parser.add_argument('--user', type=str, default='', help='Embeddings database user')
    parser.add_argument('--database', type=str, default='', help='Embeddings database name')
    parser.add_argument('--password', type=str, default='', help='Embeddings database password')
    parser.add_argument('--to-faiss', '--to-faiss-only', dest='to_faiss', action='store_true',
                        help='Update Faiss index from embeddings database, without computing embeddings')

    args = parser.parse_args()

    publications_db_connector = PublicationsDBConnector()

    # Initialize EmbeddingsDBConnector with command line arguments
    embeddings_db_connector = EmbeddingsDBConnector(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        embeddings_model_name=config.embeddings_model_name,
        embedding_dimension=config.embeddings_dimension
    )
    embeddings_db_connector.init_database()

    if args.to_faiss:
        print('Updating Faiss index from embeddings DB')
        # Imported here: Stage 4 doesn't need Faiss, Stage 5 doesn't need the embeddings model
        from pysrc.faiss.faiss_connector import FaissConnector
        # TODO: support other sources
        faiss_connector = FaissConnector(
            "Pubmed",
            embeddings_model_name=config.embeddings_model_name,
            embeddings_dimension=config.embeddings_dimension,
            create=True
        )
        update_faiss(
            publications_db_connector=publications_db_connector,
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
            max_year=args.max_year,
            min_year=args.min_year
        )
    else:
        print('Updating embeddings DB')
        from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
        update_embeddings(
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=EmbeddingsModelConnector(),
            embeddings_db_connector=embeddings_db_connector,
            max_year=args.max_year,
            min_year=args.min_year
        )
