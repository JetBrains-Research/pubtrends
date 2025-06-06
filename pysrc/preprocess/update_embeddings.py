import argparse
import logging

import numpy as np
import pandas as pd
from more_itertools import sliced
from tqdm.auto import tqdm

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
from pysrc.preprocess.embeddings.faiss_connector import FaissConnector
from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector
from pysrc.preprocess.embeddings.work_manager import WorkManager
from pysrc.config import PubtrendsConfig

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 2_000


def update_year(
        df: pd.DataFrame,
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector | None,
):
    wm = WorkManager(
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector,
    )
    print('Storing embeddings into DB')
    for index_slice in tqdm(list(sliced(range(len(df)), CHUNK_SIZE))):
        print(f'\rProcessing chunks {index_slice[0]}-{index_slice[-1]}          ', end='')
        chunk_df = df.iloc[index_slice]
        texts = [f'{title}. {abstract}' for title, abstract in zip(chunk_df['title'], chunk_df['abstract'])]
        wm.process_compute_and_store_embeddings_work(
            list(chunk_df['id']), texts, embeddings_model_connector.device == 'cpu'
        )
    # Finally, process the work left in the queue
    for _ in range(10):
        wm.process_compute_and_store_embeddings_work([], [], embeddings_model_connector.device == 'cpu')
    print('\rDone                           ')
    


def update_year_with_faiss_train(
        df: pd.DataFrame,
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector,
):
    print('Preparing embeddings for Faiss index training')
    update_year(
        df=df,
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=None,  # Don't store into Faiss before it's not trained
    )

    print('Sampling embeddings for Faiss index training')
    embeddings = embeddings_db_connector.sample_embeddings()

    print('Training Faiss index on embeddings')
    faiss_connector.faiss_index.train(embeddings)

    update_faiss_index(
        df=df,
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector
    )


def update_faiss_index(
        df: pd.DataFrame,
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector,
):
    print('Storing embeddings into Faiss')
    wm = WorkManager(
        publications_db_connector=publications_db_connector,
        embeddings_model_connector=embeddings_model_connector,
        embeddings_db_connector=embeddings_db_connector,
        faiss_connector=faiss_connector,  # Now we can store embeddings
    )
    for index_slice in tqdm(list(sliced(range(len(df)), CHUNK_SIZE))):
        print(f'\rProcessing chunks {index_slice[0]}-{index_slice[-1]}          ', end='')
        chunk_df = df.iloc[index_slice]
        wm.process_load_and_store_embeddings_to_faiss_work(
            list(list(chunk_df['id'])),
        )
    # Finally, process the work left in the queue
    for _ in range(10):
        wm.process_load_and_store_embeddings_to_faiss_work([])
    print('\rDone                           ')


def update_embeddings(
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector | None,
        max_year,
        min_year,
        test_limit_per_year = None
):
    if faiss_connector is not None:
        faiss_index, pids_idx = faiss_connector.create_or_load_faiss()
    else:
        faiss_index, pids_idx = None, None

    for year in range(max_year, min_year, - 1):
        print(f'Processing year {year}')
        df = publications_db_connector.fetch_year(year)
        pids_in_db = embeddings_db_connector.collect_ids_with_embeddings(df['id'])
        pids_to_add = list(sorted(set(df['id']) - set(pids_in_db)))
        print(f'To process {len(pids_to_add)}')
        df = df[df['id'].isin(pids_to_add)]

        if test_limit_per_year is not None:
            print(f'Limit to {test_limit_per_year}')
            df = df.head(test_limit_per_year)

        if faiss_index is not None and faiss_index.ntotal == 0:
            print(f'Empty Faiss index')
            update_year_with_faiss_train(
                df,
                publications_db_connector=publications_db_connector,
                embeddings_model_connector=embeddings_model_connector,
                embeddings_db_connector=embeddings_db_connector,
                faiss_connector=faiss_connector,
            )
        else:
            update_year(
                df,
                publications_db_connector=publications_db_connector,
                embeddings_model_connector=embeddings_model_connector,
                embeddings_db_connector=embeddings_db_connector,
                faiss_connector=faiss_connector,
            )
        if faiss_connector is not None:
            faiss_connector.save()


def update_faiss(
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector,
        max_year,
        min_year,
):
    faiss_index, pids_idx = faiss_connector.create_or_load_faiss()
    if faiss_index.ntotal == 0:
        print('Sampling embeddings for Faiss index training')
        embeddings = embeddings_db_connector.sample_embeddings()

        print('Training Faiss index on embeddings')
        faiss_index.train(embeddings)

    for year in range(max_year, min_year, - 1):
        print(f'Processing year {year}')
        print(f'Collecting {year}')
        pids_year = publications_db_connector.collect_pids_types_year(year)
        pids_embds = embeddings_db_connector.collect_ids_with_embeddings(pids_year)
        pids_to_add = list(sorted(set(pids_year).intersection(set(pids_embds)) - set(pids_idx['pmid'])))
        print(f'To process {len(pids_to_add)}')
        if len(pids_to_add) == 0:
            continue
        update_faiss_index(
            df=pd.DataFrame(dict(id=pids_to_add)),
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=embeddings_model_connector,
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
        )
        faiss_connector.save()


if __name__ == '__main__':
    print('Updating embeddings')

    parser = argparse.ArgumentParser(description='Update embeddings')
    parser.add_argument('--max-year', type=int, required=True, help='Maximum year to process')
    parser.add_argument('--min-year', type=int, required=True, help='Minimum year to process')
    parser.add_argument('--host', type=str, required=True, help='Embeddings database host')
    parser.add_argument('--port', type=str, required=True, help='Embeddings database port')
    parser.add_argument('--user', type=str, required=True, help='Embeddings database user')
    parser.add_argument('--database', type=str, required=True, help='Embeddings database name')
    parser.add_argument('--password', type=str, required=True, help='Embeddings database password')
    parser.add_argument('--pgvector-only', action='store_true',
                        help='Update only embeddings database')
    parser.add_argument('--faiss-only', action='store_true',
                        help='Update only faiss embeddings from embeddings database')

    args = parser.parse_args()
    assert not (args.pgvector_only and args.faiss_only), "Cannot use --pgvector-only with --faiss-only"

    # Initialize connectors
    publications_db_connector = PublicationsDBConnector()
    embeddings_model_connector = EmbeddingsModelConnector()

    # Initialize EmbeddingsDBConnector with command line arguments
    embeddings_db_connector = EmbeddingsDBConnector(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        embeddings_model_name=embeddings_model_connector.embeddings_model_name,
        embedding_dimension=embeddings_model_connector.embeddings_dimension
    )

    # Initialize FaissConnector
    faiss_connector = FaissConnector(
        embeddings_model_name=embeddings_model_connector.embeddings_model_name,
        embeddings_dimension=embeddings_model_connector.embeddings_dimension
    ) if not args.pgvector_only else None

    # Call update_embeddings with the initialized connectors and command line arguments
    if args.faiss_only:
        print('Updating faiss embeddings only')
        update_faiss(
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=embeddings_model_connector,
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
            max_year=args.max_year,
            min_year=args.min_year
        )
    else:
        if args.pgvector_only:
            print('Updating embeddings database only')
        else:
            print('Updating embeddings')
        update_embeddings(
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=embeddings_model_connector,
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
            max_year=args.max_year,
            min_year=args.min_year
        )
