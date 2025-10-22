import argparse
import logging
import os
from datetime import datetime

import pandas as pd
from more_itertools import sliced
from tqdm.auto import tqdm

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
from pysrc.faiss.faiss_connector import FaissConnector
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
    faiss_index, pids_idx = faiss_connector.create_or_load_faiss()
    for year in range(max_year, min_year, - 1):
        print(f'Processing year {year}')
        df = publications_db_connector.load_publications_year(year)
        pids_to_process = set(embeddings_db_connector.collect_ids_without_embeddings(df['id']))
        print(f'To process {len(pids_to_process)}')
        df = df[df['id'].isin(pids_to_process)]

        if test_limit_per_year is not None:
            print(f'Limit to {test_limit_per_year}')
            df = df.head(test_limit_per_year)

        if faiss_index.ntotal == 0:
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
        faiss_connector.save()


def update_faiss(
        publications_db_connector: PublicationsDBConnector,
        embeddings_model_connector: EmbeddingsModelConnector,
        embeddings_db_connector: EmbeddingsDBConnector,
        faiss_connector: FaissConnector,
        max_year,
        min_year,
        test_limit_per_year = None
):
    faiss_index, pids_idx = faiss_connector.create_or_load_faiss()
    assert faiss_index.ntotal != 0

    for year in range(max_year, min_year, - 1):
        print(f'Processing year {year}')
        pids_year = publications_db_connector.collect_pids_types_year(year)
        pids_year = list(set(pids_year) - set(pids_idx['pmid']))

        if test_limit_per_year is not None:
            print(f'Limit to {test_limit_per_year}')
            pids_year = pids_year[:test_limit_per_year]

        print(f'To process {len(pids_year)}')
        if len(pids_year) == 0:
            continue
        update_faiss_index(
            df=pd.DataFrame(dict(id=pids_year)),
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=embeddings_model_connector,
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
        )
        faiss_connector.save()


if __name__ == '__main__':
    # Disable tokenizers parallelism to avoid issues with multiprocessing
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description='Update embeddings')
    parser.add_argument('--max-year', type=int, default=current_year, help='Maximum year to process')
    parser.add_argument('--min-year', type=int, default=current_year - 1, help='Minimum year to process')
    parser.add_argument('--host', type=str, default='', help='Embeddings database host')
    parser.add_argument('--port', type=str, default='', help='Embeddings database port')
    parser.add_argument('--user', type=str, default='', help='Embeddings database user')
    parser.add_argument('--database', type=str, default='', help='Embeddings database name')
    parser.add_argument('--password', type=str, default='', help='Embeddings database password')
    parser.add_argument('--to-faiss-only', type=bool, default=False,
                        help='Update faiss embeddings only from embeddings database')

    args = parser.parse_args()

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
        embeddings_model_name=config.embeddings_model_name,
        embedding_dimension=config.embeddings_dimension
    )
    embeddings_db_connector.init_database()

    # TODO: support other sources
    # Initialize FaissConnector
    faiss_connector = FaissConnector(
        "Pubmed",
        embeddings_model_name=config.embeddings_model_name,
        embeddings_dimension=config.embeddings_dimension
    )

    # Call update_embeddings with the initialized connectors and command line arguments
    if args.to_faiss_only:
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
        print('Updating embeddings')
        update_embeddings(
            publications_db_connector=publications_db_connector,
            embeddings_model_connector=embeddings_model_connector,
            embeddings_db_connector=embeddings_db_connector,
            faiss_connector=faiss_connector,
            max_year=args.max_year,
            min_year=args.min_year
        )
