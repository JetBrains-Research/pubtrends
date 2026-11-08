from __future__ import annotations

import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.papers.analysis.text import parallel_collect_chunks, paper_text

if TYPE_CHECKING:
    # Imported for type hints only: model module loads the model on import, faiss module requires faiss directory
    from pysrc.faiss.faiss_connector import FaissConnector
    from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
    from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector

logger = logging.getLogger(__name__)

TEXT_CHUNK_SIZE = 128


def run_in_parallel(*tasks):
    """Run tasks in threads and re-raise the first failure (plain threads swallow exceptions)"""
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(task) for task in tasks]
        for f in futures:
            f.result()


class WorkManager:
    def __init__(
            self,
            embeddings_model_connector: EmbeddingsModelConnector | None,
            embeddings_db_connector: EmbeddingsDBConnector,
            faiss_connector: FaissConnector | None,
            publications_db_connector: PublicationsDBConnector | None = None,
    ):
        self.embeddings_model_connector = embeddings_model_connector
        self.embeddings_db_connector = embeddings_db_connector
        self.faiss_connector = faiss_connector
        self.publications_db_connector = publications_db_connector

        # Thread safe queue to store chunks
        self.chunks_queue = queue.Queue()
        # Thread safe queue to store embeddings
        self.embeddings_queue = queue.Queue()

    def empty_queues(self):
        while not self.chunks_queue.empty():
            self.chunks_queue.get()
        while not self.embeddings_queue.empty():
            self.embeddings_queue.get()

    def collect_chunks_from_texts_work(self, pids, texts):
        if pids is None or texts is None:
            return
        chunks, chunk_idx = parallel_collect_chunks(pids, texts, TEXT_CHUNK_SIZE)
        self.chunks_queue.put((chunks, chunk_idx))

    def compute_embeddings_work(self):
        try:
            chunks, chunk_idx = self.chunks_queue.get_nowait()  # Non-blocking
            chunk_embeddings = self.embeddings_model_connector.batch_texts_embeddings(chunks)
            self.embeddings_queue.put((chunk_embeddings, chunk_idx))
        except queue.Empty:
            pass

    def store_embeddings_work(self):
        try:
            chunk_embeddings, chunk_idx = self.embeddings_queue.get_nowait()  # Non-blocking
            if self.embeddings_db_connector is not None:
                self.embeddings_db_connector.store_embeddings_to_postgresql(chunk_embeddings, chunk_idx)
            if self.faiss_connector is not None:
                self.faiss_connector.store_embeddings(chunk_idx, chunk_embeddings)
        except queue.Empty:
            pass

    def compute_embeddings_and_store_work(self):
        try:
            chunks, chunk_idx = self.chunks_queue.get_nowait()  # Non-blocking
            chunk_embeddings = self.embeddings_model_connector.batch_texts_embeddings(chunks)
            if self.embeddings_db_connector is not None:
                self.embeddings_db_connector.store_embeddings_to_postgresql(chunk_embeddings, chunk_idx)
            if self.faiss_connector is not None:
                self.faiss_connector.store_embeddings(chunk_idx, chunk_embeddings)
        except queue.Empty:
            pass

    def process_compute_and_store_embeddings_work(self, pids, texts, cpu):
        if self.embeddings_model_connector is None:
            raise RuntimeError("embeddings_model_connector is required but not configured")
        if self.embeddings_db_connector is None:
            raise RuntimeError("embeddings_db_connector is required but not configured")

        if cpu:
            run_in_parallel(
                lambda: self.collect_chunks_from_texts_work(pids, texts),
                self.compute_embeddings_and_store_work,
            )
        else:
            run_in_parallel(
                lambda: self.collect_chunks_from_texts_work(pids, texts),
                self.compute_embeddings_work,
                self.store_embeddings_work,
            )

    def finish_compute_and_store_embeddings_work(self, cpu):
        while not self.chunks_queue.empty() or not self.embeddings_queue.empty():
            if cpu:
                self.compute_embeddings_and_store_work()
            else:
                run_in_parallel(
                    self.compute_embeddings_work,
                    self.store_embeddings_work,
                )

    def load_embeddings_work(self, pids):
        if len(pids) == 0:
            return
        index, embeddings = self.embeddings_db_connector.load_embeddings_by_ids(pids)
        # Check if some PMIDs are missing embeddings and can be recovered
        loaded_pmids = {p for p, _ in index}
        missing_pids = [p for p in pids if p not in loaded_pmids]
        if missing_pids and self.publications_db_connector is not None:
            try:
                pubs_df = self.publications_db_connector.load_publications(missing_pids)
                if not pubs_df.empty:
                    texts = [paper_text(t, a) for t, a in zip(pubs_df['title'], pubs_df['abstract'])]
                    valid_pubs = [(p, t) for p, t in zip(pubs_df['pmid'], texts) if t]
                    if valid_pubs:
                        rec_pids = [p for p, _ in valid_pubs]
                        rec_texts = [t for _, t in valid_pubs]
                        chunks, chunk_idx = parallel_collect_chunks(rec_pids, rec_texts, TEXT_CHUNK_SIZE)
                        if chunks:
                            if self.embeddings_model_connector is None:
                                from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
                                self.embeddings_model_connector = EmbeddingsModelConnector()
                            rec_embeddings = self.embeddings_model_connector.batch_texts_embeddings(chunks)
                            self.embeddings_db_connector.store_embeddings_to_postgresql(rec_embeddings, chunk_idx)
                            index.extend(chunk_idx)
                            embeddings.extend(rec_embeddings)
            except Exception as e:
                logger.warning(f"Failed to recover missing embeddings for {len(missing_pids)} pids: {e}")
        self.embeddings_queue.put((index, embeddings))

    def store_embeddings_to_faiss_work(self):
        try:
            chunk_idx, chunk_embeddings = self.embeddings_queue.get_nowait()  # Non-blocking
            self.faiss_connector.store_embeddings(chunk_idx, chunk_embeddings)
        except queue.Empty:
            pass

    def process_load_and_store_embeddings_to_faiss_work(self, pids):
        if self.embeddings_db_connector is None:
            raise RuntimeError("embeddings_db_connector is required but not configured")
        if self.faiss_connector is None:
            raise RuntimeError("faiss_connector is required but not configured")
        run_in_parallel(
            lambda: self.load_embeddings_work(pids),
            self.store_embeddings_to_faiss_work,
        )

    def finish_load_and_store_embeddings_to_faiss_work(self):
        while not self.embeddings_queue.empty():
            self.store_embeddings_to_faiss_work()
