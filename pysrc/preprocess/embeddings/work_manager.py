import logging
import queue
import threading

from pysrc.preprocess.embeddings.embeddings_db_connector import EmbeddingsDBConnector
from pysrc.preprocess.embeddings.embeddings_model_connector import EmbeddingsModelConnector
from pysrc.preprocess.embeddings.faiss_connector import FaissConnector
from pysrc.preprocess.embeddings.publications_db_connector import PublicationsDBConnector
from pysrc.config import PubtrendsConfig
from pysrc.papers.analysis.text import parallel_collect_chunks

config = PubtrendsConfig(test=False)

logger = logging.getLogger(__name__)

TEXT_CHUNK_SIZE = 128


class WorkManager:
    def __init__(
            self,
            publications_db_connector: PublicationsDBConnector,
            embeddings_model_connector: EmbeddingsModelConnector | None,
            embeddings_db_connector: EmbeddingsDBConnector,
            faiss_connector: FaissConnector | None,
    ):
        self.publications_db_connector = publications_db_connector
        self.embeddings_model_connector = embeddings_model_connector
        self.embeddings_db_connector = embeddings_db_connector
        self.faiss_connector = faiss_connector

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
                self.faiss_connector.store_embeddings(chunk_embeddings, chunk_idx)
        except queue.Empty:
            pass

    def compute_embeddings_and_store_work(self):
        try:
            chunks, chunk_idx = self.chunks_queue.get_nowait()  # Non-blocking
            chunk_embeddings = self.embeddings_model_connector.batch_texts_embeddings(chunks)
            if self.embeddings_db_connector is not None:
                self.embeddings_db_connector.store_embeddings_to_postgresql(chunk_embeddings, chunk_idx)
            if self.faiss_connector is not None:
                self.faiss_connector.store_embeddings(chunk_embeddings, chunk_idx)
        except queue.Empty:
            pass

    def process_compute_and_store_embeddings_work(self, pids, texts, cpu):
        assert self.publications_db_connector is not None
        assert self.embeddings_model_connector is not None
        assert self.embeddings_db_connector is not None

        # Create threads
        if cpu:
            threads = [
                threading.Thread(target=self.collect_chunks_from_texts_work, args=(pids, texts)),
                threading.Thread(target=self.compute_embeddings_and_store_work, args=()),
            ]
        else:
            threads = [
                threading.Thread(target=self.collect_chunks_from_texts_work, args=(pids, texts)),
                threading.Thread(target=self.compute_embeddings_work, args=()),
                threading.Thread(target=self.store_embeddings_work, args=())
            ]
        # Start the threads
        for t in threads:
            t.start()
        # Wait for both threads to complete
        for t in threads:
            t.join()

    def load_embeddings_work(self, pids):
        if len(pids) == 0:
            return
        index, embeddings = self.embeddings_db_connector.load_embeddings_by_ids(pids)
        self.embeddings_queue.put((index, embeddings))


    def store_embeddings_to_faiss_work(self):
        try:
            chunk_embeddings, chunk_idx = self.embeddings_queue.get_nowait()  # Non-blocking
            self.faiss_connector.store_embeddings(chunk_embeddings, chunk_idx)
        except queue.Empty:
            pass

    def process_load_and_store_embeddings_to_faiss_work(self, pids):
        assert self.publications_db_connector is not None
        assert self.embeddings_db_connector is not None
        assert self.faiss_connector is not None
        # Create threads
        threads = [
            threading.Thread(target=self.load_embeddings_work, args=([pids])),
            threading.Thread(target=self.store_embeddings_to_faiss_work, args=()),
        ]
        # Start the threads
        for t in threads:
            t.start()
        # Wait for both threads to complete
        for t in threads:
            t.join()
