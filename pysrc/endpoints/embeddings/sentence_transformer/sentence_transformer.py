import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import onnxruntime as ort
from lazy import lazy
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from pysrc.config import PubtrendsConfig

logger = logging.getLogger(__name__)

BACKENDS = {"onnx", "torch"}

MODEL_PATHS = ['/sentence-transformers', os.path.expanduser('~/.pubtrends/sentence-transformers')]
for p in MODEL_PATHS:
    if os.path.isdir(p):
        # Configure HuggingFace models cache directory
        os.environ['HF_HOME'] = p
        break
else:
    raise RuntimeError('Failed to configure model cache directory')


class SentenceTransformerModel:
    def __init__(self):
        config = PubtrendsConfig()
        self.model_name = config.sentence_transformer_model
        self.embeddings_dimension = config.embeddings_dimension
        self.backend = self._resolve_backend()
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        self.max_threads = max(1, (os.cpu_count() or 1) - 1)
        self._configure_threading_env()
        # noinspection PyStatementEffect
        self.download_and_load_model

    def _configure_threading_env(self):
        thread_count = str(self.max_threads)
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["NUMEXPR_NUM_THREADS"] = thread_count
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    @lazy
    def download_and_load_model(self):
        if self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_torch()

        return self

    def encode(self, texts, batch_size=8, show_progress_bar=False):
        if self.backend == "onnx":
            return self._encode_onnx(texts, batch_size=batch_size)
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar
        )

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        summed = np.sum(last_hidden_state * mask, axis=1)
        denom = np.clip(np.sum(mask, axis=1), 1e-9, None)
        return summed / denom

    def _encode_onnx(self, texts, batch_size):
        if not texts:
            return np.zeros((0, self.embeddings_dimension), dtype=np.float32)

        def encode_batch(batch_texts):
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='np',
            )
            outputs = self.model(**inputs)
            pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            return normalized.astype(np.float32)

        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        if len(batches) == 1:
            return encode_batch(batches[0])

        worker_count = min(self.max_threads, len(batches))
        vectors_by_idx = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(encode_batch, batch): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                vectors_by_idx[futures[future]] = future.result()

        return np.vstack(vectors_by_idx)

    @staticmethod
    def _resolve_backend():
        backend = os.getenv("PUBK_EMBEDDINGS_BACKEND", "onnx").strip().lower()
        if backend not in BACKENDS:
            raise ValueError(f'Unsupported PUBK_EMBEDDINGS_BACKEND={backend}. Use "onnx" or "torch".')
        return backend

    def _load_onnx(self):
        logger.info(f"Loading model {self.model_name} with ONNX runtime")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        options.intra_op_num_threads = self.max_threads
        options.inter_op_num_threads = 1
        logger.info(
            "ONNX threading configured with intra_op_num_threads=%s inter_op_num_threads=%s",
            self.max_threads,
            1,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_name,
            export=True,
            provider="CPUExecutionProvider",
            session_options=options,
        )
        self.max_length = self._resolve_max_length()
        self.device = "cpu"

    def _load_torch(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch backend requires optional GPU dependencies. "
                "Install with: uv sync --no-install-project --extra gpu"
            ) from exc

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Loading model {self.model_name} into {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        # Utilize parallelism
        torch.set_num_threads(self.max_threads)
        torch.set_num_interop_threads(1)
        logger.info(
            "Torch threading configured with intra_op_num_threads=%s inter_op_num_threads=%s",
            self.max_threads,
            1,
        )

    def _resolve_max_length(self):
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        model_max = getattr(self.model.config, "max_position_embeddings", None)

        candidates = [
            value
            for value in (tokenizer_max, model_max)
            if isinstance(value, int) and 0 < value < 100_000
        ]
        max_length = min(candidates) if candidates else 512

        logger.info(f"Using ONNX tokenizer max_length={max_length}")
        return max_length

logger.info('Prepare embeddings pretrained model')
SENTENCE_TRANSFORMER_MODEL = SentenceTransformerModel()
logger.info('Model is ready')
