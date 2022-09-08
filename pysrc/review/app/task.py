import logging
import os
from threading import Lock

import sys
import torch
from celery import current_task
from lazy import lazy
from os.path import dirname

from pysrc.celery.pubtrends_celery import pubtrends_celery
from pysrc.papers.analysis.citations import find_top_cited_papers
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.progress import Progress
from pysrc.papers.config import PubtrendsConfig

# Configure source path for pubtrends-review repository
sys.path.append(os.path.abspath(f'{dirname(__file__)}/../../../pubtrends-review'))

import review.config as cfg
from review.model import setup_cuda_device, load_model
from review.text import text_to_data

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


@pubtrends_celery.task(name='prepare_review_data_async')
def prepare_review_data_async(data, source, num_papers, num_sents):
    progress = Progress(total=5)
    result = generate_review(data, source, num_papers, num_sents, progress=progress, task=current_task)
    progress.done('Done review', task=current_task)
    return result


# Deployment and development
MODEL_PATHS = ['/model', os.path.expanduser('~/.pubtrends/model')]


class ModelCache:
    @lazy
    def model_and_device(self):
        logger.info('Loading base BERT model')
        model = load_model("bert", "froze_all", 512)
        model, device = setup_single_gpu(model)
        # TODO: add model path to config properties
        for model_path in [os.path.join(p, cfg.model_name) for p in MODEL_PATHS]:
            if os.path.exists(model_path):
                logger.info(f'Loading trained model weights {cfg.model_name}')
                model.load(model_path)
                break
        else:
            raise RuntimeError(f'Model weights file {cfg.model_name} not found among: {MODEL_PATHS}')
        return model, device


MODEL_CACHE = ModelCache()

REVIEW_LOCK = Lock()


def generate_review(data, source, num_papers, num_sents, progress, task):
    try:
        REVIEW_LOCK.acquire()
        progress.info(f'Generating review', current=1, task=task)
        loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
        analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
        analyzer.init(data)
        progress.info('Initializing model and device', current=2, task=task)
        model, device = MODEL_CACHE.model_and_device
        progress.info('Configuring model for evaluation', current=3, task=task)
        model.eval()
        top_cited_papers, top_cited_df = find_top_cited_papers(
            analyzer.df, n_papers=int(num_papers)
        )
        progress.info(f'Processing abstracts for {len(top_cited_papers)} top cited papers', current=4, task=task)
        result = []
        for pid in top_cited_papers:
            logger.debug(f'Processing review for {pid}')
            cur_paper = top_cited_df[top_cited_df['id'] == pid]
            logger.debug(f'Found {len(cur_paper)} papers for id')
            title = cur_paper['title'].values[0]
            year = int(cur_paper['year'].values[0])
            cited = int(cur_paper['total'].values[0])
            abstract = cur_paper['abstract'].values[0]
            logger.info(f'Length of abstract {len(abstract)}')
            topic = int(cur_paper['comp'].values[0] + 1)
            choose_from = predict_review_score(device, model, abstract)
            to_add = sorted(choose_from, key=lambda x: -x[1])[:int(num_sents)]
            for sent, score in to_add:
                result.append([title, year, cited, topic, sent, url_prefix + pid, float("{:.2f}".format(score))])
            logger.debug(f'Review result contains {len(result)} records')
        return result
    finally:
        REVIEW_LOCK.release()


def predict_review_score(device, model, abstract):
    data = text_to_data(abstract, 512, model.tokenizer)
    logger.debug(f'Data to process {len(data)}')
    result = []
    for input_ids, attention_mask, token_type_ids, offset, sents in data:
        input_ids = torch.tensor([input_ids]).to(device)
        attention_mask = torch.tensor([attention_mask]).to(device)
        token_type_ids = torch.tensor([token_type_ids]).to(device)
        model_scores = model(input_ids, attention_mask, token_type_ids, )
        result.extend(zip(sents[offset:], model_scores.cpu().detach().numpy()[offset:]))
    return result
