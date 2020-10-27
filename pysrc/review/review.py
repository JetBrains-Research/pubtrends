import logging

import torch
from lazy import lazy

from pysrc.papers.analyzer import KeyPaperAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders
from pysrc.review.model import load_model
from pysrc.review.text import text_to_data
from pysrc.review.train.main import setup_single_gpu

logger = logging.getLogger(__name__)

PUBTRENDS_CONFIG = PubtrendsConfig(test=False)

REVIEW_ANALYSIS_TITLE = 'review'


class ModelCache:
    @lazy
    def model_and_device(self):
        logger.info('Loading BERT model')
        model = load_model("bert", "froze_all", 512)
        model, gpu = setup_single_gpu(model)
        return model, gpu


MODEL_CACHE = ModelCache()


def generate_review(data, source, num_papers, num_sents, progress, task):
    progress.info(f'Initializing analyzer for review', current=1, task=task)
    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = KeyPaperAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)
    progress.info('Initializing model and device', current=2, task=task)
    model, device = MODEL_CACHE.model_and_device
    progress.info('Configuring model for evaluation', current=3, task=task)
    model.eval()
    top_cited_papers, top_cited_df = analyzer.find_top_cited_papers(
        analyzer.df, n_papers=int(num_papers)
    )
    progress.info(f'Processing abstracts for {len(top_cited_papers)} top cited papers', current=4, task=task)
    result = []
    for id in top_cited_papers:
        cur_paper = top_cited_df[top_cited_df['id'] == id]
        title = cur_paper['title'].values[0]
        year = cur_paper['year'].values[0]
        cited = cur_paper['total'].values[0]
        abstract = cur_paper['abstract'].values[0]
        topic = cur_paper['comp'].values[0] + 1
        data = text_to_data(abstract, 512, model.tokenizer)
        choose_from = []
        for article_ids, article_mask, article_seg, magic, sents in data:
            input_ids = torch.tensor([article_ids]).to(device)
            input_mask = torch.tensor([article_mask]).to(device)
            input_segment = torch.tensor([article_seg]).to(device)
            draft_probs = model(
                input_ids, input_mask, input_segment,
            )
            choose_from.extend(zip(sents[magic:], draft_probs.cpu().detach().numpy()[magic:]))
        to_add = sorted(choose_from, key=lambda x: -x[1])[:int(num_sents)]
        for sent, score in to_add:
            result.append([title, year, cited, topic, sent, url_prefix + id, score])
    return result
