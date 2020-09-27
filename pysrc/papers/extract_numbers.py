import logging
import math
import re
from collections import Counter

import pandas as pd
import spacy
from nltk.stem import WordNetLemmatizer
from spacy import displacy
from text_to_num import alpha2digit

NUMBER = re.compile(r'-?[\d]+([\.,][\d]+)?([eE][+-]?\d+)?')
spacy_en = spacy.load('en_core_web_sm')

lemmatizer = WordNetLemmatizer()


def process_candidate(metrics, token, value, idx):
    tt = token.text
    if tt != '%' and token.pos_ in set(['NOUN', 'PROPN']):
        tt = lemmatizer.lemmatize(tt)
        if tt not in metrics:
            metrics[tt] = []
        metrics[tt].append((value, idx))
        return True
    return False


def process_number(token, value, idx, metrics):
    logging.debug(f'Number: {value}')
    if token.head.pos_ == 'NUM':
        tht = token.head.text
        if tht == 'hundred':
            value *= 100
        elif tht == 'thousand':
            value *= 1000
        elif tht == 'million':
            value *= 1000000
        elif tht == 'billion':
            value *= 1000000000
        logging.debug(f'Value adjusted: {value}')
        token = next(token.ancestors, token)

    # Analyze children and siblings, then ancestors if first was not enough
    # TODO: is there a better way?
    # TODO: use close nouns as a fallback when it is hard to find a dependency?
    # TODO: expand nouns with adjectives or other nouns? (rate -> information transfer rate)
    # logging.debug(f'Token children: {",".join(t.text for t in token.children)}')
    children_matched = False
    for t in token.children:
        if process_candidate(metrics, t, value, idx):
            logging.debug(f'Child term: {t.text}')
            children_matched = True
        if children_matched:
            return
    # logging.debug('Head with children: '
    #              f'{token.head.text} | {",".join(t.text for t in token.head.children)}')
    if process_candidate(metrics, token.head, value, idx):
        logging.debug(f'Head term: {token.head.text}')
        return

    head_children_matched = False
    for t in token.head.children:
        if process_candidate(metrics, t, value, idx):
            logging.debug(f'Child of head term: {t.text}')
            head_children_matched = True
    if head_children_matched:
        return

    logging.debug(f'Token anscestors: {",".join(t.text for t in token.ancestors)}')
    for t in token.ancestors:
        if process_candidate(metrics, t, value, idx):
            logging.debug(f'Ancestor: {t.text}')
            return


def extract_metrics(text, visualize_dependencies=False):
    """
    Parses abstract and returns a dict of numbers with nouns that could be suitable as a metric.
    :return list of tuples (sentence, [metrics]), where metrics is a list of tuples (number, [nouns], sentence_number)
    """
    metrics = {}
    sentences = {}
    # Convert textual numbers to digits (three -> 3)
    text = alpha2digit(text.lower(), 'en', relaxed=True)
    # Convect 10th -> 10
    text = re.sub(r"([\d]+)th", r"\g<1>", text)
    # Split text into sentences and find numbers in sentences
    doc = spacy_en(text)
    for idx, sent in enumerate(doc.sents):
        # logging.debug('###')
        # logging.debug(sent.text)
        sentences[idx] = sent.text
        for token in sent:
            #             print(token.text, token.pos_, list(token.ancestors))
            if NUMBER.fullmatch(token.text):
                tt = token.text.replace(',', '')  # TODO: can fail in case of different locale
                try:
                    value = float(tt)
                    if math.ceil(value) == value:
                        value = int(value)
                    process_number(token, value, idx, metrics)
                except Exception as e:
                    logging.error(f'/Failed to process number {tt}', e)
        if visualize_dependencies:
            displacy.render(sent, style="dep", jupyter=True)
    return metrics, sentences


class MetricExtractor:
    def __init__(self, metrics_data):
        self.metrics_df = pd.DataFrame(metrics_data, columns=['ID', 'Metrics', 'Sentences'])

    def get_top_metrics(self, number=20):
        metrics_counter = Counter()
        for metric_dict in self.metrics_df['Metrics']:
            for metric, occasions in metric_dict.items():
                metrics_counter[metric] += len(occasions)
        return metrics_counter.most_common(number)

    def get_metric_values(self, *metrics, min_value=None, max_value=None, detailed=False):
        values = []
        for _, data in self.metrics_df.iterrows():
            metric_dict = data['Metrics']
            sentences = data['Sentences']

            for metric in metrics:
                if metric in metric_dict:
                    for value, sentence_number in metric_dict[metric]:
                        if min_value and value < min_value or max_value and value > max_value:
                            continue
                        if detailed:
                            sentence = sentences[sentence_number]
                            values.append([data['ID'], value, sentence])
                        else:
                            values.append(value)
        if detailed:
            return pd.DataFrame(values, columns=['PMID', ', '.join(metrics), 'Sentence'])
        return values

    def filter_papers(self, metrics):
        """
        :param metrics - list of tuples ([list of keywords], min_value, max_value)
               e.g. (['subjects', 'participants'], 5, None)
        :return list of PMIDs
        """
        selection = []
        for _, data in self.metrics_df.iterrows():
            suitable = True
            metric_dict = data['Metrics']

            for metric in metrics:
                metric_suitable = False
                words, min_value, max_value = metric

                for word in words:
                    if word in metric_dict:
                        for value, _ in metric_dict[word]:
                            if min_value and value < min_value or max_value and value > max_value:
                                continue
                            metric_suitable = True
                    if metric_suitable:
                        break

                suitable &= metric_suitable

            if suitable:
                selection.append(data['ID'])
        return selection
