import logging
import math
import pandas as pd
import re
import spacy
from nltk.stem import WordNetLemmatizer
from spacy import displacy
from text_to_num import alpha2digit

NUMBER = re.compile(r'-?[\d]+([\.,][\d]+)?([eE][+-]?\d+)?')
ENTITY = re.compile(r'[a-zA-Z_]+[a-zA-Z0-9_\-]*')
URL = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")
SMALL_SENTENCE = 3
spacy_en = spacy.load('en_core_web_sm')

lemmatizer = WordNetLemmatizer()


def extract_numbers(df):
    """
    Extracts number from the titles of papers
    :param df: Papers dataframe
    :return: Dataframe [id, title, numbers]
    """
    metrics_data = []
    for _, data in df.iterrows():
        paper_metrics_data = [data['id'], *extract_metrics(data['abstract'])]
        metrics_data.append(paper_metrics_data)
    metrics_df = pd.DataFrame(metrics_data, columns=['ID', 'Metrics', 'Sentences'])
    result = pd.merge(left=metrics_df, left_on='ID', right=df[['id', 'title']], right_on='id')
    result = result[['id', 'title', 'Metrics']]
    result['numbers'] = [
        '; '.join(
            f'{number}:{",".join(str(v) for v in sorted(set(v[0] for v in values)))}'
            for number, values in row['Metrics'].items()
        ) for _, row in result.iterrows()
    ]
    result = result.loc[result['numbers'] != '']
    return result[['id', 'title', 'numbers']]


def _preprocess_text(text):
    # Replace all emails / urls to avoid irrelevant numbers
    text = re.sub(URL, "", text)
    # Replace non-ascii or punctuation with space
    text = re.sub('[^a-zA-Z0-9-+.,:!?]+', ' ', text, flags=re.IGNORECASE)
    # Whitespaces normalization, see #215
    text = re.sub('\s{2,}', ' ', text.strip())
    # Convert textual numbers to digits (three -> 3)
    text = alpha2digit(text, 'en', relaxed=True)
    # Convect 1st, 2nd, 3rd, 4th
    text = re.sub(r"([\d]+)(st|nd|rd|th)", r"\g<1>", text, flags=re.IGNORECASE)
    return text


def _process_candidate(metrics, token, value, idx):
    tt = token.text
    if ENTITY.fullmatch(tt) and token.pos_ in {'NOUN', 'PROPN'}:
        if re.match(r'[A-Z\-0-9_]+s', tt):  # plural of abbreviation
            tt = tt[:-1]
        tt = lemmatizer.lemmatize(tt.lower())
        if tt not in metrics:
            metrics[tt] = []
        metrics[tt].append((value, idx))
        return True
    return False


def process_number(token, value, idx, metrics):
    # logging.debug(f'Number: {value}')
    if token.head.pos_ == 'NUM':
        tht = token.head.text
        if re.match(r'hundred(s?)|100', tht, flags=re.IGNORECASE):
            value *= 100
        elif re.match(r'thousand(s?)|1000', tht, flags=re.IGNORECASE):
            value *= 1000
        elif re.match(r'million(s?)|1000000', tht, flags=re.IGNORECASE):
            value *= 1000000
        elif re.match(r'billion(s?)|1000000000', tht, flags=re.IGNORECASE):
            value *= 1000000000
        # logging.debug(f'Value adjusted: {value}')
        token = next(token.ancestors, token)

    # Analyze children and siblings, then ancestors if first was not enough
    # TODO: is there a better way?
    # TODO: use close nouns as a fallback when it is hard to find a dependency?
    # TODO: expand nouns with adjectives or other nouns? (rate -> information transfer rate)
    # logging.debug(f'Token children: {",".join(t.text for t in token.children)}')
    for t in token.children:
        if t != token and _process_candidate(metrics, t, value, idx):
            # logging.debug(f'Child term: {t.text}')
            return

    # logging.debug('Head with children: '
    #               f'{token.head.text} | {",".join(t.text for t in token.head.children)}')
    if token != token.head:
        if _process_candidate(metrics, token.head, value, idx):
            # logging.debug(f'Head term: {token.head.text}')
            return

        for t in token.head.children:
            if t != token and _process_candidate(metrics, t, value, idx):
                # logging.debug(f'Child of head term: {t.text}')
                return

    # logging.debug(f'Token anscestors: {",".join(t.text for t in token.ancestors)}')
    for i, t in enumerate(token.ancestors):
        if i == 3:  # Don't go too high
            return
        if t != token and _process_candidate(metrics, t, value, idx):
            # logging.debug(f'Ancestor: {t.text}')
            return


def extract_metrics(text, visualize_dependencies=False):
    """
    Parses abstract and returns a dict of numbers with nouns that could be suitable as a metric.
    :return list of tuples (sentence, [metrics]), where metrics is a list of tuples (number, [nouns], sentence_number)
    """
    text = _preprocess_text(text)
    # Split text into sentences and find numbers in sentences
    doc = spacy_en(text)
    metrics = {}
    sentences = {}
    for idx, sent in enumerate(doc.sents):
        # # logging.debug('###')
        # # logging.debug(sent.text)
        if len(sent) < SMALL_SENTENCE:
            continue
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
