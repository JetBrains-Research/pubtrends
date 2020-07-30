import re

import pandas as pd
import torch
import torch.distributed as distrib
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from unidecode import unidecode

import pysrc.review.config as cfg
from pysrc.review.utils import ids_to_sent


def load_pubmedtop50(parts):
    return [
        pd.read_csv(f"{cfg.data_path}/pubmedtop50_{part}.csv", index_col='id')
        for part in parts
    ]


def preprocess_paper(text, max_len, tokenizer):
    sents = [[tokenizer.artBOS.tkn] + tokenizer.tokenize(sent) + [tokenizer.artEOS.tkn]
             for sent in sent_tokenize(text)]
    ids, segments, segment_signature = [], [], 0
    n_setns = 0
    for s in sents:
        if len(ids) + len(s) <= max_len:
            n_setns += 1
            ids.extend(tokenizer.convert_tokens_to_ids(s))
            segments.extend([segment_signature] * len(s))
            segment_signature = (segment_signature + 1) % 2
        else:
            break
    mask = [1] * len(ids)

    pad_len = max(0, max_len - len(ids))
    ids += [tokenizer.PAD.idx] * pad_len
    mask += [0] * pad_len
    segments += [segment_signature] * pad_len

    return ids, mask, segments, n_setns


def standardize(text):
    """ Standardize text span
    """

    text = unidecode(text)
    text = text.replace('--', '-')
    text = text.replace(';', ".")
    text = text.replace('...', ".")
    text = text.replace('..', ".")
    text = text.replace("'''", "'")
    text = text.replace("''", "'")
    text = text.replace("```", "`")
    text = text.replace("``", "`")
    text = text.strip()
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
    text = re.sub(r'\([^)]*\)', '', text)

    return text


class TrainDataset(Dataset):
    """ Custom Train Dataset
    """

    def __init__(self, dataframe, tokenizer, article_len):
        self.df = dataframe
        self.n_examples = len(dataframe)
        self.tokenizer = tokenizer
        self.article_len = article_len

    def __getitem__(self, idx):

        ex = self.df.iloc[idx]
        paper = standardize(ex.paper_top50)
        # abstract = standardize(ex.abstract)
        try:
            gold_ids = [int(e) for e in ex.gold_ids_top6.strip("[]").split(',')]
        except Exception:
            gold_ids = []

        article_ids, article_mask, article_segment, n_setns = \
            preprocess_paper(paper, self.article_len, self.tokenizer)

        # form target
        target = [(1 if i in gold_ids else 0) for i in range(n_setns)]

        return article_ids, article_mask, article_segment, target

    def __len__(self):
        return self.n_examples


class EvalDataset(Dataset):
    """ Custom Valid/Test Dataset
    """

    def __init__(self, dataframe, tokenizer, article_len):
        self.df = dataframe
        self.n_examples = len(dataframe)
        self.tokenizer = tokenizer
        self.article_len = article_len

    def __getitem__(self, idx):
        ex = self.df.iloc[idx]
        paper = standardize(ex.paper_top50)
        abstract = standardize(ex.abstract)

        gold_ids = [int(e) for e in ex.gold_ids_top6.strip("[]").split(',')]
        gold_sents = self.extract_gold_sents(paper, gold_ids)

        article_ids, article_mask, article_segment, n_setns = \
            preprocess_paper(paper, self.article_len, self.tokenizer)

        # cut gold ids
        gold_ids = [e for e in gold_ids if e < n_setns]
        gold_text = ' '.join(gold_sents[:len(gold_ids)])
        if not gold_text:
            gold_text = ' '.join(gold_sents)

        return paper, article_ids, article_mask, article_segment, gold_text, abstract

    @staticmethod
    def extract_gold_sents(paper, gold_ids):
        paper = sent_tokenize(paper)
        gold_sents = [sent for i, sent in enumerate(paper) if i in gold_ids]
        return gold_sents

    def __len__(self):
        return self.n_examples


def train_collate_fn(batch_data):
    """ Function to pull batch for train

    :param batch_data: list of `TrainDataset` Examples
    :return:
        one batch of data
    """
    data0, data1, data2, data3 = list(zip(*batch_data))

    return torch.tensor(data0, dtype=torch.long), \
           torch.tensor(data1, dtype=torch.long), \
           torch.tensor(data2, dtype=torch.long), \
           [torch.tensor(e, dtype=torch.float) for e in data3]


def eval_collate_fn(batch_data):
    """ Function to pull batch for valid/test

    :param batch_data: list of `EvalDataset` Examples
    :return:
        one batch of data
    """

    return [torch.tensor(data_prt, dtype=torch.long) if not isinstance(data_prt[0], str)
            else data_prt for data_prt in list(zip(*batch_data))]


def load_data(dataset_type, parts):
    assert dataset_type in ['pubmed']
    return load_pubmedtop50(parts)


def create_ddp_loader(dataset, batch_size, collate_fn):
    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        sampler=DistributedSampler(
            dataset=dataset, num_replicas=distrib.get_world_size(), rank=distrib.get_rank()
        ),
        num_workers=cfg.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn,
    )


def create_loader(dataset, batch_size, collate_fn):
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, collate_fn=collate_fn, num_workers=cfg.num_workers
    )


if __name__ == "__main__":
    """
    some tests
    """

    data_sz = 'small'
    btch_sz = 1
    dstype = 'cnndm'
    data_parts = load_data(dstype, data_sz)

    train_dl, valid_dl, test_dl = [
        create_loader(data_part, btch_sz, collate_fn) for data_part, collate_fn
        in zip(data_parts, [train_collate_fn, eval_collate_fn, eval_collate_fn])
    ]

    print(len(train_dl.dataset))
    print(len(valid_dl.dataset))
    print(len(test_dl.dataset))

    for btch in train_dl:
        art_ids = btch[0]
        sum_ids = btch[3]

        print(ids_to_sent(art_ids[0]))
        print('----------------')
        print(ids_to_sent(sum_ids[0]))
        break
