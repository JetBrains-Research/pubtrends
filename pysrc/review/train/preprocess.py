import argparse
import glob
import os
import re
from argparse import RawTextHelpFormatter
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from rouge import Rouge
from torch.multiprocessing.spawn import spawn
from tqdm import tqdm
from transformers import BertTokenizer
from unidecode import unidecode

import pysrc.review.config as cfg
from pysrc.review.train.ngram_utils import _get_word_ngrams

REPLACE_SYMBOLS = {
    '—': '-',
    '–': '-',
    '―': '-',
    '…': '...',
    '´´': "´",
    '´´´': "´´",
    "''": "'",
    "'''": "'",
    "``": "`",
    "```": "`",
    ":": " : ",
}
ROUGE_METER = Rouge()
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def get_args():
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)

    parser.add_argument("-mode", dest="mode", default="preprocess", choices=['preprocess', 'merge'],
                        help="mode")
    parser.add_argument("-root2data", dest="root2data", default="pudmed", type=str,
                        help="path to raw pudmed")
    parser.add_argument("-split", dest="split", default="train", choices=['train', 'dev', 'test'],
                        help="split")
    parser.add_argument("-save_prefix", dest="save_prefix", default="pudmedtop50", type=str,
                        help="save prefix")
    parser.add_argument("-n_procs", dest="n_procs", default="1", type=int,
                        help="number of processes")

    return parser.parse_args()


def get_files(root, split, part):
    assert split in ['train', 'dev', 'test']
    assert part in ['fragments', 'abstracts']
    #     files = glob.glob(str(root / split / f'{part}*.txt'))
    files = glob.glob(str(root / f'{part}_{split}_*.txt'))
    files = sorted(files, key=lambda file: int(re.search(r"[0-9]+", file).group(0)))
    return files


def parse(file):
    with open(file, 'r') as f:
        data = defaultdict(list)
        for line in f:
            name, _, text = extract_info(line)
            data[name].append(text)
    return data


def extract_info(line):
    info, text = line.split("\t", 1)
    name, idx = info.rsplit('/', 1)
    return name, idx, text


def merge(papers, abstracts):
    return {k: (papers[k], abstracts[k]) for k in abstracts.keys() if k in papers}


def parse_and_merge(papers_file, abstracts_file):
    return merge(parse(papers_file), parse(abstracts_file))


def get_rouge(sent1, sent2):
    rouges = ROUGE_METER.get_scores(sent1, sent2)[0]
    rouges = [rouges[f'rouge-{x}']["f"] for x in ('1', '2', 'l')]
    return np.mean(rouges) * 100


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size, stop=True):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    doc_sent_list = [sent.split() for sent in doc_sent_list]
    abstract_sent_list = [sent.split() for sent in abstract_sent_list]
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge if stop else 0.0
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def preprocess_part(papers_file, abstracts_file, is_tqdm):
    data = parse_and_merge(papers_file, abstracts_file)
    preprocessed_papers, preprocessed_abstracts, preprocessed_gold = [], [], []

    for _, (paper, abstract) in tqdm(data.items(), total=len(data), leave=True, disable=not is_tqdm):
        # prepare paper
        paper = parse_sents(paper)
        paper = standardize(paper)
        # prepare abstract
        abstract = parse_sents(abstract)
        abstract = standardize(abstract)

        if len(paper) < 30:
            continue

        # extract gold sentences
        if len(paper) > 50:
            cheating_ids = greedy_selection(paper, abstract, 50, stop=False)
            cheating_source = [paper[i] for i in cheating_ids]
            paper = cheating_source

        gold_ids = greedy_selection(paper, abstract, 6)

        if not gold_ids:
            continue

        # to text
        paper = ' '.join(paper)
        abstract = ' '.join(abstract)

        """
        print("paper:\n=======\n", paper)
        print("abstract:\n=======\n", abstract)
        print("n tokens:\n=======\n", len(TOKENIZER.encode(paper)))
        print(gold_ids)
        print('\n\n')
        """

        # add to preprocessed
        preprocessed_papers.append(paper)
        preprocessed_abstracts.append(abstract)
        preprocessed_gold.append(gold_ids)

    del data
    return preprocessed_papers, preprocessed_abstracts, preprocessed_gold


def parse_sents(data):
    sents = sum([sent_tokenize(text) for text in data], [])
    sents = list(filter(lambda x: len(x) > 3, sents))
    return sents


def standardize(text):
    def sent_standardize(sent):
        sent = unidecode(sent)
        sent = re.sub(r"\[(\[xref\])(,\[xref\])*\]", " ", sent)  # delete [[xref],...]
        sent = re.sub(r"\((\[xref\])(; \[xref\])*\)", " ", sent)  # delete ([xref]; ...)
        sent = re.sub(r"\[xref\]", " ", sent)  # delete [xref]
        sent = re.sub(r"\[\[xref\]\]", " ", sent)  # delete [[xref]]
        for k, v in REPLACE_SYMBOLS.items():
            sent = sent.replace(k, v)
        return sent.strip()

    text = [sent_standardize(sent) for sent in text]
    return list(filter(lambda x: len(x) > 3, text))


def main(rank=0, size=1, args=None, papers_files=None, abstracts_files=None):
    split = args.split
    save_prefix = args.save_prefix
    paper_file = papers_files[rank]
    abstract_file = abstracts_files[rank]
    save_root = Path(cfg.data_path)
    save_filename = f"{save_prefix}_{split}_{rank}.csv"

    print(f"Processing {rank + 1} part out of {size}...")
    papers, abstracts, gold = preprocess_part(paper_file, abstract_file, is_tqdm=(rank == 0))
    assert len(papers) == len(abstracts)
    print(f"Obtained {len(papers)} new exemplars in {rank + 1} part...")

    d = {'paper_top50': papers, 'abstract': abstracts, 'gold_ids_top6': gold}
    df = pd.DataFrame(data=d)

    df.to_csv(save_root / save_filename, index=False)


def merge_csvs(split, save_root, save_prefix):
    all_filenames = [i for i in glob.glob(f'data/{save_prefix}_{split}_*.csv')]
    print(all_filenames)
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv['id'] = range(len(combined_csv))
    combined_csv.to_csv(f"data/{save_prefix}_{split}.csv", index=False)

    for file_path in all_filenames:
        os.remove(file_path)


if __name__ == '__main__':
    args = get_args()

    split = args.split
    save_prefix = args.save_prefix
    save_root = Path("models/summarization/data")
    root2data = Path(args.root2data)
    n_procs = args.n_procs

    if args.mode == "merge":
        merge_csvs(split, save_root, save_prefix)

    elif args.mode == "preprocess":
        papers_files = get_files(root2data, split, 'fragments')
        abstracts_files = get_files(root2data, split, 'abstracts')

        #         assert n_procs >= len(papers_files)

        spawn(main, args=(n_procs, args, papers_files, abstracts_files), nprocs=min(n_procs, len(papers_files)))
