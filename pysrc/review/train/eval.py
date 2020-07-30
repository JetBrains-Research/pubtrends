import numpy as np
import pandas as pd
import torch
import torch.distributed as distrib
from nltk.tokenize import sent_tokenize
from rouge import Rouge
from tqdm import tqdm

import pysrc.review.config as cfg

ROUGE_TYPES = ['r1', 'r2', 'rl', 'rmean']


def get_rougevalues(pred_sent, target_sent):
    try:
        rouges = Rouge().get_scores(pred_sent, target_sent)[0]
        rouges = [rouges[f'rouge-{x}']["f"] for x in ('1', '2', 'l')]
        rouges += [np.mean(rouges)]
    except Exception:
        return None
    return np.array(rouges) * 100


def distribute(rouges_values, device):
    tch_rouge = torch.from_numpy(rouges_values).float().to(device)
    distrib.all_reduce(tch_rouge)
    return tch_rouge.cpu().numpy()


def evaluate(model, dataloader, device, rank, writer, distributed, epoch=0, save_filename=None, to_write=True):
    rouges_values = np.zeros(4)  # r1, r2, rl, rmean
    arouges_values = np.zeros(4)  # r1, r2, rl, rmean
    model.eval()
    model_ref = model.module if distributed else model

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, disable=rank != 0)
    print(pbar, len(dataloader))

    for batch in pbar:

        id, papers, article_ids, article_mask, article_segment, gold_texts, abstracts = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

        with torch.no_grad():
            # get draft | torch.Size([batch_size, summary_len])
            draft_ids = model_ref.evaluate(article_ids, article_mask, article_segment)

        # transform ids to sents
        pred_sents = []
        for p, pred_idx in zip(papers, draft_ids):
            p = sent_tokenize(p)
            pred = [sent for i, sent in enumerate(p) if i in pred_idx]
            pred_sents.append(' '.join(pred))

        print("pred:\n", pred_sents[0])
        print("\n\n")
        #         print("target:\n", gold_texts[0])

        # list of np.shape(4)
        current_rouges = [get_rougevalues(ps, ts) for ps, ts in zip(pred_sents, gold_texts)]
        abstract_rouges = [get_rougevalues(ps, ts) for ps, ts in zip(pred_sents, abstracts)]

        current_rouges = sum(filter(lambda r: r is not None, current_rouges))
        abstract_rouges = sum(abstract_rouges)

        rouges_values += current_rouges
        arouges_values += abstract_rouges

        pbar.set_description(' '.join([
            f"{rouge_type}:{val1 / len(article_ids):.2f}({val2 / len(article_ids):.2f})"
            for rouge_type, val1, val2
            in zip(ROUGE_TYPES, current_rouges, abstract_rouges)
        ]))

        for rouge_type, val1, val2 in zip(ROUGE_TYPES, current_rouges / len(article_ids),
                                          abstract_rouges / len(article_ids)):
            writer.add_scalar(f'Eval/{rouge_type}', val1, writer.eval_step)
            writer.add_scalar(f'Eval/a{rouge_type}', val2, writer.eval_step)
        writer.eval_step += 1

    # write and log overall info
    if distributed:
        rouges_values = distribute(rouges_values, device)
    for rouge_type, val in zip(ROUGE_TYPES, rouges_values):
        cfg.logger.log(f"{rouge_type}: {val / len(dataloader.dataset):.2f}", is_print=rank == 0)
        writer.add_scalar(f"Eval_Overall/{rouge_type}", val / len(dataloader.dataset), epoch)

    # save model if need
    if rouges_values[-1] > model_ref.rouge_mean and save_filename is not None and rank == 0:
        model_ref.rouges_values = rouges_values / len(dataloader.dataset)
        model_ref.save(save_filename)

    return model, writer


def evaluate_topic(model, dataloader, device, rank, writer, save_root):
    rouges_values = np.zeros(4)  # r1, r2, rl, rmean
    arouges_values = np.zeros(4)  # r1, r2, rl, rmean
    model.eval()
    model_ref = model

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, disable=rank != 0)

    pred_summ = []
    for batch in pbar:
        papers, article_ids, article_mask, article_segment, gold_texts, abstracts = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

        with torch.no_grad():
            # get draft | torch.Size([batch_size, summary_len])
            draft_ids = model_ref.evaluate(article_ids, article_mask, article_segment)

        # transform ids to sents
        pred_sents = []
        for p, pred_idx in zip(papers, draft_ids):
            p = sent_tokenize(p)
            pred = [sent for i, sent in enumerate(p) if i in pred_idx]
            pred_sents.append(' '.join(pred))

        print(len(pred_sents), pred_sents)

        for pred_text in pred_sents:
            pred_summ.append(pred_text)

    #         print("pred:\n", pred_sents[0])
    #         print("\n\n")

    pred_csv = pd.DataFrame(data={'pred_sents': pred_summ})
    pred_csv['id'] = range(len(pred_csv))
    pred_csv.to_csv(f"{save_root}/results.csv", index=False)

    return model, writer


if __name__ == "__main__":
    """
    some test's
    """
    pass
