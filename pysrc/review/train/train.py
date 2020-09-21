import math

import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

import pysrc.review.config as cfg
from pysrc.review.utils import get_enc_lr, get_dec_lr


def distribute(loss, device):
    loss_tch = torch.tensor([loss]).float().to(device)
    distrib.all_reduce(loss_tch)
    return loss_tch.item()

def backward_step(loss: torch.Tensor, optimizer: Optimizer, model: nn.Module, clip: float, amp_enabled: int):
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    return total_norm

def train(model, dataloader, optimizer, scheduler, criter, device, rank, writer, distributed):

    # draft, refine
    model.train()
    model_ref = model.module if distributed else model

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=rank != 0)
    for idx_batch, batch in pbar:

        input_ids, input_mask, input_segment, target_ids = \
            [(x.to(device) if isinstance(x, torch.Tensor) else x) for x in batch]
        target_ids = torch.cat(target_ids).to(device)

        # forward pass
        draft_logprobs = model(
            input_ids, input_mask, input_segment,
        )

        # loss
        loss = criter(
            draft_logprobs,
            target_ids,
        )

        # backward
        grad_norm = backward_step(loss, optimizer, model, optimizer.clip_value, amp_enabled=cfg.amp_enabled)
        grad_norm = 0 if (math.isinf(grad_norm) or math.isnan(grad_norm)) else grad_norm

        # record a loss value
        # loss_val += loss.item() * len(input_ids)
        pbar.set_description(f"loss:{loss.item():.2f}")
        writer.add_scalar(f"Train/loss", loss.item(), writer.train_step)
        writer.add_scalar("Train/grad_norm", grad_norm, writer.train_step)
        writer.add_scalar("Train/lr_enc", get_enc_lr(optimizer), writer.train_step)
        writer.add_scalar("Train/lr_dec", get_dec_lr(optimizer), writer.train_step)
        writer.train_step += 1

        # make a gradient step
        if (idx_batch + 1) % optimizer.accumulation_interval == 0 or (idx_batch + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

    # overall loss per epoch
    # if distributed:
    #     loss_val = distribute(loss_val, device)
    # cfg.logger.log(f"mean loss: {loss_val / len(dataloader.dataset):.4f}", is_print=rank == 0)

    # save model, just in case
    if rank == 0:
        model_ref.save('temp')

    return model, optimizer, scheduler, writer


if __name__ == "__main__":
    """
    some test's
    """

    # torch.cuda.empty_cache()

    tnz = torch.empty(3, 15).random_(0, 4)
    print(tnz)
