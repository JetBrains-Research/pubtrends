import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import psutil
import re
import random
import argparse

import pysrc.review.config as cfg


def setup_single_gpu(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_weights(net):
    ans = 0
    for param in net.parameters():
        if param.requires_grad:
            ans += torch.sum(param ** 2).item()
    return np.sqrt(ans)


def compute_weights_grad(net):
    ans = 0
    for param in net.parameters():
        if param.requires_grad:
            ans += torch.sum(param.grad ** 2).item()
    return np.sqrt(ans)


def count_parameters(model):
    """
    count parameters to train
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_future_mask(size, device=torch.device('cpu')):
    """
    Get an upper triangular attention mask to avoid using the subsequent info
    :param size: mask size | int
    :param device:
    :return: torch.Size([1, size, size])
    """
    return torch.ones(1, size, size, dtype=torch.uint8, device=device).triu(diagonal=1)


def get_ids_mask(ids, PAD):
    """
    :param ids: torch.Size([batch_size, len])
    :return: torch.Size([batch_size, 1, 1, len])
    """
    return (ids == PAD.idx).unsqueeze(-2).unsqueeze(-2)


def get_pos_encoding(seq_len, d_hidden):
    """
    Generates positional encoding for transformer
    :param seq_len: sequence length |  int
    :param d_hidden: hidden dim | int % 2 == 0
    :return: torch.Size([1, seq_len, d_hidden])
    """
    assert d_hidden % 2 == 0, 'incorrect d_hidden num'

    ans = torch.arange(1, d_hidden + 1, dtype=torch.float) / d_hidden
    ans = torch.pow(10000, ans)
    ans = ans.expand(seq_len, d_hidden)

    pos = torch.arange(1, seq_len + 1, dtype=torch.float).unsqueeze(1).expand(seq_len, d_hidden)
    ans = torch.div(pos, ans)

    ans[:, 0::2].sin_()
    ans[:, 1::2].cos_()

    return ans.unsqueeze(0)


def ids_to_sent(ids, model):
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
    summary = model.tokenizer.decode(ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
    for spt in [model.sumBOS.tkn, model.sumEOS.tkn, model.sumEOA.tkn]:
        summary = summary.replace(spt, ' ')
    summary = re.sub(r'\s([?.!"](?:\s|$))', r'\1', summary)
    summary = ' '.join(summary.split())
    return summary


def get_bash_command_out(command):
    aux = os.popen(command)
    ans = aux.read().strip()
    aux.close()
    return ans


def gpu_info(text):
    gpu_util, mem_used, mem_total = [
        int(get_bash_command_out(x)) for x in (
            'nvidia-smi --query-gpu=utilization.gpu --format=csv | tail -1 | cut -d " " -f 1',
            'nvidia-smi --query-gpu=memory.used --format=csv | tail -1 | cut -d " " -f 1',
            'nvidia-smi --query-gpu=memory.total --format=csv | tail -1 | cut -d " " -f 1',
        )
    ]
    print(f'{text} | GPU: {gpu_util} %  {mem_used}/{mem_total} MB')


def cpu_info(text):
    cpu_util = round(psutil.cpu_percent())
    memory_info = psutil.virtual_memory()
    MEMORY_COEF = 2. ** 20
    used = round(memory_info.used / MEMORY_COEF)
    total = round(memory_info.total / MEMORY_COEF)
    print(f'{text} | CPU: {cpu_util} %  {used}/{total} MB')


def w_report(model):
    layers_norms = {name: torch.norm(p).item() for name, p in model.named_parameters() if p.requires_grad}
    max_layer = max(layers_norms, key=layers_norms.get)
    min_layer = min(layers_norms, key=layers_norms.get)
    mean_norm = np.mean(list(layers_norms.values()))
    return f"weights norm: " \
           f"mean: {mean_norm:.2f}|" \
           f"max: {layers_norms[max_layer]:.2f} ({max_layer})|" \
           f"min: {layers_norms[min_layer]:.2f} ({min_layer})"


def wgrad_report(model):
    layers_gradnorms = {name: torch.norm(p.grad).item() for name, p in model.named_parameters() if p.requires_grad}
    max_layer = max(layers_gradnorms, key=layers_gradnorms.get)
    min_layer = min(layers_gradnorms, key=layers_gradnorms.get)
    mean_norm = np.mean(list(layers_gradnorms.values()))
    return f"weights grad norm: " \
           f"mean: {mean_norm:.2f}|" \
           f"max: {layers_gradnorms[max_layer]:.2f} ({max_layer})|" \
           f"min: {layers_gradnorms[min_layer]:.2f} ({min_layer})"


def get_enc_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_dec_lr(optimizer):
    return optimizer.param_groups[1]['lr']


def get_gradnorm(model):
    norms = [torch.norm(p.grad).item() for p in model.parameters() if p.requires_grad]
    return np.mean(norms)


if __name__ == "__main__":
    """ some test's"""

    l = 8
    dh = 8
    pe = get_pos_encoding(l, dh)
    print(pe)