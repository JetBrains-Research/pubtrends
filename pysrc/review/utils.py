import argparse
import os
import random
import re

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import pysrc.review.config as cfg


class DummyWriter:

    def __init__(self):
        self.log_dir = cfg.tb_logdir

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def close(self):
        pass


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


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
        self.const = np.sqrt(2 / np.pi)

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self.const * (x + 0.044715 * torch.pow(x, 3))))


class DecoderLayer(nn.Module):
    """
    Represents one decoder layer of the transformer decoder
    """

    def __init__(self, d_hidden, n_heads):
        """
        init for decoder layer
        :param d_hidden: hidden size | int % n_heads == 0
        :param n_heads: number of multi-head attention heads | int
        """
        super(DecoderLayer, self).__init__()

        self.dec_mha, self.encdec_mha = \
            [MultiHeadAttention(d_hidden, n_heads) for _ in range(2)]
        self.ff = FeedForward(d_hidden, cfg.d_ff, d_hidden)
        self.dec_layernorm, self.encdec_layernorm = [nn.LayerNorm(d_hidden, eps=1e-6) for _ in range(2)]
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, inpt):
        """
        forward pass for decoder layer
        :param inpt: (x, enc_output, dec_mask, encdec_mask)
            x: decoder input | torch.Size([batch_size, dec_len, d_hidden])
            enc_output: encoder output | torch.Size([batch_size, enc_len, d_hidden])
            dec_mask: torch.Size([batch_size, 1, dec_len, dec_len])
            encdec_mask: torch.Size([batch_size, 1, 1, enc_len])
        """

        x, enc_output, dec_mask, encdec_mask = inpt
        # layer normalization before decoder self attention | torch.Size([batch_size, dec_len, d_hidden])
        x_norm = self.dec_layernorm(x)
        # masked multi-head attention | torch.size([batch_size, dec_len, d_hidden])
        y = self.dec_mha(x_norm, x_norm, x_norm, dec_mask)
        # dropout and residual after self-attention | torch.size([batch_size, dec_len, d_hidden])
        x = self.dropout(y) + x
        # layer normalization before encoder-decoder attention | torch.size([batch_size, dec_len, d_hidden])
        x_norm = self.encdec_layernorm(x)
        # multi-head encoder-decoder attention | torch.Size([batch_size, dec_len, d_hidden])
        y = self.encdec_mha(x_norm, enc_output, enc_output, encdec_mask)
        # dropout and residual after encoder-decoder attention | torch.size([batch_size, dec_len, d_hidden])
        x = self.dropout(y) + x
        # feed-forward | torch.size([batch_size, dec_len, d_hidden])
        x = self.ff(x)

        return x, enc_output, dec_mask, encdec_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention
    """

    def __init__(self, d_hidden, n_heads):
        """
        Multi-head init
        :param d_hidden: Size of last dimension of keys/values/queries | int % n_heads == 0
        :param n_heads: Number of attention heads | int
        """
        super(MultiHeadAttention, self).__init__()

        self.query_scale = np.sqrt(d_hidden / n_heads)
        self.n_heads = n_heads
        self.q_linear, self.k_linear, self.v_linear = [nn.Linear(d_hidden, d_hidden) for _ in range(3)]
        self.output_linear = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

        assert d_hidden % n_heads == 0, 'MH hidden dim must be divisible by n_heads'

    def split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        :param x: torch.Size([batch_size, len, d_hidden])
        :return: torch.Size([batch_size, n_heads, len, d_hidden // n_heads])
        """

        return x.view(x.size(0), x.size(1), self.n_heads, x.size(2) // self.n_heads).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        :param x: torch.Size([batch_size, n_heads, len, dim // n_heads])
        :return: torch.Size([batch_size, len, dim])
        """

        return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), x.size(3) * self.n_heads)

    def forward(self, queries, keys, values, mask=None):
        """
        forward pass for Multi-Head Attention
        :param queries: torch.size([batch_size, q_len, d_hidden])
        :param keys: torch.size([batch_size, kv_len, d_hidden])
        :param values: torch.size([batch_size, kv_len, d_hidden])
        :param mask: torch.size([batch_size, 1, q_len, kv_len])
        :return:
        """

        # linear for each component | torch.size([batch_size, len, d_hidden])
        queries = self.q_linear(queries)
        keys = self.k_linear(keys)
        values = self.v_linear(values)
        # Split into multiple heads | torch.size([batch_size, n_heads, len, d_hidden // n_heads])
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        # Combine queries and keys | torch.size([batch_size, n_heads, q_len, kv_len])
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.query_scale
        # masking | torch.size([batch_size, n_heads, q_len, kv_len])
        if mask is not None:
            logits = logits.masked_fill(mask, -1e4 if cfg.amp_enabled else -1e20)
        # Convert to probabilities | torch.size([batch_size, n_heads, q_len, kv_len])
        weights = F.softmax(logits, dim=-1)
        # Dropout | torch.size([batch_size, n_heads, q_len, kv_len])
        weights = self.dropout(weights)
        # Combine with values | torch.size([batch_size, n_heads, q_len, d_hidden // n_heads])
        contexts = torch.matmul(weights, values)
        # Merge heads | torch.size([batch_size, q_len, d_hidden])
        contexts = self.merge_heads(contexts)
        # Linear to get output | torch.size([batch_size, q_len, d_output])
        outputs = self.output_linear(contexts)

        return outputs


class FeedForward(nn.Module):
    """
    Position wise Feed-Forward
    """

    def __init__(self, d_input, d_hidden, d_output):
        """
        Init for FeedForward net
        :param d_input: input dimension | int
        :param d_hidden: hidden dimension | int
        :param d_output: output dimension | int
        """
        super(FeedForward, self).__init__()

        assert d_input == d_output, 'Incorrect in/out sizes!'

        self.layers = nn.Sequential(
            nn.LayerNorm(d_input, eps=1e-6),
            nn.Linear(d_input, d_hidden),
            GeLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_hidden, d_output),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        """
        forward pass for FeedForward
        :param x: torch.size([batch_size, len, d_input])
        :return: torch.size([batch_size, len, d_output])
        """

        return self.layers(x) + x


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
