import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_single_gpu(model):
    logging.info('Setup single-device settings...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def get_enc_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_dec_lr(optimizer):
    return optimizer.param_groups[1]['lr']


def get_gradnorm(model):
    norms = [torch.norm(p.grad).item() for p in model.parameters() if p.requires_grad]
    return np.mean(norms)
