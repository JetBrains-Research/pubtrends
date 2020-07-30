import argparse
from argparse import RawTextHelpFormatter

import torch
from tensorboardX import SummaryWriter
from torch.multiprocessing.spawn import spawn
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW

import pysrc.review.config as cfg
from pysrc.review.model import Summarizer
from pysrc.review.train.data import load_data, train_collate_fn, \
    eval_collate_fn, create_ddp_loader, create_loader, TrainDataset, EvalDataset
from pysrc.review.train.eval import evaluate, evaluate_topic
from pysrc.review.train.scheduler import NoamScheduler
from pysrc.review.train.train import train
from pysrc.review.utils import count_parameters, DummyWriter
from pysrc.review.utils import str2bool


def get_args():

    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)

    parser.add_argument("-lr_decoder", dest="lr_decoder", default=0.001, type=float,
                        help="initial decoder learning rate")
    parser.add_argument("-lr_encoder", dest="lr_encoder", default=0.00001, type=float,
                        help="initial encoder learning rate")
    parser.add_argument("-dataset", dest="dataset", choices=['pubmed'], default='pubmed',
                        help="pubmed")
    parser.add_argument("-article_len", dest="article_len", default=512, type=int,
                        help="article len")
    parser.add_argument("-epochs", dest="epochs", default=1, type=int,
                        help="number of training epochs")
    parser.add_argument("-warmup", dest="warmup", default=50, type=int,
                        help="number of warmup batches")
    parser.add_argument("-weight_decay", dest="weight_decay", default=0.005, type=float,
                        help="decay weight for regularize")
    parser.add_argument("-clip_value", dest="clip_value", default=1.0, type=float,
                        help="clip value")
    parser.add_argument("-batch_size", dest="batch_size", default=1, type=int,
                        help="size of training batch")
    parser.add_argument("-accumulation_interval", dest="accumulation_interval", default=1, type=int,
                        help="accumulation interval")
    parser.add_argument("-valid_interval", dest="valid_interval", default=1, type=int,
                        help="validation interval")
    parser.add_argument("-model_type", dest="model_type", choices=['bert', 'roberta'],
                        default='bert', help="model type")
    parser.add_argument("-distributed", dest="distributed", default=False, type=str2bool,
                        help="distributed on/off")
    parser.add_argument("-froze_strategy", dest="froze_strategy", choices=['froze_all',
                        'unfroze_last4', 'unfroze_all'], default='froze_all', help="froze backbone")
    parser.add_argument("-bert_strategy", dest="bert_strategy", choices=['lastlayer',
                        'last4layers_cat'], default='lastlayer', help="bert strategy")
    parser.add_argument("-mode", dest="mode", choices=['trainval', 'test', 'test_topic'], default="trainval",
                        type=str, help="model mode")
    parser.add_argument("-save_filename", dest="save_filename", default='bertas', type=str,
                        help="file to save net weights with .pth extension")
    parser.add_argument("-tb_tag", dest="tb_tag", default='last_experiment', type=str,
                        help="tensorboard comment to differ from other experiments")

    return parser.parse_args()


def load_model(model_type, froze_strategy, rank, article_len):
    model = Summarizer(model_type, article_len)
    model.expand_posembs_ifneed()
#     model.load('temp')
    model.froze_backbone(froze_strategy)
    model.unfroze_head()
    if rank == 0:
        cfg.logger.log(f"Model trainable parameters: {count_parameters(model)}")
    return model


def get_dataloaders(raw_data, batch_size,
                    article_len, tokenizer, ddp):
    datasets = [
        DsClass(data_part, tokenizer, article_len) for DsClass, data_part
        in zip([TrainDataset, EvalDataset], raw_data)
    ]
    dl_func = create_ddp_loader if ddp else create_loader
    return [
        dl_func(dataset, batch_size, collate_fn) for dataset, collate_fn
        in zip(datasets, [train_collate_fn, eval_collate_fn])
    ]


def get_tools(model, enc_lr, dec_lr, warmup,
              weight_decay, clip_value,
              accumulation_interval):

    enc_parameters = [
        param for name, param in model.named_parameters()
        if param.requires_grad and name.startswith('bert.')
    ]
    dec_parameters = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith('bert.')
    ]
    optimizer = AdamW([
        {'params': enc_parameters, 'lr': enc_lr},
        {'params': dec_parameters, 'lr': dec_lr},
    ], weight_decay=weight_decay)
    optimizer.clip_value = clip_value
    optimizer.accumulation_interval = accumulation_interval

    scheduler = NoamScheduler(optimizer, warmup=warmup)
    criter = BCEWithLogitsLoss(reduction='mean')

    return optimizer, scheduler, criter


# def setup_multi_gpu(model, optimizer, rank, size):
#     cfg.logger.log('Setup distributed settings...')
#     distrib_config = DistributedConfig(local_rank=rank, size=size, amp_enabled=cfg.amp_enabled)
#     setup_distributed(distrib_config)
#     device = choose_device(local_rank=rank)
#     model = model.to(device)
#     model, optimizer = setup_apex_if_enabled(model, optimizer, config=distrib_config)
#     model = setup_distrib_if_enabled(model, config=distrib_config)
#     return model, device, optimizer


def setup_single_gpu(model):
    cfg.logger.log('Setup single-device settings...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def main(rank=0, size=1, args=None, data=None):

    if rank == 0:
        writer = SummaryWriter(log_dir=f"{cfg.tb_logdir}/{args.tb_tag}")
    else:
        writer = DummyWriter()
    writer.add_text("Hparams", '<br />'.join([f"{k}: {v}" for k, v in args.__dict__.items()]))
    writer.train_step, writer.eval_step = 0, 0

    cfg.logger.log('Load model...')
    model = load_model(args.model_type, args.froze_strategy, rank, args.article_len)

    if args.mode == 'trainval':

        cfg.logger.log('Load training tools...')
        optimizer, scheduler, criter = \
            get_tools(model, args.lr_encoder, args.lr_decoder, args.warmup, args.weight_decay,
                      args.clip_value, args.accumulation_interval)

        if args.distributed:
            pass
#             model, device, optimizer = setup_multi_gpu(model, optimizer, rank, size)
        else:
            model, device = setup_single_gpu(model)

        cfg.logger.log('Create dataloaders...')
        model_ref = model.module if args.distributed else model
        train_loader, valid_loader = \
            get_dataloaders(data, args.batch_size, model_ref.article_len,
                            model_ref.tokenizer, ddp=args.distributed)

        for epoch in range(1, args.epochs + 1):

            # for correct distrib sampling
#             if isinstance(train_loader, DistributedSampler):
#                 train_loader.set_epoch(epoch)

            cfg.logger.log(f"{epoch} epoch training...", is_print=rank == 0)
            model, optimizer, scheduler, writer = train(
                model, train_loader, optimizer, scheduler,
                criter, device, rank, writer, args.distributed
            )

            if not epoch % args.valid_interval:
                cfg.logger.log(f"{epoch} epoch validation...", is_print=rank == 0)
                model, writer = evaluate(
                    model, valid_loader, device, rank, writer, args.distributed,
                    epoch, args.save_filename, to_write=epoch > args.epochs // 2,
                )

    elif args.mode == 'test':

        if args.distributed:
            pass
#             dummy_optimizer = AdamW(model.parameters())
#             model, device, optimizer = setup_multi_gpu(model, dummy_optimizer, rank, size)
        else:
            model, device = setup_single_gpu(model)

        model_ref = model.module if args.distributed else model
        _, valid_loader = \
            get_dataloaders(data, args.batch_size, model_ref.article_len,
                            model_ref.tokenizer, ddp=args.distributed)

        model, writer = evaluate(
            model, valid_loader, device, rank, writer, args.distributed,
            1, args.save_filename, to_write=True,
        )
        
    elif args.mode == 'test_topic':
        model, device = setup_single_gpu(model)
        
        model_ref = model.module if args.distributed else model
            
        _, valid_loader = \
            get_dataloaders(data, args.batch_size, model_ref.article_len,
                            model_ref.tokenizer, ddp=args.distributed)

        model, writer = evaluate_topic(
            model, valid_loader, device, rank, writer, save_root=cfg.predicted_path)   

    else:
        raise Exception(f"wrong value for argument -mode: {args.mode}")

    writer = cfg.logger.write(writer)
    writer.close()


if __name__ == '__main__':

    args = get_args()

    if args.mode == 'test_topic':
        parts = ['train', 'test_topic']
    else:
        parts = ['train', 'valid']
    data = load_data(args.dataset, parts)
    
    for prt, ds in zip(['train', 'valid', 'test'], data):
        cfg.logger.log(f"{prt} examples: {len(ds)}")

    cfg.logger.log('Main starting point...')
    if args.distributed:
        spawn(main, args=(cfg.n_devices, args, data), nprocs=cfg.n_devices)
    else:
        main(args=args, data=data)
