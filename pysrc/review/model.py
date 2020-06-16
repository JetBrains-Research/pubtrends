import logging
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from transformers import BertTokenizer, RobertaTokenizer

import pysrc.review.config as cfg
from pysrc.review.text import convert_token_to_id

SpecToken = namedtuple('SpecToken', ['tkn', 'idx'])

# Deployment and development
MODEL_PATHS = ['/model', os.path.expanduser('~/.pubtrends/model')]

logger = logging.getLogger(__name__)


def load_model(model_type, froze_strategy, article_len, features=False):
    model = Summarizer(model_type, article_len, features)
    model.expand_posembs_ifneed()
    # TODO: add model path to config properties
    for model_path in [os.path.join(p, cfg.model_name) for p in MODEL_PATHS]:
        if os.path.exists(model_path):
            model.load(model_path)
            break
    else:
        raise RuntimeError(f'Model file {cfg.model_name} not found among: {MODEL_PATHS}')

    model.froze_backbone(froze_strategy)
    model.unfroze_head()
    return model


class Summarizer(nn.Module):

    enc_output: torch.Tensor
    rouges_values: np.array = np.zeros(4)
    dec_ids_mask: torch.Tensor
    encdec_ids_mask: torch.Tensor

    def __init__(self, model_type, article_len, with_features=False, num_features=10):
        super(Summarizer, self).__init__()

        self.article_len = article_len

        if model_type == 'bert':
            self.backbone, self.tokenizer, BOS, EOS, PAD = self.initialize_bert()
        elif model_type == 'roberta':
            self.backbone, self.tokenizer, BOS, EOS, PAD = self.initialize_roberta()
        else:
            raise Exception(f"Wrong model_type argument: {model_type}")

        if with_features:
            self.features = nn.Sequential(nn.Linear(num_features, 100),
                                          nn.ReLU(),
                                          nn.Linear(100, 100),
                                          nn.ReLU(),
                                          nn.Linear(100, 50))
        else:
            self.features = None

        self.PAD = SpecToken(PAD, convert_token_to_id(self.tokenizer, PAD))
        self.artBOS = SpecToken(BOS, convert_token_to_id(self.tokenizer, BOS))
        self.artEOS = SpecToken(EOS, convert_token_to_id(self.tokenizer, EOS))

        # add special tokens tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<sum>", "</sent>", "</sum>"]})
        self.vocab_size = len(self.tokenizer)
        self.sumBOS = SpecToken("<sum>", convert_token_to_id(self.tokenizer, "<sum>"))
        self.sumEOS = SpecToken("</sent>", convert_token_to_id(self.tokenizer, "</sent>"))
        self.sumEOA = SpecToken("</sum>", convert_token_to_id(self.tokenizer, "</sum>"))
        self.backbone.resize_token_embeddings(200 + self.vocab_size)

        # tokenizer
        self.tokenizer.PAD = self.PAD
        self.tokenizer.artBOS = self.artBOS
        self.tokenizer.artEOS = self.artEOS
        self.tokenizer.sumBOS = self.sumBOS
        self.tokenizer.sumEOS = self.sumEOS
        self.tokenizer.sumEOA = self.sumEOA
        self.vocab_size = len(self.tokenizer)

        # initialize backbone emb pulling
        def backbone_forward(input_ids, input_mask, input_segment, input_pos):
            return self.backbone(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=input_segment,
                position_ids=input_pos,
            )
        self.encoder = lambda *args: backbone_forward(*args)[0]

        # initialize decoder
        if not with_features:
            self.decoder = Classifier(cfg.d_hidden)
        else:
            self.decoder = Classifier(cfg.d_hidden + 50)

    def expand_posembs_ifneed(self):
        print(self.backbone.config.max_position_embeddings, self.article_len)
        if self.article_len > self.backbone.config.max_position_embeddings:
            print("OK")
            old_maxlen = self.backbone.config.max_position_embeddings
            old_w = self.backbone.embeddings.position_embeddings.weight
            logger.log(f"Backbone pos embeddings expanded from {old_maxlen} upto {self.article_len}")
            self.backbone.embeddings.position_embeddings = \
                nn.Embedding(self.article_len, self.backbone.config.hidden_size)
            self.backbone.embeddings.position_embeddings.weight[:old_maxlen].data.copy_(old_w)
            self.backbone.config.max_position_embeddings = self.article_len
        print(self.backbone.config.max_position_embeddings)

    @staticmethod
    def initialize_bert():
        backbone = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=False
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        BOS = "[CLS]"
        EOS = "[SEP]"
        PAD = "[PAD]"
        return backbone, tokenizer, BOS, EOS, PAD

    @staticmethod
    def initialize_roberta():
        backbone = RobertaModel.from_pretrained(
            'roberta-base', output_hidden_states=False
        )
        # initialize token type emb, by default roberta doesn't have it
        backbone.config.type_vocab_size = 2
        backbone.embeddings.token_type_embeddings = nn.Embedding(2, backbone.config.hidden_size)
        backbone.embeddings.token_type_embeddings.weight.data.normal_(
            mean=0.0, std=backbone.config.initializer_range
        )
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        BOS = "<s>"
        EOS = "</s>"
        PAD = "<pad>"
        return backbone, tokenizer, BOS, EOS, PAD

    def save(self, model_path):
        """ Save model in filename

        :param model_path: str
        """
        if not self.features:
            state = {
                'encoder_dict': self.backbone.state_dict(),
                'decoder_dict': self.decoder.state_dict(),
            }
        else:
            state = {
                'encoder_dict': self.backbone.state_dict(),
                'decoder_dict': self.decoder.state_dict(),
                'features_dict': self.features.state_dict(),
            }

        torch.save(state, model_path)

    def load(self, model_path):
        state = torch.load(model_path, map_location=lambda storage, location: storage)
        self.backbone.load_state_dict(state['encoder_dict'])
        self.decoder.load_state_dict(state['decoder_dict'])
        if self.features:
            self.features.load_state_dict(state['features_dict'])


    def froze_backbone(self, froze_strategy):

        assert froze_strategy in ['froze_all', 'unfroze_last4', 'unfroze_all'],\
            f"incorrect froze_strategy argument: {froze_strategy}"

        if froze_strategy == 'froze_all':
            for name, param in self.backbone.named_parameters():
                param.requires_grad_(False)

        elif froze_strategy == 'unfroze_last4':
            for name, param in self.backbone.named_parameters():
                param.requires_grad_(True if (
                    'encoder.layer.11' in name or
                    'encoder.layer.10' in name or
                    'encoder.layer.9' in name or
                    'encoder.layer.8' in name
                ) else False)

        elif froze_strategy == 'unfroze_all':
            for param in self.backbone.parameters():
                param.requires_grad_(True)

    def unfroze_head(self):

        for name, param in self.decoder.named_parameters():
            param.requires_grad_(True)

    @property
    def rouge_1(self):
        return self.rouges_values[0]

    @property
    def rouge_2(self):
        return self.rouges_values[1]

    @property
    def rouge_l(self):
        return self.rouges_values[2]

    @property
    def rouge_mean(self):
        return self.rouges_values[3]

    def forward(self, input_ids, input_mask, input_segment, input_features=None):
        """ Train for 1st stage of model

        :param input_ids: torch.Size([batch_size, article_len])
        :param input_mask: torch.Size([batch_size, article_len])
        :param input_segment: torch.Size([batch_size, article_len])
        :return:
            logprobs | torch.Size([batch_size, summary_len, vocab_size])
        """

        cls_mask = (input_ids == self.artBOS.idx)

        # position ids | torch.Size([batch_size, article_len])
        pos_ids = torch\
            .arange(0, self.article_len, dtype=torch.long, device=input_ids.device)\
            .unsqueeze(0)\
            .repeat(len(input_ids), 1)
        # extract bert embeddings | torch.Size([batch_size, article_len, d_bert])
        enc_output = self.encoder(input_ids, input_mask, input_segment, pos_ids)

        if self.features:
            temp_features = self.features(input_features)
            draft_logprobs = self.decoder(torch.cat([enc_output[cls_mask], temp_features], dim=-1))
        else:
            draft_logprobs = self.decoder(enc_output[cls_mask])

        return draft_logprobs

    def evaluate(self, input_ids, input_mask, input_segment, input_features=None):
        """ Eval for 1st stage of model

        :param input_ids: torch.Size([batch_size, article_len])
        :param input_mask: torch.Size([batch_size, article_len])
        :param input_segment: torch.Size([batch_size, article_len])
        :return:
            draft_ids | torch.Size([batch_size, summary_len])
        """

        cls_mask = (input_ids == self.artBOS.idx)

        # position ids | torch.Size([batch_size, article_len])
        pos_ids = torch\
            .arange(0, self.article_len, dtype=torch.long, device=input_ids.device)\
            .unsqueeze(0)\
            .repeat(len(input_ids), 1)
        # extract bert embeddings | torch.Size([batch_size, article_len, d_bert])
        enc_output = self.encoder(input_ids, input_mask, input_segment, pos_ids)

        ans = []
        for eo, cm in zip(enc_output, cls_mask):
            if self.features:
                scores = self.decoder.evaluate(torch.cat([eo[cm], self.features(input_features)], dim=-1))
            else:
                scores = self.decoder.evaluate(eo[cm])
            ans.append(scores)
        return ans


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x).squeeze(-1)
        scores = self.sigmoid(x)
        return scores

    def evaluate(self, x):
        x = self.linear1(x).squeeze(-1)
        scores = self.sigmoid(x)
        return scores