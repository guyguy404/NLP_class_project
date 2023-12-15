#coding=utf8
import re
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForMaskedLM

from utils.batch import Batch


class SLUBertTagging(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.hidden_size = 21128
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

        self.model_path = "./model/bert-base-chinese"
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.bert = BertForMaskedLM.from_pretrained(self.model_path)

        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

    def forward(self, batch: Batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask

        # 英文 bert-base-chinese 无法识别，将其转换成 [UNK]
        utt_with_en = batch.utt
        utt = [re.sub("[a-zA-Z\s]", "[UNK]", s) for s in utt_with_en]
        # 与 batch.input_ids 相比，开头和结尾各多了一个 token
        tokenizer_out = self.tokenizer(utt, return_tensors='pt', padding=True).to(self.config.device)
        out = self.bert(**tokenizer_out)
        hidden = out.logits[:, 1:-1, :]    # 去掉开头和结尾的 token
        hidden = self.dropout_layer(hidden)
        tag_output = self.output_layer(hidden, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        # self.output_layer = nn.Linear(input_size, num_tags)
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_tags)
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
