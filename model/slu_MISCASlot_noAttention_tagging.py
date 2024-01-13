# coding=utf8
import re
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.batch import Batch
import torch.nn.functional as F
import numpy as np


class MISCA_noAtt(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_out_size = 768
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = MISCADecoder(self.bert_out_size, config.num_tags, config.tag_pad_idx)

        self.model_path = "./model/" + config.bert_name
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.bert = BertModel.from_pretrained(self.model_path).to(config.device)

        for name, param in self.bert.named_parameters():
            param.requires_grad = False

    def forward(self, batch: Batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        seq_len = batch.tag_ids.shape[1]

        # 英文 bert-base-chinese 无法识别，将其转换成 [UNK]
        utt_with_en = batch.utt
        utt = [re.sub("[a-zA-Z\s]", "[UNK]", s) for s in utt_with_en]
        # 与 batch.input_ids 相比，开头和结尾各多了一个 token
        tokenizer_out = self.tokenizer(utt, return_tensors='pt', padding=True).to(self.config.device)
        
        out = self.bert(**tokenizer_out)
        hidden = out.last_hidden_state[:, 1:-1, :]  # 去掉第一个和最后一个 token
        hidden = self.dropout_layer(hidden)
        tag_output = self.output_layer(hidden, tag_mask, tag_ids, seq_len)

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

class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True, enforce_sorted=False)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens

class slotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(slotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class AttentionLayer(nn.Module):

    def __init__(self,
                 att_mode, size,d_a,lps,n_labels,n_level
                 ):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(AttentionLayer, self).__init__()
        self.attention_mode = att_mode

        self.size = size
        # For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
        # d_a is the output size of the first linear layer
        self.d_a = d_a if d_a > 0 else self.size

        # r is the number of attention heads

        self.n_labels = n_labels
        self.n_level = n_level

        self.level_projection_size = lps

        self.linear = nn.Linear(self.size, self.size, bias=False)
        
        self.first_linears = nn.ModuleList([nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)])
        self.second_linears = nn.ModuleList([nn.Linear(self.d_a, self.n_labels[0], bias=False) for label_lvl in range(self.n_level)])##
        self.third_linears = nn.ModuleList([nn.Linear(self.size +
                                            (self.level_projection_size if label_lvl > 0 else 0),
                                            self.n_labels[0], bias=True) for label_lvl in range(self.n_level)])##

        # self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal_(first_linear.weight, 0, 3)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal_(linear.weight, 0, 5)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        for linear in self.third_linears:
            torch.nn.init.normal_(linear.weight, 0, 7)

    def forward(self, x, previous_level_projection=None, label_level=0, masks=None):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]

        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        weights = torch.tanh(self.first_linears[label_level](x)) #torch.Size([32, 26, 32])

        att_weights = self.second_linears[label_level](weights)#torch.Size([32, 26, 74])
        att_weights = F.softmax(att_weights, 1).transpose(1, 2) #torch.Size([32, 74, 26])
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        context_vector = att_weights @ x

        batch_size = context_vector.size(0)

        if previous_level_projection is not None:
            temp = [context_vector, #torch.Size([32, 74, 256])
                    previous_level_projection.repeat(1, self.n_labels[0]).view(batch_size, self.n_labels[0], -1)] #torch.Size([32, 74, 74])
            context_vector = torch.cat(temp, dim=2)

        weighted_output = self.third_linears[label_level].weight.mul(context_vector).sum(dim=2).add(
            self.third_linears[0].bias)

        return context_vector, weighted_output, att_weights #torch.Size([32, 74, 256]) torch.Size([32, 74]) torch.Size([32, 74, 26]),WO传回

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)


def perform_attention(model, name, all_output, n_levels):
    attention_weights = None
    previous_level_projection = None
    weighted_outputs = []
    attention_weights = []
    context_vectors = []
    attention_slot = AttentionLayer("label", 256,32,74,[74],n_levels).to("cuda:0")
    linear = nn.Linear(74,74,False).to("cuda:0")
    for level in range(n_levels):
        context_vector, weighted_output, attention_weight = attention_slot(all_output,
                                                            previous_level_projection, label_level=level)

        previous_level_projection = linear(
            torch.sigmoid(weighted_output) )
        previous_level_projection = torch.sigmoid(previous_level_projection) #torch.Size([32, 74])
        weighted_outputs.append(weighted_output)
        attention_weights.append(attention_weight)
        context_vectors.append(context_vector)
        
    return context_vectors, weighted_outputs, attention_weights

class MISCADecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(MISCADecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.lstm_slot = LSTMEncoder(
            input_size,
            256,
            0.1
        ) #(768, 256, 0.1)
        self.lstm_slot = LSTMEncoder(
            input_size,
            256,
            0.1
        )#(768, 256, 0.1)
        self.slot_detection = slotClassifier(self.num_tags, self.num_tags, 0.1)
        self.slot_loss_cnt = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.temp_linear = nn.Linear(256,74)

    def forward(self, encoded, mask, labels=None,seq_lens=20):
        lens = torch.sum(mask, dim=-1).cpu()
        slot_output = self.lstm_slot(encoded, lens)
        heads = torch.zeros([int(mask.shape[0]),int(mask.shape[1])],dtype=torch.int)
        for i in range(lens.shape[0]):
            heads[i,:int(lens[i])]=torch.arange(0,float(lens[i]))
        heads = torch.from_numpy(np.array(heads)).to("cuda:0")
        slot_output = torch.cat(
            [torch.index_select(slot_output[i], 0, heads[i]).unsqueeze(0) for i in range(slot_output.size(0))],
            dim=0
        )
    
        slot_logits = self.temp_linear(slot_output)
        B,l,f = slot_logits.shape
        slot_logits = slot_logits.view(B*l,f)
        
        
        slot_dec = self.slot_detection(slot_logits).view(B,l,f)
        slot_dec += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(slot_dec, dim=-1)  
        total_loss = 0
        # 1. slot Softmax
        if labels is not None:
            slot_count = torch.sum(labels, dim=-1).long()
            slot_loss = self.slot_loss_cnt(slot_dec.view(-1, slot_dec.shape[-1]), labels.view(-1)) #32,26,74
            count_loss = self.slot_loss_cnt(slot_dec.view(B, l*f), slot_count)
            total_loss = (slot_loss + count_loss)
            return prob, slot_loss
        
        return (prob,)

    