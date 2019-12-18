# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminator import CNNDiscriminator

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]


class RGD_D(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(RGD_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                    gpu, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.emb2one = nn.Linear(self.embed_dim, 1)
        self.len2one = nn.Linear(int(max_seq_len/2), 1)

        self.init_params()

    def forward(self, inp, struct_inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        # (64, 1, 80, 64)
        emb = self.embeddings(inp).unsqueeze(
            1)  # batch_size * 1 * max_seq_len * embed_dim

        # 4 entry list, each (64, 300, k=76-79, 64)
        # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        cons = [F.relu(conv(emb)) for conv in self.convs]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2)
                 for con in cons]  # [batch_size * num_filter * num_rep]
        # (64, 1200, 64)
        pred = torch.cat(pools, 1)
        # (4096, 1200)
        # (batch_size * num_rep) * feature_dim
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)
        highway = self.highway(pred)
        # (4096, 1200)
        pred = torch.sigmoid(highway) * F.relu(highway) + \
            (1. - torch.sigmoid(highway)) * pred  # highway

        # (4096, 100)
        pred = self.feature2out(self.dropout(pred))
        # (4096,)
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        # the strutual input part
        # share embedding
        # (64 batch, 40 max_len/2, 64 embed_dim)
        emb_s = self.embeddings(struct_inp.float())
        # (64, 40)
        pred_s = F.relu(self.emb2one(emb_s)).squeeze(2)
        # (64,)
        pred_s = self.len2one(self.dropout(pred_s)).squeeze(1)
        # (4096 ,)
        pred_s = torch.repeat_interleave(pred_s, 64)

        logits = logits + 0.1 * pred_s

        return logits
