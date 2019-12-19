# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory


class RGD_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 R_module, gpu=False):
        super(RGD_G, self).__init__(embedding_dim, hidden_dim,
                                    vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relgan'

        self.temperature = 1.0  # init value is 1.0

        self._read = R_module  # the reader module

        # RMC
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)
        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # LSTM
        # self.hidden_dim = 32
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
        # self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # in the _parse_struct
        self.mem_size = head_size * num_heads
        internal_dim = 8
        self.emb2_8 = nn.Linear(embedding_dim, internal_dim)
        self.s2mem_entry = nn.Linear(
            int(cfg.max_seq_len / 2) * internal_dim, self.mem_size)

        self.init_params()
        pass

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len, (64, 80)
        :param hidden: (h, c) (64, 1, 512)
        :param need_hidden: if return hidden, use for sampling
        """
        # (64, 80, 32)
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        # provide struct_inp
        # (64, 80, 512)
        struct_inp = torch.zeros(inp.shape[0], inp.shape[1], self.mem_size)
        if self.gpu:
            struct_inp = struct_inp.cuda()
        for i_s in range(cfg.read_interval, inp.shape[1], cfg.read_interval):
            with torch.no_grad():
                struct_t, _ = self.get_R_out(i_s, inp)
                if self.gpu:
                    struct_t = struct_t.cuda()
            # (64, 1, 512) -> project to (64, 80, 512)
            stop_pos = (i_s + cfg.read_interval) % inp.shape[1]
            struct_inp[:, i_s:stop_pos, :] = self._parse_struct(struct_t)

        # out: batch_size * seq_len * hidden_dim  # also out is the pred, hidden is the memory
        out, hidden = self.lstm(emb, hidden, struct_inp)
        # out: (batch_size * len) * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def _parse_struct(self, struct_t):
        """parses struct_t into struct vectors compatable with the memory slot.

        Arguments:
            struct_t {torch.Tensor} -- (64, 40) index vector

        Returns:
            torch.Tensor -- (64, 1, 512)
        """
        batch_size = struct_t.shape[0]
        # (64, 40, 32)
        emb_s = self.embeddings(struct_t)
        # (64, 40, 4) -> (64, 160)
        s_inp = F.relu(self.emb2_8(emb_s))
        s_inp = s_inp.view(batch_size, -1)
        # (64, 1, 512)
        struct_inp = self.s2mem_entry(s_inp).unsqueeze(dim=1)
        struct_inp = 0.1 * struct_inp  # the initial scale factor

        return struct_inp

    def get_R_out(self, step, samples):
        """parses a batch of inputs

        The inputs, samples are a batch of seq during generation, 
        so the step indicates where it reaches right now. Using
        the Reader module to perform the parse. 

        Arguments:
            step {int}  --  indicates on which position the model is
                            currently generating.
            samples {torch.Tensor} 
                        --  an array of index vectors or one-hot vectors

        Returns:
            [torch.Tensor] -- should be array of index vectors
            [torch.Tensor] -- should be array of index vectors
        """
        smp = samples[:, :step]
        # (64, 40)
        batch_text, batch_dep = self._read(smp)
        return batch_text, batch_dep

    def step(self, inp, hidden, struct_inp):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :param struct_t: structure memory, (batch, length)
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        # (64, 1, 32)
        emb = self.embeddings(inp).unsqueeze(1)
        # (64, 1, 512) for out.shape and hidden.shape
        out, hidden = self.lstm(emb, hidden, struct_inp)
        # (64, 4151)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        # (64, )
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        # (64, 4151)
        pred = F.softmax(gumbel_t * self.temperature,
                         dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o

    def sample(self, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        # (64, 80)
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            # (64, 80, 4151)
            all_preds = torch.zeros(
                batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        for b in range(num_batch):
            # (64 batch, 1, 512) a batch of (1, 512) memories, first element 1, others 0
            hidden = self.init_hidden(batch_size)
            # (64,)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            # struct_t = torch.zeros(int(cfg.max_seq_len / 2)).long()
            struct_inp = torch.zeros(batch_size, 1, self.mem_size)
            if self.gpu:
                struct_inp = struct_inp.cuda()
            for i in range(self.max_seq_len):
                # flag for when to call the reader to form and use the structure summary
                if_use_struct = (i >= cfg.read_interval and i %
                                 cfg.read_interval == 0)
                if if_use_struct:
                    # form the structure summary
                    with torch.no_grad():
                        # (64, 40)
                        struct_t, _ = self.get_R_out(
                            i, samples[b * batch_size:(b + 1) * batch_size])
                        if self.gpu:
                            struct_t = struct_t.cuda()
                        # make it (64, 1, 512) serve as a memory slot entry
                    struct_inp = self._parse_struct(struct_t)
                pred, hidden, next_token, _, _ = self.step(
                    inp, hidden, struct_inp)
                # samples - (64, 80), next_token - (64,)
                # samples - (64, 80) only the first i tokens are non-zero elements
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    # (64, 80, 4151)
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples

    def init_hidden(self, batch_size=cfg.batch_size):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.cuda() if self.gpu else memory

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t
