# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import config as cfg

from utils.text_process import *


class RGD_R(object):
    def __init__(self, if_test_data):
        self.nlp = spacy.load("en_core_web_sm")
        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)
        if if_test_data:  # used for the classifier
            self.word2idx_dict, self.idx2word_dict = load_test_dict(
                cfg.dataset)
        self.w2i_dep_dict, self.i2w_dep_dict = load_dict(cfg.depname)

    def parse_one(self, inp):
        """wraps the input sentences into a list of structure nodes.

        If  the inp is not string, first convert string using a dictionary.

        Arguments:
            inp {Optional[string, array]} -- [description]
        """
        text = inp
        doc = self.nlp(text)
        text_pile = []
        dep_pile = []
        for token in doc:
            if token.dep_ == 'ROOT':
                text_pile.append(token.text)
                dep_pile.append(token.dep_)
                for child in token.children:
                    if child.dep_ == 'punct':
                        continue
                    text_pile.append(child.text)
                    dep_pile.append(child.dep_)

        # here the two piles are list of real Texts
        # convert to index vectors now, and later in the instructor will convert to one hot vectors
        def tokens_to_index(token_list, mode):
            if mode == 'text':
                dictionary = self.word2idx_dict
            elif mode == 'dep':
                dictionary = self.w2i_dep_dict
            else:
                raise NotImplementedError
            ind_tensor = torch.zeros(int(cfg.max_seq_len / 2)).long()
            cnt = 0
            for token in token_list[:int(cfg.max_seq_len / 2)]:
                ind_tensor[cnt] = int(
                    dictionary[token]) if token in dictionary else 0
                cnt += 1
            return ind_tensor

        # text_pile, one line of toke for a doc line, so it is like (structure_maxlen, )
        text_pile = tokens_to_index(text_pile, mode='text')
        dep_pile = tokens_to_index(dep_pile, mode='dep')
        return text_pile, dep_pile

    def __call__(self, inp):
        """parses a batch of inputs

        Arguments:
            inp {[torch.Tensor]} -- a batch of sampled index tensors, eg. shape (64, 37)

        Returns:
            [torch.Tensor] -- should be array of one-hot vectors
        """
        text_list = self._convert(inp)  # a list of sentences in real words
        batch_text = []
        batch_dep = []
        for text in text_list:
            t_pile, d_pile = self.parse_one(text)
            batch_text.append(t_pile)
            batch_dep.append(d_pile)
        batch_text = torch.stack(batch_text)
        batch_dep = torch.stack(batch_dep)
        return batch_text, batch_dep

    def _convert(self, inp):
        if isinstance(inp, str):
            return inp
        elif isinstance(inp, torch.Tensor) and inp.ndim == 2:
            # a list of index
            # shape in (64, 37) index vectors
            sample_words = tensor_to_tokens(inp, self.idx2word_dict)
            sentences = [' '.join(sent) for sent in sample_words]
            return sentences
        elif isinstance(inp, torch.Tensor) and inp.ndim == 3:
            # a list of (one hot) arrays
            # shape should be like (64, 20 length, 5000 vocabs)
            # and this should be applied in the gen_samples scenario
            samples = inp.max(2)
            sample_words = tensor_to_tokens(samples, self.idx2word_dict)
            sentences = [' '.join(sent) for sent in sample_words]
            return sentences
            # sample words should be list of list of words
        else:
            raise NotImplementedError


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")

    def _test(inp):
        text = inp
        doc = nlp(text)
        text_pile = []
        dep_pile = []
        for token in doc:
            if token.dep_ == 'ROOT':
                text_pile.append(token.text)
                dep_pile.append(token.dep_)
                for child in token.children:
                    if child.dep_ == 'punct':
                        continue
                    text_pile.append(child.text)
                    dep_pile.append(child.dep_)
        return text_pile, dep_pile

    # parse a training document
    doc_dir = '/home/haotian/Code/TextGAN-PyTorch/dataset/yelp.txt'
    out_dir = '/home/haotian/Code/TextGAN-PyTorch/dataset/yelp_dep.txt'
    with open(doc_dir, 'r') as f:
        with open(out_dir, 'w') as f_out:
            while True:
                line = f.readline()
                _, dep_pile = _test(line)
                f_out.write(' '.join(dep_pile) + '\n')
                if not line:
                    break
