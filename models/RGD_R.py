# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy


class RGD_R(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, inp):
        """wraps the input sentences into a list of structure nodes.

        If  the inp is not string, first convert string using a dictionary.

        Arguments:
            inp {Optional[string, array]} -- [description]
        """
        text = self._convert(inp)
        doc = self.nlp(text)
        text_pile = []
        dep_pile = []
        for token in doc:
            if token.dep_ == 'ROOT':
                text_pile.append(token.text)
                dep_pile.append(token.dep_)
                for child in token.children:
                    text_pile.append(child.text)
                    dep_pile.append(child.dep_)
        ('change here to some usual input format and types')
        return text_pile, dep_pile

    def _convert(self, inp):
        if isinstance(inp, str):
            return inp
        elif isinstance(inp, torch.Tensor):
            # a list of (one hot) arrays
            raise NotImplementedError


if __name__ == '__main__':
    # parse a training document
    doc_dir = '/home/haotian/Code/TextGAN-PyTorch/dataset/yelp.txt'
    out_dir = '/home/haotian/Code/TextGAN-PyTorch/dataset/yelp_dep.txt'
    read = RGD_R()
    with open(doc_dir, 'r') as f:
        with open(out_dir, 'w') as f_out:
            while True:
                line = f.readline()
                _, dep_pile = read(line)
                f_out.write(' '.join(dep_pile) + '\n')
                if not line:
                    break
