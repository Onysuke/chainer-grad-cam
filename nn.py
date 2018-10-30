#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class NN_ResNet50(chainer.Chain):
    def __init__(self, n_out):
        super(NN_ResNet50, self).__init__()
        with self.init_scope():
            self.resnet50 = L.ResNet50Layers()
            self.fc = L.Linear(in_size=None, out_size=n_out)

    def __call__(self, x):
        h1 = self.resnet50(x, layers=["res5"])["res5"]
        h2 = F.average_pooling_2d(h1, 7, stride=1)
        out = self.fc(h2)
        return out

    def extract_bnf(self, x):
        h1 = self.resnet50(x, layers=["res5"])["res5"]
        bnf = F.average_pooling_2d(h1, 7, stride=1)
        return bnf

class NN_ResNet50_bnf(chainer.Chain):
    def __init__(self, n_bn, n_out, n_out_new):
        super(NN_ResNet50_bnf, self).__init__()
        with self.init_scope():
            self.resnet50 = L.ResNet50Layers()
            self.fc = L.Linear(in_size=None, out_size=n_out)
            self.fc_bn = L.Linear(2048, n_bn)
            self.fc_out_new = L.Linear(n_bn, n_out_new)

    def __call__(self, x):
        h1 = self.resnet50(x, layers=["res5"])["res5"]
        h2 = F.average_pooling_2d(h1, 7, stride=1) #h2 2048
        h_bn = F.dropout(F.relu(self.fc_bn(h2)), ratio=0.3)
        return self.fc_out_new(h_bn)
    '''
    def add_layers(self, n_bn, n_out_new):
        with self.init_scope():
            self.fc_bn = L.Linear(2048, n_bn)
            self.fc_out_new = L.Linear(n_bn, n_out_new)
    '''
    def extract_bnf(self, x):
        h1 = self.resnet50(x, layers=["res5"])["res5"]
        h2 = F.average_pooling_2d(h1, 7, stride=1)
        h_bn = self.fc_bn(h2) # reluかけるべきか
        return h_bn
