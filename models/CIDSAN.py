#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
from models.SimAM_CNN import SimCNN


class CIDSAN_SimCNN(nn.Module):
    def __init__(self, pretrained=False):
        super(CIDSAN_SimCNN, self).__init__()
        self.SimCNN = SimCNN(pretrained)
        self.__in_features = 256 * 1

    def forward(self, x):
        x = self.SimCNN(x)
        return x

    def output_num(self):
        return self.__in_features



