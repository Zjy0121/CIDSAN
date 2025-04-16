#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import warnings


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, l = x.size()

        n = l - 1

        x_minus_mu_square = (x - x.mean(dim=2, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)


# 构建卷积神经网络的模型
class SimCNN(nn.Module):
    def __init__(self, pretrained=False, simam=True, in_channel=1, out_channel=10):
        super(SimCNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=64, stride=15, padding=1),
            nn.BatchNorm1d(16),
            SimAM(),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            SimAM(),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            SimAM(),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            SimAM(),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            SimAM(),
            nn.ReLU(inplace=True))

        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveMaxPool1d(2)

        self.fc = nn.Linear(256 * 2, 256)

    def forward(self, x):
        x = self.pooling(self.conv1(x))
        x = self.pooling(self.conv2(x))
        x = self.pooling(self.conv3(x))
        x = self.pooling(self.conv4(x))
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x