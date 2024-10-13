import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CSCE(nn.Module):
    def __init__(self, num_classes, feat_dim, class_weights, cls_num_list, reduction='mean'):
        super(CSCE, self).__init__()
        self.device = torch.device("cuda")
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.reduction = reduction
        self.cls_num_list = cls_num_list

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(self.device))
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    def forward(self, x, target):

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        centers_batch = self.centers.to(x.device)
        target = target.to(x.device)

        centers_batch = torch.index_select(centers_batch, 0, target)

        distances = torch.norm(x - centers_batch, p=2, dim=1)

        LHC = distances * self.class_weights[target]
        pt = torch.exp(-LHC)

        CE_loss = nn.CrossEntropyLoss(reduction='none')(x, target)

        ccbl_loss = self.class_weights[target] ** (1-pt) * CE_loss
        if self.reduction == 'mean':
            loss = torch.mean(ccbl_loss)
            return loss
        elif self.reduction == 'sum':
            loss = torch.sum(ccbl_loss)
            return loss

