import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class LDAM_CCBL(nn.Module):
    def __init__(self, num_classes, class_weights, cls_num_list, Lambda, weight=None, max_m=0.5, s=1, reduction='mean'):
        super(LDAM_CCBL, self).__init__()
        self.device = torch.device("cuda")
        self.num_classes = num_classes
        self.reduction = reduction
        self.Lambda = Lambda
        self.cls_num_list = cls_num_list

        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        # Learnable parameters for specific categories of centers
        self.centers = nn.Parameter(torch.randn(num_classes, num_classes).to(self.device))

    def forward(self, x, target):
        ################################################################################################################
        index = torch.zeros_like(x, dtype=torch.uint8).cuda()
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        ladm_loss = F.cross_entropy(self.s*output, target, weight=self.weight)
        ################################################################################################################
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # kmeans = KMeans(n_clusters=self.num_classes, random_state=0)
        # kmeans.fit(x.detach().cpu().numpy())
        # centers_batch = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(x.device)

        # Collect centers that correspond to labels
        centers_batch = self.centers.to(x.device)
        target = target.to(x.device)

        centers_batch = torch.index_select(centers_batch, 0, target)

        distances = torch.norm(x - centers_batch, p=2, dim=1)

        LHC = distances * self.class_weights[target]
        pt = torch.exp(-LHC)

        CE_loss = nn.CrossEntropyLoss(reduction='none')(x, target)

        ccbl_loss = self.class_weights[target] ** (1-pt) * CE_loss
        ################################################################################################################
        if self.reduction == 'mean':
            ccbl_loss = torch.mean(ccbl_loss)
            loss = self.Lambda * ccbl_loss + (1 - self.Lambda) * ladm_loss
            return loss
        elif self.reduction == 'sum':
            ccbl_loss = torch.sum(ccbl_loss)
            loss = self.Lambda * ccbl_loss + (1 - self.Lambda) * ladm_loss
            return loss
        else:
            return ccbl_loss

