'''
Time representaton to generate time embeddings
'''
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn

TIME_TYPE = {'no-time':0, 'point-in-time':1, 'only-begin':2, 'only-end':3, 'full-interval':4}
INVERSE_TIME_TYPE = {0:'no-time', 1:'point-in-time', 2:'only-begin', 3:'only-end', 4:'full-interval'}

class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class Lambda3(Regularizer): ## time regularizer for complex vectors 　　 
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        # ddiff = factor[1:] - factor[:-1]
        ddiff = factor
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

        # diff = torch.sqrt(torch.sum(ddiff**2, 1))
        # return self.weight *torch.sum(diff) / (factor.shape[0] - 1)

        # diff = torch.sqrt(ddiff**2)**2
        # return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]

class L2(Regularizer):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factor):
        norm = 0
        return self.weight*torch.mean(torch.norm(factor, dim=-1))
        