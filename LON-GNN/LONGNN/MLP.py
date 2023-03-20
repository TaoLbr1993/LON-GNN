import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hidden, dropout):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_feat, hidden)
        self.lin2 = nn.Linear(hidden, out_feat)
        self.dropout = dropout
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.lin2(x)
        return x

class MLPJ(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(MLPJ, self).__init__()
        self.lin1 = nn.Linear(in_feat, out_feat)
        self.lin2 = nn.Linear(out_feat, out_feat)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
''' 
    Adapted from https://github.com/GraphPKU/JacobiConv/blob/master/impl/models.py
    Module for residual MLP
    '''
class Seq(nn.Module):
    
    def __init__(self, modlist: Iterable[nn.Module]):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

class ResBlock(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return x + self.mod(x)