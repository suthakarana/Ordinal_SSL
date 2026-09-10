import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from Models.DataLoader import *
import torch
import torch.nn.functional as F

class CE(nn.Module):
    def __init__(self, nclass, gpuid):
        super(CE, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.nclass = nclass
        self.gpuid = gpuid


    def CE(self, logits, targets, cw=None, mask=None):
        device = logits.device

        if cw is not None:
            cw = cw.to(device)

        loss = F.cross_entropy(logits, targets, weight=cw, reduction='none').view(-1)

        if mask is None:
            mask = torch.ones_like(loss, device=device)
        else:
            mask = mask.to(device)

        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-8)
        # loss = loss.mean()
        return loss


