import torch
import torch.nn.functional as F

def emd(x, y):
    xc = torch.cumsum(x, dim=-1)
    yc = torch.cumsum(y, dim=-1)
    xr = x + xc[...,-1:] - xc
    yr = y + yc[...,-1:] - yc 
    
    return 0.5 * (F.l1_loss(xc, yc) + F.l1_loss(xr, yr))

def abs_l1(x, y):
    return F.l1_loss(torch.sum(x, dim=-1), torch.sum(y, dim=-1))