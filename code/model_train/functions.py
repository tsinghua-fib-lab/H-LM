import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from allrank.models.losses import neuralNDCG

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def loss_func(x: Tensor, y: Tensor, reduction: str = 'mean'):
    target = torch.arange(0, x.size(0), 1, device=x.device, dtype=torch.long)
    logits = torch.matmul(x, y.transpose(0, 1))

    loss=F.cross_entropy(logits, target, reduction=reduction)
    prec=(torch.argmax(logits,dim=1)==target).cpu().tolist()

    return prec,loss

def loss_func_cpVSpVSn_ndcg(x: Tensor, y: Tensor, k: int = 5, reduction: str = 'mean'):
    batch_size = x.size(0)
    logits = torch.matmul(x, y.transpose(0, 1))
    target = torch.cat([2*torch.eye(batch_size)]*5+[torch.eye(batch_size)]*5, dim=1).to(x.device)
    loss = neuralNDCG(logits, target)

    _,indices=torch.topk(logits,k,dim=1)
    indices=indices.cpu()
    indices_cp = torch.arange(0, 5*batch_size, batch_size).repeat(batch_size, 1) + torch.arange(0, batch_size).unsqueeze(1)
    indices_p = torch.arange(0, 10*batch_size, batch_size).repeat(batch_size, 1) + torch.arange(0, batch_size).unsqueeze(1)
    cp_corrects = (indices.unsqueeze(2) == indices_cp.unsqueeze(1)).any(2).sum(1)/k
    p_corrects = (indices.unsqueeze(2) == indices_p.unsqueeze(1)).any(2).sum(1)/k

    return cp_corrects.tolist(),p_corrects.tolist(),loss