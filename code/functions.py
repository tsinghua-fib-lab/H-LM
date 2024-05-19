import torch
from torch import Tensor

def test_cp_logits_topk(logits: Tensor, k: int, reduction: str = 'mean'):
    batch_size = logits.size(0)

    _,indices=torch.topk(logits,k,dim=1)
    indices=indices.cpu()

    indices_cp = torch.arange(0, 5*batch_size, batch_size).repeat(batch_size, 1) + torch.arange(0, batch_size).unsqueeze(1)
    cp_corrects = (indices.unsqueeze(2) == indices_cp.unsqueeze(1)).any(2).sum(1)/k

    return cp_corrects.tolist()