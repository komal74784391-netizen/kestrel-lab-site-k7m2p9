"""Neighbourhood contrastive objective."""

import torch
import torch.nn.functional as F


def neighbourhood_contrastive_loss(anchor, positive, negatives, temperature=0.15):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / temperature
    neg_sim = torch.einsum("bd,bkd->bk", anchor, negatives) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
