import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = nn.functional.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cosine_similarity, 2) + (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))
        return loss_contrastive