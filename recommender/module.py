import torch.nn as nn
import torch

class RecommenderModule(nn.Module):
    def __init__(self) -> None:
        super(RecommenderModule, self).__init__()

    def forward(self, *args):
        pass

    def embed(self, *args) -> torch.Tensor:
        pass