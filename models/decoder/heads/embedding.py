import torch.nn as nn
import torch.nn.functional as F

class EmbeddingHead(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x):
        x = self.head(x)
        x = F.normalize(x, dim=-1)
        return x