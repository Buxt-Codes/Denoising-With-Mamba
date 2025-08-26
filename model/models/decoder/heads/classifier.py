import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_model * mlp_ratio, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)