import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, d_input: int, mlp_ratio: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_input, d_input * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_input * mlp_ratio, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)