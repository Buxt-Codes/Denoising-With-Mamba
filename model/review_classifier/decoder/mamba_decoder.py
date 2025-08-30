import torch.nn as nn
from .mamba import Mamba

class MambaDecoder(nn.Module):
    def __init__(
        self,
        num_layers : int,
        d_input : int,
        d_model : int,
        d_context : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
        parallel : bool = False,
        mlp_ratio: int = 2,
        multi_classes: bool = False
    ):
        super().__init__()
        
        mamba_par = {
            'num_layers' : num_layers,
            'd_input' : d_input,
            'd_context': d_context,
            'd_model' : d_model,
            'd_state' : d_state,
            'd_discr' : d_discr,
            'ker_size': ker_size,
            'parallel': parallel,
        }

        if multi_classes:
            num_classes = 4
        else:
            num_classes = 1

        self.mamba = Mamba(**mamba_par)
        self.classifier = nn.Sequential(
            nn.Linear(d_input, d_input * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_input * mlp_ratio, num_classes),
        )

        self.multi_classes = multi_classes
    
    def forward(self, x, context):
        x, _ = self.mamba(x, context)
        x = x.mean(dim=1)
        if self.multi_classes:
            return self.classifier(x)
        else:
            return nn.Sigmoid(self.classifier(x))