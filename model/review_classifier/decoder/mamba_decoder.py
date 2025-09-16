import torch.nn as nn
from torch import Tensor

from .mamba import MambaFiLM, MambaFiLMConfig

class MambaFiLMDecoder(nn.Module):
    def __init__(
        self,
        d_input : int = 768,
        d_context : int = 768,
        d_model : int = 256,
        num_layers : int = 4,
        parallel : bool = True,
        mlp_ratio: int = 2
    ):
        """
        Initializes the Mamba FiLM decoder.

        Args:
            d_input (int): The input feature dimension.
            d_context (int): The context feature dimension.
            d_model (int): The model feature dimension.
            num_layers (int): The number of mamba layers.
            parallel (bool, optional): Whether to use parallel scan. Defaults to True.
            mlp_ratio (int, optional): The ratio of the hidden dimension in the output MLP. Defaults to 2.
        """
        super().__init__()
        
        mamba_par = {
            'n_layers' : num_layers,
            'd_model' : d_model,
            'pscan': parallel,
        }

        config = MambaFiLMConfig(**mamba_par)

        self.mamba = MambaFiLM(config)

        self.input_proj = nn.Linear(d_input, d_model)
        self.context_proj = nn.Linear(d_context, d_model)


        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_model * mlp_ratio, 4),
        )
    
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Forward pass of the Mamba FiLM decoder.

        Args:
            x (torch.Tensor): The input sequence.
            context (torch.Tensor): The context vector.

        Returns:
            torch.Tensor: The class logits.
        """
        x = self.input_proj(x)
        context = self.context_proj(context)

        x = self.mamba(x, context)
        x = x.mean(dim=1)
        return self.classifier(x)