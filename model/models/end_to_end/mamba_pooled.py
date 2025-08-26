import torch.nn as nn
from ..decoder import MambaPooledEncoder, ClassifierHead, EmbeddingHead

class MambaPooled(nn.Module):
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

        self.mamba = MambaPooledEncoder(**mamba_par)
        
        self.classifier_head = ClassifierHead(d_input)
        self.embedding_head = EmbeddingHead(d_input)
    
    def forward(self, x, context, return_embeddings=False):
        x, _ = self.mamba(x, context)
        x = x.mean(dim=1)
        if return_embeddings:
            return self.embedding_head(x)
        else:
            return self.classifier_head(x)