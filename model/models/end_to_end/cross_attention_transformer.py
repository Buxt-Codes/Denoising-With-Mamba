import torch.nn as nn
from ..decoder import CrossAttentionEncoder, ClassifierHead, EmbeddingHead

class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        d_input : int,
        d_context : int,
        d_model : int,
        d_layers : int,
        d_heads : int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.transformer = CrossAttentionEncoder(
            d_input=d_input,
            d_context=d_context,
            d_model=d_model,
            d_layers=d_layers,
            d_heads=d_heads,
            dropout=dropout
        )
        
        self.classifier_head = ClassifierHead(d_model)
        self.embedding_head = EmbeddingHead(d_model)
    
    def forward(self, x, context, return_embeddings=False):
        x = self.transformer(x, context)
        
        if return_embeddings:
            return self.embedding_head(x)
        else:
            return self.classifier_head(x)