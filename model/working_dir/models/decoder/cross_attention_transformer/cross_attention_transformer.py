import torch.nn as nn

class CrossAttentionEncoder(nn.Module):
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

        self.input_proj = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_context, d_model, d_heads, dropout) for _ in range(d_layers)
        ])

    def forward(self, seq, context):
        seq = self.input_proj(seq)
        for layer in self.layers:
            seq = layer(seq, context)
        return seq

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_context : int,
        d_model : int,
        d_heads : int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.context_proj = nn.Linear(d_context, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, d_heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(d_model, d_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, seq, context):
        context = self.context_proj(context).unsqueeze(1)
        out1, _ = self.cross_attention(query=seq, key=context, value=context)
        out1 = self.norm1(out1 + seq)

        out2, _ = self.self_attention(query=out1, key=out1, value=out1)
        out2 = self.norm2(out2 + out1)

        return out2