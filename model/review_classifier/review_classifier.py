from .mamba_decoder import MambaDecoder
from .nomic_embedder import NomicEmbedder
from typing import List

class ReviewClassifier():
    def __init__(self, model_path: str):
        self.encoder = NomicEmbedder()
        self.decoder = MambaDecoder(
            num_layers=3,
            d_input=768,
            d_model=256,
            d_context=768
        )

    def encode(self, texts: List[str] | str, return_sequence: bool = False):
        return self.encoder.embed(texts=texts, return_tokens=return_sequence)
    
    def decode(self, texts: List[Tensor])
    