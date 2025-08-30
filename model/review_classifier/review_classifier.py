from .mamba_decoder import MambaDecoder
from .nomic_embedder import NomicEmbedder
from typing import List
from torch import Tensor
import torch

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
    
    def decode(self, texts: List[Tensor] | Tensor, locations: List[Tensor] | Tensor):
        if isinstance(texts, list):
            texts = torch.stack(texts, dim=0)
        
        assert texts.ndim == 3, "Expected 3D tensor for texts, got {}".format(texts.ndim)
        assert texts.shape[2] == 768, "Expected last dimension to be 768, got {}".format(texts.shape[2])
        
        if isinstance(locations, list):
            locations = torch.stack(locations, dim=0)
        
        assert locations.ndim == 2, "Expected 2D tensor for location, got {}".format(locations.ndim)
        assert locations.shape[1] == 768, "Expected last dimension to be 768, got {}".format(locations.shape[1])
        
        if locations.shape[0] != texts.shape[0]:
            assert locations.shape[0] == 1, "Expected either 1 location or same number of locations as texts, got {} and {}".format(locations.shape[0], texts.shape[0])
            location = locations.repeat(texts.shape[0], 1)
        out = self.decoder(texts=texts, location=locations)
        label = (out > 0.5).long()
        confidence = torch.where(label == 1, out, 1 - out)
        return label, confidence
    