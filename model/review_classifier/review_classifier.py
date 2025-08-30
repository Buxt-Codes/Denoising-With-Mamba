from .decoder import MambaDecoder
from .embedder import NomicEmbedder
from typing import List, Tuple
from torch import Tensor
import torch

class ReviewClassifier():
    def __init__(self, model_path: str, encoder_path: str | None = None):
        self.encoder = NomicEmbedder(encoder_path if encoder_path is not None else "nomic-ai/nomic-embed-text-v1.5")
        self.decoder = MambaDecoder(
            num_layers=3,
            d_input=768,
            d_model=256,
            d_context=768
        )

        self.load(model_path)

    def encode(self, texts: List[str] | str, return_sequence: bool = False) -> Tensor:
        return self.encoder.embed(texts=texts, return_tokens=return_sequence)
    
    def decode(self, texts: List[Tensor] | Tensor, locations: List[Tensor] | Tensor) -> Tuple[List[int], List[float]]:
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
            locations = locations.repeat(texts.shape[0], 1)
        with torch.no_grad():
            out = self.decoder(texts=texts, location=locations)
        out = out.detach().cpu()    
        label = (out > 0.5).long()
        confidence = torch.where(label == 1, out, 1 - out)
        return label.tolist(), confidence.tolist()
    
    def classify(self, texts: List[str] | str, locations: List[str] | str):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(locations, str):
            locations = [locations]

        if len(texts) != len(locations):
            assert len(locations) == 1, "Expected either 1 location or same number of locations as texts, got {} and {}".format(len(locations), len(texts))
            
        embed_texts = self.encode(texts=texts, return_sequence=True)
        embed_locations = self.encode(texts=locations, return_sequence=False)
        return self.decode(texts=embed_texts, locations=embed_locations)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
    
    def load(self, model_path: str):
        self.decoder.load_state_dict(torch.load(model_path))