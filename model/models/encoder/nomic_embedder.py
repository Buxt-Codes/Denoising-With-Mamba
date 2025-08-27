from transformers import AutoTokenizer, AutoModel
import torch

class NomicEmbedder:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.max_length = self.tokenizer.model_max_length
        self.max_tokens = max_tokens

    def embed(self, texts, return_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_tokens if self.max_tokens < self.max_length else self.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
    
        token_embeddings = outputs.last_hidden_state  

        if return_tokens:
            return token_embeddings
        
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  
        masked_embeddings = token_embeddings * attention_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        batch_vectors = sum_embeddings / sum_mask  

        return batch_vectors
    
    def to(self, device):
        self.model.to(device)
