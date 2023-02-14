
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.jit import isinstance
import numpy as np
from tqdm import tqdm

class MiniLMVectorizer():
    def __init__(self):
        
        #self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        #self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        pass
    
    def vectorize_docs(self, sentences, device="cuda"):
        
        self.model = self.model.to(device)
        
        if isinstance( sentences, np.ndarray ):
            sentences = sentences.copy()
            sentences = list(sentences)
        
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(len(sentences)), desc="Vectorizing docs"):
                sentence = [sentences[i]]
                encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
                encoded_input = encoded_input.to(device)
                model_output = self.model(**encoded_input)
                sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
                sentence_embedding = sentence_embedding.cpu().detach().numpy()[0]
                embeddings.append( sentence_embedding )
        embeddings = np.array( embeddings )
        
        return embeddings