from typing import List
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from configs import load_configs

class MultilingualE5LargeInstruct:
    def __init__(self, device: str = None):
        """
        Initialize the embedding model with a pre-trained model and tokenizer.
        
        Args:
            device (str): Device to use for computation (e.g., 'cuda' or 'cpu'). If None, auto-detects.
        """
        cache_dir=load_configs()['multilingual-e5-large-instruct']['cache_dir']
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct', cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct', cache_dir=cache_dir)
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Apply average pooling to the last hidden states.
        
        Args:
            last_hidden_states (Tensor): Last hidden states from the model.
            attention_mask (Tensor): Attention mask for the input tokens.
        
        Returns:
            Tensor: Pooled embeddings.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(self, query: str, task_description: str = None) -> str:
        """
        Format the input text with task description and query.
        
        Args:
            task_description (str): Description of the task.
            query (str): Input query.
        
        Returns:
            str: Formatted input text.
        """
        return f'Instruct: {task_description}\nQuery: {query}' if task_description else query

    def generate_embedding(self, input_text: str, task_description: str = None) -> Tensor:
        """
        Generate embeddings for a single input text.
        
        Args:
            input_text (str): Input text.
            task_description (str): Task description for the model.
        
        Returns:
            Tensor: Embedding for the input text (1D tensor).
        """
        # Format input text with task description
        formatted_text = self.get_detailed_instruct(input_text, task_description)

        # Tokenize the input text
        batch_dict = self.tokenizer(
            [formatted_text],  # Wrap in a list to create a batch of size 1
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return embeddings.squeeze(0)  # Convert (1, embedding_dim) to (embedding_dim,)

    def generate_embeddings(self, input_texts: List[str], task_description: str = None) -> Tensor:
        """
        Generate embeddings for a list of input texts.
        
        Args:
            input_texts (List[str]): List of input texts.
            task_description (str): Task description for the model.
        
        Returns:
            Tensor: Embeddings for the input texts (2D tensor of shape (num_texts, embedding_dim)).
        """
        # Format input texts with task description
        formatted_texts = [self.get_detailed_instruct(text, task_description) for text in input_texts]

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            formatted_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return embeddings  # Shape: (num_texts, embedding_dim)