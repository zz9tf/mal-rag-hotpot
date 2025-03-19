from typing import List
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from configs import load_configs
import numpy as np

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
        self.bytes_per_token = None
        self.model.to(self.device)
        
    def _calibrate_memory_per_token(self):
        """Calibrate memory usage per token using a test batch"""
        test_texts = ["Calibration text " * 10]  # ~20 tokens
        tokens = self.tokenizer(test_texts, return_tensors='pt', padding=True).to(self.device)
        
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
        
        # Temporary model initialization
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        end_mem = torch.cuda.memory_allocated()
        del outputs, tokens
        torch.cuda.empty_cache()
        
        total_tokens = sum(len(t) for t in self.tokenizer(test_texts)['input_ids'])
        self.bytes_per_token = (end_mem - start_mem) / total_tokens
        print(f"Calibrated memory per token: {self.bytes_per_token/1024:.2f} KB")

    def _calculate_safe_batch_size(self, token_counts):
        """Dynamically calculate batch size based on current memory"""
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated()
        
        # Use 90% of available memory as safety margin
        available_mem = free_mem * 0.9
        
        if self.bytes_per_token is None:
            self._calibrate_memory_per_token()
        
        # Sort texts descending by token count to minimize padding
        sorted_indices = np.argsort(-np.array(token_counts))
        sorted_counts = [token_counts[i] for i in sorted_indices]
        
        batch = []
        max_seq_len = 0
        batch_map = []

        for count in sorted_counts:
            # Estimate memory for this sequence including padding
            seq_mem = count * self.bytes_per_token
            potential_max = max(max_seq_len, count)
            batch_mem = len(batch) * potential_max * self.bytes_per_token
            
            if batch_mem + seq_mem > available_mem:
                if not batch:  # Single sequence too large
                    print(f"Warning: Sequence with {count} tokens exceeds available memory")
                    break
                batch_map.append(batch)
                batch = [count]
                max_seq_len = count
            else:
                batch.append(count)
                max_seq_len = max(max_seq_len, count)
        
        if batch:
            batch_map.append(batch)
            
        return [len(b) for b in batch_map], sorted_indices

    def generate_embeddings(self, input_texts: List[str], task_description: str = None) -> Tensor:
        if self.model is None:
            self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').to(self.device)
            self.model.eval()
        
        # Tokenize all texts first
        tokenized = self.tokenizer(input_texts, padding=False, truncation=True)
        token_counts = [len(ids) for ids in tokenized['input_ids']]
        
        # Calculate optimal batch sizes
        batch_sizes, sorted_indices = self._calculate_safe_batch_size(token_counts)
        sorted_texts = [input_texts[i] for i in sorted_indices]
        
        all_embeddings = []
        
        ptr = 0
        for batch_size in batch_sizes:
            batch_texts = sorted_texts[ptr:ptr+batch_size]
            ptr += batch_size
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            all_embeddings.append(embeddings)
            
            # Clean up
            del inputs, outputs
            torch.cuda.empty_cache()
        
        # Restore original order
        reverse_indices = np.argsort(sorted_indices)
        final_embeddings = np.concatenate(all_embeddings)[reverse_indices]
        
        return final_embeddings.tolist()

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

    # def generate_embeddings(self, input_texts: List[str], task_description: str = None) -> Tensor:
    #     """
    #     Generate embeddings for a list of input texts.
        
    #     Args:
    #         input_texts (List[str]): List of input texts.
    #         task_description (str): Task description for the model.
        
    #     Returns:
    #         Tensor: Embeddings for the input texts (2D tensor of shape (num_texts, embedding_dim)).
    #     """
    #     # Format input texts with task description
    #     formatted_texts = [self.get_detailed_instruct(text, task_description) for text in input_texts]

    #     # Tokenize the input texts
    #     batch_dict = self.tokenizer(
    #         formatted_texts,
    #         max_length=512,
    #         padding=True,
    #         truncation=True,
    #         return_tensors='pt'
    #     ).to(self.device)

    #     # Generate embeddings
    #     with torch.no_grad():
    #         outputs = self.model(**batch_dict)
    #         embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    #     return embeddings  # Shape: (num_texts, embedding_dim)