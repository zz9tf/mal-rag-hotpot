import chromadb
from chromadb.config import Settings

class EmbeddingDatabaseHandler:
    def __init__(self, chroma_db_path):
        self.chroma_db_path = chroma_db_path
        # Chroma client initialization
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection("embeddings")
    
    def insert_embedding(self, id, embedding_value, metadata=None):
        """
        Insert an embedding into the database
        
        Args:
            id: A unique identifier for the embedding
            embedding_value: The embedding vector
            metadata: Optional metadata to store with the embedding
        
        Returns:
            id: The id of the stored embedding
        """
        self.collection.add(
            embeddings=[embedding_value],
            ids=[str(id)],
            metadatas=[metadata] if metadata else None
        )
        return id
    
    def get_embedding(self, id):
        """
        Retrieve an embedding by its id
        
        Args:
            id: The id of the embedding to retrieve
            
        Returns:
            The embedding vector or None if not found
        """
        result = self.collection.get(ids=[str(id)], include=["embeddings"])
        if result and result.get("embeddings") is not None and result["embeddings"].size > 0:
            return result["embeddings"][0]
        return None
    
    def search_by_embedding(self, query_embedding, n_results=10, where=None):
        """
        Search for similar embeddings
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            where: Optional filtering condition (dict) for metadata
            
        Returns:
            Dictionary with search results
        """
        # Use the where parameter to filter by metadata if provided
        if where:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        return results
    
    def get_by_metadata(self, metadata_filter, limit=100):
        """
        Retrieve embeddings by metadata
        
        Args:
            metadata_filter: Dictionary with metadata conditions
            limit: Maximum number of results to return
        
        Returns:
            List of matching embeddings
        """
        results = self.collection.get(
            where=metadata_filter,
            limit=limit
        )
        return results

    def bulk_insert_embeddings(self, embedding_data, batch_size=500):
        """
        Insert multiple embeddings in batches.
        
        Args:
            embedding_data: List of dicts with 'id', 'embedding', and 'metadata' keys
            batch_size: Number of embeddings to insert in a single ChromaDB operation
            
        Returns:
            True if successful
        """
        # Process in sub-batches for ChromaDB
        for i in range(0, len(embedding_data), batch_size):
            batch = embedding_data[i:i+batch_size]
            
            ids = [item['id'] for item in batch]
            embeddings = [item['embedding'] for item in batch]
            metadatas = [item['metadata'] for item in batch]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        
        return True
    
if __name__ == "__main__":
    from configs import load_configs
    config = load_configs()
    # Example usage
    db_handler = EmbeddingDatabaseHandler(config['EmbeddingDatabase']['db_path'])

    # Insert an embedding
    db_handler.insert_embedding(id=1, embedding_value=[0.1, 0.2, 0.3], metadata={"category": "example"})

    # Retrieve an embedding
    embedding = db_handler.get_embedding(id=1)
    print(embedding)

    # Search for similar embeddings
    results = db_handler.search_by_embedding(query_embedding=[0.1, 0.2, 0.3], n_results=5)
    print(results)