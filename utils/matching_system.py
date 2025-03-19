# matching_system.py
import os
import json
import torch
from data_classes import Document, Section, Paragraph, MultiSentence, Summary
from original_content_database import OriginalContentDatabaseHandler
from embedding_database import EmbeddingDatabaseHandler
from hotpot_database import HotpotDatabaseHandler
from embedding_model import MultilingualE5LargeInstruct

class MatchingSystem:
    def __init__(self, db_original_content_path, db_embedding_path, db_hotpot_path, device=None):
        self.db_original_content = OriginalContentDatabaseHandler(db_original_content_path)
        self.db_embedding = EmbeddingDatabaseHandler(db_embedding_path)
        self.db_hotpot = HotpotDatabaseHandler(db_hotpot_path)
        
        # Initialize the advanced embedding model
        self.embedding_model = MultilingualE5LargeInstruct(device=device)
    
    def generate_embedding(self, text):
        """
        Generate embedding for text using MultilingualE5LargeInstruct model
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List representation of the embedding vector
        """
        # Use the embedding model to generate embeddings
        embedding_tensor = self.embedding_model.generate_embedding(
            input_text=text
        )
        
        # Convert the PyTorch tensor to a list for storage in ChromaDB
        embedding_list = embedding_tensor.cpu().numpy().tolist()
        
        return embedding_list
    
    def generate_embeddings_batch(self, texts):
        """
        Generate embeddings for multiple texts in a batch
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        # Use the embedding model to generate embeddings for a batch
        embedding_tensors = self.embedding_model.generate_embeddings(
            input_texts=texts
        )
        
        # Convert the PyTorch tensors to lists for storage in ChromaDB
        embedding_lists = embedding_tensors.cpu().numpy().tolist()
        
        return embedding_lists
    
    def add_document(self, title, sections_data):
        """
        Add a complete document with sections, paragraphs, and sentences
        Generate embeddings at all levels
        
        Args:
            title: Document title
            sections_data: List of dictionaries with structure:
                [{
                    'title': section_title,
                    'paragraphs': [
                        {
                            'sentences': ['sentence1', 'sentence2', ...] 
                        },
                        ...
                    ]
                }, ...]
        
        Returns:
            document_id: ID of the created document
        """
        # Create document
        document = Document(title=title, db_handler=self.db_original_content)
        document_id = self.db_original_content.insert_document(document)
        document.id = document_id
        
        # Process sections
        for section_data in sections_data:
            section = Section(title=section_data['title'], document_id=document_id, db_handler=self.db_original_content)
            section_id = self.db_original_content.insert_section(section)
            section.id = section_id
            
            section_paragraphs = []
            # Process paragraphs
            for paragraph_data in section_data['paragraphs']:
                paragraph = Paragraph(section_id=section_id, db_handler=self.db_original_content)
                paragraph_id = self.db_original_content.insert_paragraph(paragraph)
                paragraph.id = paragraph_id
                
                # Collect sentences to generate embeddings in batch for efficiency
                sentences = paragraph_data['sentences']
                multi_sentences = []
                
                # Create and store multi-sentences first
                for sentence_content in sentences:
                    multi_sentence = MultiSentence(content=sentence_content, paragraph_id=paragraph_id)
                    multi_sentence_id = self.db_original_content.insert_multi_sentence(multi_sentence)
                    multi_sentence.id = multi_sentence_id
                    multi_sentences.append(multi_sentence)
                    paragraph.add_multi_sentence(multi_sentence)
                
                # Generate embeddings for all sentences in batch
                sentence_embeddings = self.generate_embeddings_batch(sentences)
                
                # Store sentence embeddings
                for i, multi_sentence in enumerate(multi_sentences):
                    self.db_embedding.insert_embedding(
                        f"sentence_{multi_sentence.id}",
                        sentence_embeddings[i],
                        metadata={
                            'type': 'sentence',
                            'document_id': document_id,
                            'section_id': section_id,
                            'paragraph_id': paragraph_id
                        }
                    )
                
                # Generate and store paragraph embedding
                paragraph_text = " ".join(sentences)
                paragraph_embedding = self.generate_embedding(paragraph_text)
                self.db_embedding.insert_embedding(
                    f"paragraph_{paragraph_id}",
                    paragraph_embedding,
                    metadata={
                        'type': 'paragraph',
                        'document_id': document_id,
                        'section_id': section_id
                    }
                )
                
                section.add_paragraph(paragraph)
                section_paragraphs.append(paragraph_text)
            
            # Generate and store section embedding
            section_text = f"Section: {section_data['title']}\n" + "\n".join(section_paragraphs)
            section_embedding = self.generate_embedding(section_text)
            self.db_embedding.insert_embedding(
                f"section_{section_id}",
                section_embedding,
                metadata={
                    'type': 'section',
                    'document_id': document_id
                }
            )
            
            document.add_section(section)
        
        # Generate and store document embedding
        document_text = document.get_full_text()
        document_embedding = self.generate_embedding(document_text)
        self.db_embedding.insert_embedding(
            f"document_{document_id}",
            document_embedding,
            metadata={
                'type': 'document'
            }
        )
        
        return document_id
    
    def search_by_text(self, query_text, level='sentence', n_results=10):
        """
        Search for similar content using text similarity
        
        Args:
            query_text: Text to search for
            level: Level to search at ('document', 'section', 'paragraph', 'sentence', 'summary', 'all')
            n_results: Number of results to return
        """
        # Generate embedding for the query text
        query_embedding = self.generate_embedding(query_text)
        
        # Define where filter based on the requested level
        where_filter = None
        if level != 'all':
            if level == 'document':
                where_filter = {"type": "document"}
            elif level == 'section':
                where_filter = {"type": "section"}
            elif level == 'paragraph':
                where_filter = {"type": "paragraph"}
            elif level == 'sentence':
                where_filter = {"type": "sentence"}
            elif level == 'document_summary':
                where_filter = {"type": "document_summary"}
            elif level == 'section_summary':
                where_filter = {"type": "section_summary"}
        
        # Perform search with appropriate filters for non-summary types
        results = self.db_embedding.search_by_embedding(
            query_embedding, 
            n_results=n_results,
            where=where_filter
        )
        
        return self._format_search_results(results)

    def generate_summaries_for_documents(self, document_ids):
        """
        Generate summaries for a list of document IDs.
        
        Args:
            document_ids: List of document IDs to generate summaries for.
        
        Returns:
            List of summaries generated for the documents.
        """
        summaries = []
        
        for doc_id in document_ids:
            document = self.db_original_content.get_document(doc_id)
            if document:
                # get section_ids from document
                section_ids = document.get_section_ids()
                section_texts = []
                for section_id in section_ids:
                    section = self.db_original_content.get_section(section_id)
                    if section:
                        # Assuming we have a method to generate a summary from the document text
                        section_text = section.get_full_text()
                        summary_content = self.summarize_text(section_text)
                        summary = Summary(content=summary_content, document_id=doc_id, section_id=section_id)
                        summary.id = self.db_original_content.insert_summary(summary)
                        self.db_embedding.insert_embedding(
                            f"section_summary_{summary.id}",
                            self.generate_embedding(summary_content),
                            metadata={
                                'type': 'section_summary'
                            }
                        )
                        section_texts.append(section_text)
                    else:
                        print(f"Section {section_id} not found for document {doc_id}")
                document_text = f"Title: {document.title}\n\n" + "\n\n".join(section_texts)
                summary_content = self.summarize_text(document_text)
                summary = Summary(content=summary_content, document_id=doc_id)
                summary.id = self.db_original_content.insert_summary(summary)
                self.db_embedding.insert_embedding(
                    f"document_summary_{summary.id}",
                    self.generate_embedding(summary_content),
                    metadata={
                        'type': 'document_summary'
                    }
                )

    def summarize_text(self, text):
        """
        Placeholder for text summarization logic.
        
        Args:
            text: The text to summarize.
        
        Returns:
            A summarized version of the text.
        """
        # Implement your summarization logic here (e.g., using an NLP model)
        return text[:100]  # Example: return the first 100 characters as a summary