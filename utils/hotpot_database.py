# database_hotpot.py
import sqlite3
import json
from tqdm import tqdm
from embedding_model import MultilingualE5LargeInstruct

class HotpotDatabaseHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_db()
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # This enables column access by name
        self.cursor = self.conn.cursor()
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
    
    def initialize_db(self):
        self.connect()
        
        # Create ContentData table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ContentData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL, -- A sentence stored as JSON string
            sentence_id TEXT, -- Matched sentence id from original content database
            paragraph_id TEXT, -- Matched paragraph id from original content database
            section_id TEXT, -- Matched section id from original content database
            document_id TEXT, -- Matched document id from original content database
            embedding_id INTEGER, -- Foreign key to ContentEmbedding table
            FOREIGN KEY (embedding_id) REFERENCES ContentEmbedding(id)
        )
        ''')
        
        # Create Question table with updated parameters
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS Question (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_in_dataset TEXT UNIQUE NOT NULL,
            question TEXT NOT NULL,
            supporting_facts_refs TEXT NOT NULL,  -- A list of ids stored as JSON string
            context_refs TEXT NOT NULL,  -- A list of ids stored as JSON string
            answer TEXT NOT NULL,
            type TEXT NOT NULL,
            level TEXT NOT NULL,
            question_type TEXT NOT NULL,
            embedding_id INTEGER, -- Foreign key to QuestionEmbedding table
            FOREIGN KEY (embedding_id) REFERENCES QuestionEmbedding(id)
        )
        ''')
        
        # Create QuestionEmbedding table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS QuestionEmbedding (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL  -- Store embedding as a binary large object
        )
        ''')
        
        # Create ContentEmbedding table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ContentEmbedding (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL  -- Store embedding as a binary large object
        )
        ''')
        
        self.conn.commit()
        self.disconnect()
        
    def get_content_data(self, content_data_id=None, title=None, content=None):
        self.connect()
        
        if content_data_id:
            self.cursor.execute(
                "SELECT id, title, content FROM ContentData WHERE id = ?",
                (content_data_id,)
            )
        elif title:
            self.cursor.execute(
                "SELECT id, title, content FROM ContentData WHERE title = ?",
                (title,)
            )
        elif content:
            self.cursor.execute(
                "SELECT id, title, content FROM ContentData WHERE content = ?",
                (content,)
            )
        else:
            self.disconnect()
            return None
        
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return {
                'id': row['id'],
                'title': row['title'],
                'content': row['content']
            }
        return None
  
    def get_question_by_id_in_dataset(self, id_in_dataset):
        """
        Get a question by its original ID from the HotpotQA dataset.
        
        Args:
            id_in_dataset: ID from the HotpotQA dataset
            
        Returns:
            Question data or None if not found
        """
        self.connect()
        self.cursor.execute(
            "SELECT id, id_in_dataset, question, supporting_facts_refs, context_refs, answer, type, level, question_type FROM Question WHERE id_in_dataset = ?",
            (id_in_dataset,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return {
                'id': row['id'],
                'id_in_dataset': row['id_in_dataset'],
                'question': row['question'],
                'supporting_facts_refs': json.loads(row['supporting_facts_refs']),
                'context_refs': json.loads(row['context_refs']),
                'answer': row['answer'],
                'type': row['type'],
                'level': row['level'],
                'question_type': row['question_type']
            }
        return None
   
    def generate_embeddings_batch(self, texts):
        """Generate embeddings for multiple texts in batch."""
        embedding_tensors = self.embedding_model.generate_embeddings(
            input_texts=texts,
        )
        return embedding_tensors.cpu().numpy().tolist()
   
    def bulk_insert_hotpot_qa_data(self, hotpot_data_list, question_type='train', batch_size=2000):
        """
        Efficiently insert HotpotQA data into the database using batched transactions.
        
        Args:
            hotpot_data_list: List of HotpotQA question data objects
            question_type: Type of questions (train, dev, test)
            batch_size: Number of items to process in a single transaction
            
        Returns:
            Dictionary with statistics about inserted data
        """
        # Initialize embedding model
        self.embedding_model = MultilingualE5LargeInstruct()
        
        stats = {
            'content_items_inserted': 0,
            'questions_inserted': 0,
            'skipped_questions': 0,
            'errors': 0
        }
        
        # Pre-cache existing content and questions to minimize DB lookups
        self.connect()
        
        # Create a content cache (content text -> id)
        self.cursor.execute("SELECT id, content FROM ContentData")
        content_cache = {row['content']: row['id'] for row in self.cursor.fetchall()}
        
        # Create a question id cache (id_in_dataset -> True)
        self.cursor.execute("SELECT id_in_dataset FROM Question")
        question_id_cache = {row['id_in_dataset']: True for row in self.cursor.fetchall()}
        
        # Enable foreign keys and begin transaction
        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()
        
        # Use tqdm to visualize progress
        total_batches = len(hotpot_data_list) // batch_size + 1
        with tqdm(total=total_batches, desc="Processing Batches", unit="batch") as pbar:
            # Process in batches to manage transaction size
            for batch_start in range(0, len(hotpot_data_list), batch_size):
                batch_end = min(batch_start + batch_size, len(hotpot_data_list))
                batch = hotpot_data_list[batch_start:batch_end]
                
                try:
                    # Start transaction for this batch
                    self.conn.execute("BEGIN TRANSACTION")
                    
                    # Prepare batch insertion data
                    content_to_insert = []
                    content_insert_ids = {}
                    questions_to_insert = []
                    
                    pbar.set_postfix_str("collecting unique content")
                    # First pass: collect all unique content to insert
                    for hotpot_item in batch:
                        # Skip if question already exists
                        if hotpot_item['_id'] in question_id_cache:
                            stats['skipped_questions'] += 1
                            continue
                        
                        # Process context items and collect all sentences
                        for context_pair in hotpot_item['context']:
                            title = context_pair[0]
                            sentences = context_pair[1]
                            
                            for sentence in sentences:
                                # Skip if content already in cache
                                if sentence in content_cache:
                                    continue
                                
                                # Skip if this content is already in our batch
                                if sentence in content_insert_ids:
                                    continue
                                    
                                # Add to batch for insertion
                                content_to_insert.append((title, sentence))
                    
                    pbar.set_postfix_str("inserting unique content")
                    # Bulk insert all new content items at once
                    if content_to_insert:
                        self.cursor.executemany(
                            "INSERT INTO ContentData (title, content) VALUES (?, ?)",
                            content_to_insert
                        )
                        self.conn.commit()
                        
                        # Fetch IDs for all inserted content in one query
                        sentences = [sentence for _, sentence in content_to_insert]
                        self.cursor.execute(
                            "SELECT id, content FROM ContentData WHERE content IN ({})".format(
                                ",".join("?" * len(sentences))
                            ),
                            sentences
                        )
                        rows = self.cursor.fetchall()
                        
                        # Update caches with the retrieved IDs
                        for row in rows:
                            content_cache[row['content']] = row['id']
                            content_insert_ids[row['content']] = row['id']
                        
                        stats['content_items_inserted'] += len(content_to_insert)
                        
                        # Generate embeddings for the new content
                        content_embeddings = self.generate_embeddings_batch(sentences)

                        # Insert embeddings into ContentEmbedding table
                        self.cursor.executemany(
                            "INSERT INTO ContentEmbedding (embedding) VALUES (?)",
                            [(json.dumps(embedding),) for embedding in content_embeddings]
                        )
                        self.conn.commit()
                        
                        # Fetch the embedding IDs for the newly inserted embeddings
                        self.cursor.execute(
                            "SELECT id FROM ContentEmbedding ORDER BY id DESC LIMIT ?",
                            (len(content_embeddings),)
                        )
                        embedding_ids = [row['id'] for row in self.cursor.fetchall()]
                        
                        # Update ContentData with the corresponding embedding IDs
                        for content_id, embedding_id in zip(content_insert_ids.values(), embedding_ids):
                            self.cursor.execute(
                                "UPDATE ContentData SET embedding_id = ? WHERE id = ?",
                                (embedding_id, content_id)
                            )
                        self.conn.commit()
                    
                    pbar.set_postfix_str("preparing questions")
                    # Second pass: prepare questions with references to content
                    for hotpot_item in batch:
                        # Skip if question already exists
                        if hotpot_item['_id'] in question_id_cache:
                            continue
                        
                        contexts_refs = {}
                        supporting_facts_refs = {}
                        
                        # Build context references mapping
                        for context_pair in hotpot_item['context']:
                            title = context_pair[0]
                            sentences = context_pair[1]
                            context_refs = []
                            
                            for sentence in sentences:
                                # Get content ID from cache or newly inserted items
                                content_id = content_cache.get(sentence)
                                if content_id is not None:
                                    context_refs.append(content_id)
                            
                            contexts_refs[title] = context_refs
                        
                        # Process supporting facts
                        for fact in hotpot_item['supporting_facts']:
                            fact_title = fact[0]
                            sentence_idx = fact[1]
                            
                            if fact_title not in supporting_facts_refs:
                                supporting_facts_refs[fact_title] = []
                                
                            # Make sure the title and index are valid
                            if fact_title in contexts_refs and sentence_idx < len(contexts_refs[fact_title]):
                                supporting_facts_refs[fact_title].append(contexts_refs[fact_title][sentence_idx])
                        
                        # Add to questions batch
                        questions_to_insert.append((
                            hotpot_item['_id'],
                            hotpot_item['question'],
                            json.dumps(supporting_facts_refs),
                            json.dumps(contexts_refs),
                            hotpot_item['answer'],
                            hotpot_item.get('type', 'comparison'),
                            hotpot_item.get('level', 'medium'),
                            question_type
                        ))
                        
                        # Update question cache with new ID
                        question_id_cache[hotpot_item['_id']] = True
                    
                    pbar.set_postfix_str("inserting questions")
                    # Bulk insert all questions at once
                    if questions_to_insert:
                        self.cursor.executemany(
                            """INSERT INTO Question 
                            (id_in_dataset, question, supporting_facts_refs, context_refs, answer, type, level, question_type) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            questions_to_insert
                        )
                        
                        stats['questions_inserted'] += len(questions_to_insert)
                        
                        # Generate embeddings for the new questions
                        questions = [item[1] for item in questions_to_insert]
                        question_embeddings = self.generate_embeddings_batch(questions)
                        
                        # Insert embeddings into QuestionEmbedding table
                        self.cursor.executemany(
                            "INSERT INTO QuestionEmbedding (embedding) VALUES (?)",
                            [(json.dumps(embedding),) for embedding in question_embeddings]
                        )
                        self.conn.commit()
                        
                        # Fetch the embedding IDs for the newly inserted embeddings
                        self.cursor.execute(
                            "SELECT id FROM QuestionEmbedding ORDER BY id DESC LIMIT ?",
                            (len(question_embeddings),)
                        )
                        embedding_ids = [row['id'] for row in self.cursor.fetchall()]
                        
                        # Update Question with the corresponding embedding IDs
                        for question_id, embedding_id in zip(question_id_cache.keys(), embedding_ids):
                            self.cursor.execute(
                                "UPDATE Question SET embedding_id = ? WHERE id_in_dataset = ?",
                                (embedding_id, question_id)
                            )
                        self.conn.commit()
                    
                    
                    # Commit the transaction for this batch
                    self.conn.commit()
                    
                except Exception as e:
                    # Rollback on error
                    self.conn.rollback()
                    stats['errors'] += 1
                    print(f"Error in batch {batch_start}-{batch_end}: {e}")
                
                # Update progress bar
                pbar.update(1)
            
        # Close connection
        self.disconnect()
        
        return stats
            
    def bulk_update_content_mapping(self, mappings, batch_size=500):
        """
        Efficiently update multiple content mappings in a single transaction.
        
        Args:
            mappings: List of dictionaries with keys: content_id, sentence_id, paragraph_id, section_id, document_id
            batch_size: Number of updates to process in a single transaction
            
        Returns:
            Number of successful updates
        """
        total_updated = 0
        
        # Process in batches
        for i in range(0, len(mappings), batch_size):
            batch = mappings[i:i+batch_size]
            
            self.connect()
            self.conn.execute("BEGIN TRANSACTION")
            
            try:
                for mapping in batch:
                    content_id = mapping.get('content_id')
                    if not content_id:
                        continue
                    
                    # Build the update query dynamically based on provided values
                    update_fields = []
                    params = []
                    
                    if 'sentence_id' in mapping:
                        update_fields.append("sentence_id = ?")
                        params.append(mapping['sentence_id'])
                    if 'paragraph_id' in mapping:
                        update_fields.append("paragraph_id = ?")
                        params.append(mapping['paragraph_id'])
                    if 'section_id' in mapping:
                        update_fields.append("section_id = ?")
                        params.append(mapping['section_id'])
                    if 'document_id' in mapping:
                        update_fields.append("document_id = ?")
                        params.append(mapping['document_id'])
                    
                    if not update_fields:
                        continue
                    
                    # Add the content_id parameter
                    params.append(content_id)
                    
                    # Execute the update
                    query = f"UPDATE ContentData SET {', '.join(update_fields)} WHERE id = ?"
                    self.cursor.execute(query, params)
                    if self.cursor.rowcount > 0:
                        total_updated += 1
                
                self.conn.commit()
                
            except Exception as e:
                self.conn.rollback()
                print(f"Error updating batch: {e}")
            
            self.disconnect()
        
        return total_updated


import json
from configs import load_configs
import argparse
from hotpot_database import HotpotDatabaseHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Handle HotpotQA data into the database.")
    parser.add_argument(
        "--action", 
        type=str, 
        default="insert",  # Default to "insert" if no action is provided
        choices=[
            "insert" # add all data into the database
        ],  
        help="Database operation action to perform (default: insert)"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.action == "insert":
        config = load_configs()

        # Load HotpotQA data
        print(f"Loading data from {config['HotpotDatabase']['data_path']}...")
        with open(config['HotpotDatabase']['data_path'], 'r', encoding='utf-8') as f:
            hotpot_data = json.load(f)

        # Initialize database handler
        db_hotpot = HotpotDatabaseHandler(config['HotpotDatabase']['db_path'])

        # Insert data with optimized bulk insertion
        stats = db_hotpot.bulk_insert_hotpot_qa_data(
            hotpot_data, 
            question_type='train',
            batch_size=2000  # Adjust based on your system's memory
        )

        print(f"Inserted {stats['content_items_inserted']} content items")
        print(f"Inserted {stats['questions_inserted']} questions")
        print(f"Skipped {stats['skipped_questions']} existing questions")
        print(f"Encountered {stats['errors']} errors")
if __name__ == "__main__":
    main()