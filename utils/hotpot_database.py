# database_c.py
import sqlite3
import json

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
            content TEXT NOT NULL -- A list of sentences stored as JSON string
        )
        ''')
        
        # Create Question table with original_id column
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS Question (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT UNIQUE NOT NULL,
            question TEXT NOT NULL,
            supporting_facts TEXT NOT NULL,  -- A list of ids stored as JSON string
            context_refs TEXT NOT NULL,  -- A list of ids stored as JSON string
            answer TEXT NOT NULL,
            type TEXT NOT NULL
        )
        ''')
        
        self.conn.commit()
        self.disconnect()
    
    def insert_content_data(self, title, content):
        self.connect()
        self.cursor.execute(
            "INSERT INTO ContentData (title, content) VALUES (?, ?)",
            (title, content)
        )
        content_data_id = self.cursor.lastrowid
        self.conn.commit()
        self.disconnect()
        return content_data_id
    
    def get_content_data(self, content_data_id=None, title=None):
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
    
    def insert_question_with_id(self, original_id, question, supporting_facts, context_refs, answer, question_type):
        """
        Insert a HotpotQA question with its original ID.
        
        Args:
            original_id: Original question ID from the HotpotQA dataset
            question: Question text
            supporting_facts: List of supporting facts
            context_refs: List of context references
            answer: Answer text
            question_type: Question type
            
        Returns:
            Database ID of the inserted question
        """
        self.connect()
        self.cursor.execute(
            "INSERT INTO Question (id, question, supporting_facts, context_refs, answer, type) VALUES (?, ?, ?, ?, ?, ?)",
            (original_id, question, json.dumps(supporting_facts), json.dumps(context_refs), answer, question_type)
        )
        question_id = self.cursor.lastrowid
        self.conn.commit()
        self.disconnect()
        return question_id
    
    def get_question_by_original_id(self, original_id):
        """
        Get a question by its original ID from the HotpotQA dataset.
        
        Args:
            original_id: Original ID from the HotpotQA dataset
            
        Returns:
            Question data or None if not found
        """
        self.connect()
        self.cursor.execute(
            "SELECT id, original_id, question, supporting_facts, context, answer, type FROM Question WHERE original_id = ?",
            (original_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return {
                'id': row['id'],
                'original_id': row['original_id'],
                'question': row['question'],
                'supporting_facts': json.loads(row['supporting_facts']),
                'context': json.loads(row['context']),
                'answer': row['answer'],
                'type': row['type']
            }
        return None
    
    def get_question(self, question_id):
        self.connect()
        self.cursor.execute(
            "SELECT id, question, supporting_facts, context, answer, type FROM Question WHERE id = ?",
            (question_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return {
                'id': row['id'],
                'question': row['question'],
                'supporting_facts': json.loads(row['supporting_facts']),
                'context': json.loads(row['context']),
                'answer': row['answer'],
                'type': row['type']
            }
        return None