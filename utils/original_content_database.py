# database_a.py
import sqlite3
from data_types import Document, Section, Paragraph, MultiSentence, Summary

class OriginalContentDatabaseHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_db()
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
    
    def initialize_db(self):
        self.connect()
        
        # Create Document table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS Document (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary_id INTEGER,
            FOREIGN KEY (summary_id) REFERENCES Summary(id)
        )
        ''')
        
        # Create Section table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS Section (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            document_id INTEGER NOT NULL,
            summary_id INTEGER,
            FOREIGN KEY (document_id) REFERENCES Document(id),
            FOREIGN KEY (summary_id) REFERENCES Summary(id)
        )
        ''')
        
        # Create Paragraph table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS Paragraph (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section_id INTEGER NOT NULL,
            FOREIGN KEY (section_id) REFERENCES Section(id)
        )
        ''')
        
        # Create MultiSentence table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS MultiSentence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paragraph_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (paragraph_id) REFERENCES Paragraph(id)
        )
        ''')
        
        # Create Summary table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS Summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            section_id INTEGER,
            content TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES Document(id),
            FOREIGN KEY (section_id) REFERENCES Section(id)
        )
        ''')
        
        self.conn.commit()
        self.disconnect()
    
    def insert_document(self, document):
        self.connect()
        self.cursor.execute(
            "INSERT INTO Document (title) VALUES (?)",
            (document.title,)
        )
        document_id = self.cursor.lastrowid
        self.conn.commit()
        self.disconnect()
        return document_id
    
    def get_document(self, document_id):
        self.connect()
        self.cursor.execute(
            "SELECT id, title, summary_id FROM Document WHERE id = ?",
            (document_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            document = Document(id=row[0], title=row[1])
            
            # Get sections
            sections = self.get_sections_by_document_id(document_id)
            for section in sections:
                document.add_section(section)
            
            # Get summary if exists
            if row[2]:
                document.set_summary(self.get_summary(row[2]))
            
            return document
        return None
    
    def insert_section(self, section):
        self.connect()
        self.cursor.execute(
            "INSERT INTO Section (title, document_id) VALUES (?, ?)",
            (section.title, section.document_id)
        )
        section_id = self.cursor.lastrowid
        self.conn.commit()
        self.disconnect()
        return section_id
    
    def get_section(self, section_id):
        self.connect()
        self.cursor.execute(
            "SELECT id, title, document_id, summary_id FROM Section WHERE id = ?",
            (section_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            section = Section(id=row[0], title=row[1], document_id=row[2])
            
            # Get paragraphs
            paragraphs = self.get_paragraphs_by_section_id(section_id)
            for paragraph in paragraphs:
                section.add_paragraph(paragraph)
            
            # Get summary if exists
            if row[3]:
                section.set_summary(self.get_summary(row[3]))
            
            return section
        return None
    
    def get_sections_by_document_id(self, document_id):
        self.connect()
        self.cursor.execute(
            "SELECT id FROM Section WHERE document_id = ?",
            (document_id,)
        )
        rows = self.cursor.fetchall()
        self.disconnect()
        
        sections = []
        for row in rows:
            section = self.get_section(row[0])
            sections.append(section)
        
        return sections
    
    def insert_paragraph(self, paragraph):
        self.connect()
        self.cursor.execute(
            "INSERT INTO Paragraph (section_id) VALUES (?)",
            (paragraph.section_id,)
        )
        paragraph_id = self.cursor.lastrowid
        self.conn.commit()
        self.disconnect()
        return paragraph_id
    
    def get_paragraph(self, paragraph_id):
        self.connect()
        self.cursor.execute(
            "SELECT id, section_id FROM Paragraph WHERE id = ?",
            (paragraph_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            paragraph = Paragraph(id=row[0], section_id=row[1])
            
            # Get multi-sentences
            multi_sentences = self.get_multi_sentences_by_paragraph_id(paragraph_id)
            for multi_sentence in multi_sentences:
                paragraph.add_multi_sentence(multi_sentence)
            
            return paragraph
        return None
    
    def get_paragraphs_by_section_id(self, section_id):
        self.connect()
        self.cursor.execute(
            "SELECT id FROM Paragraph WHERE section_id = ?",
            (section_id,)
        )
        rows = self.cursor.fetchall()
        self.disconnect()
        
        paragraphs = []
        for row in rows:
            paragraph = self.get_paragraph(row[0])
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def insert_multi_sentence(self, multi_sentence):
        self.connect()
        self.cursor.execute(
            "INSERT INTO MultiSentence (paragraph_id, content) VALUES (?, ?)",
            (multi_sentence.paragraph_id, multi_sentence.content)
        )
        multi_sentence_id = self.cursor.lastrowid
        self.conn.commit()
        self.disconnect()
        return multi_sentence_id
    
    def get_multi_sentence(self, multi_sentence_id):
        self.connect()
        self.cursor.execute(
            "SELECT id, paragraph_id, content FROM MultiSentence WHERE id = ?",
            (multi_sentence_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return MultiSentence(id=row[0], paragraph_id=row[1], content=row[2])
        return None
    
    def get_multi_sentences_by_paragraph_id(self, paragraph_id):
        self.connect()
        self.cursor.execute(
            "SELECT id FROM MultiSentence WHERE paragraph_id = ?",
            (paragraph_id,)
        )
        rows = self.cursor.fetchall()
        self.disconnect()
        
        multi_sentences = []
        for row in rows:
            multi_sentence = self.get_multi_sentence(row[0])
            multi_sentences.append(multi_sentence)
        
        return multi_sentences
    
    def insert_summary(self, summary):
        self.connect()
        self.cursor.execute(
            "INSERT INTO Summary (document_id, section_id, content) VALUES (?, ?, ?)",
            (summary.document_id, summary.section_id, summary.content)
        )
        summary_id = self.cursor.lastrowid
        self.conn.commit()
        
        # Update document or section with summary_id
        if summary.document_id:
            self.cursor.execute(
                "UPDATE Document SET summary_id = ? WHERE id = ?",
                (summary_id, summary.document_id)
            )
        elif summary.section_id:
            self.cursor.execute(
                "UPDATE Section SET summary_id = ? WHERE id = ?",
                (summary_id, summary.section_id)
            )
        
        self.conn.commit()
        self.disconnect()
        return summary_id
    
    def get_summary(self, summary_id):
        self.connect()
        self.cursor.execute(
            "SELECT id, document_id, section_id, content FROM Summary WHERE id = ?",
            (summary_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return Summary(id=row[0], document_id=row[1], section_id=row[2], content=row[3])
        return None

    def get_summary_by_document_id(self, document_id):
        self.connect()
        self.cursor.execute(
            "SELECT id FROM Summary WHERE document_id = ?",
            (document_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return self.get_summary(row[0])
        return None

    def get_summary_by_section_id(self, section_id):
        self.connect()
        self.cursor.execute(
            "SELECT id FROM Summary WHERE section_id = ?",
            (section_id,)
        )
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return self.get_summary(row[0])
        return None
    
    def get_summary_target(self, summary_id):
        """Get the target (Document or Section) that this summary belongs to.
        Returns a tuple of (target_type, target) where target_type is either 'document' or 'section'
        and target is the corresponding Document or Section object."""
        
        summary = self.get_summary(summary_id)
        if not summary:
            return None
            
        if summary.document_id:
            self.connect()
            self.cursor.execute(
                "SELECT id, title FROM Document WHERE id = ?",
                (summary.document_id,)
            )
            row = self.cursor.fetchone()
            self.disconnect()
            if row:
                return ('document', Document(id=row[0], title=row[1], db_handler=self))
        elif summary.section_id:
            self.connect() 
            self.cursor.execute(
                "SELECT id, title, document_id FROM Section WHERE id = ?",
                (summary.section_id,)
            )
            row = self.cursor.fetchone()
            self.disconnect()
            if row:
                return ('section', Section(id=row[0], title=row[1], document_id=row[2], db_handler=self))
                
        return None
    
    # Add to OriginalContentDatabaseHandler
    def begin_transaction(self):
        """Begin a database transaction."""
        self.connect()
        self.conn.execute("BEGIN TRANSACTION")

    def commit_transaction(self):
        """Commit the current transaction."""
        self.conn.commit()
        self.disconnect()

    def rollback_transaction(self):
        """Roll back the current transaction."""
        self.conn.rollback()
        self.disconnect()

    def bulk_insert_documents(self, documents):
        """
        Insert multiple documents in a single transaction.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of inserted document IDs
        """
        document_data = [(doc.title,) for doc in documents]
    
        self.cursor.execute("SELECT MAX(id) FROM Document")
        max_id_before = self.cursor.fetchone()[0] or 0
        
        self.cursor.executemany(
            "INSERT INTO Document (title) VALUES (?)",
            document_data
        )
        
        self.cursor.execute("SELECT MAX(id) FROM Document")
        max_id_after = self.cursor.fetchone()[0]
        ids = list(range(max_id_before + 1, max_id_after + 1))

        return ids

    def bulk_insert_sections(self, section_data):
        """
        Insert multiple sections in a single transaction.
        
        Args:
            section_data: List of tuples (Section object, document_id)
            
        Returns:
            List of inserted section IDs
        """
        data = [(section.title, document_id) for section, document_id in section_data]
        
        self.cursor.execute("SELECT MAX(id) FROM Section")
        max_id_before = self.cursor.fetchone()[0] or 0
        
        self.cursor.executemany(
            "INSERT INTO Section (title, document_id) VALUES (?, ?)",
            data
        )
        
        self.cursor.execute("SELECT MAX(id) FROM Section")
        max_id_after = self.cursor.fetchone()[0]
        ids = list(range(max_id_before + 1, max_id_after + 1))
        
        return ids
    
    def bulk_insert_paragraphs(self, paragraph_data):
        """
            Insert multiple paragraphs in a single transaction.
            
            Args:
                paragraph_data: List of tuples (Paragraph object, section_id)
                
            Returns:
                List of inserted paragraph IDs
        """
        data = [(section_id,) for _, section_id in paragraph_data]
        
        self.cursor.execute("SELECT MAX(id) FROM Paragraph")
        max_id_before = self.cursor.fetchone()[0] or 0
        
        self.cursor.executemany(
            "INSERT INTO Paragraph (section_id) VALUES (?)",
            data
        )
        
        self.cursor.execute("SELECT MAX(id) FROM Paragraph")
        max_id_after = self.cursor.fetchone()[0]
        ids = list(range(max_id_before + 1, max_id_after + 1))
        
        return ids

    def bulk_insert_sentences(self, sentences):
        """
        Insert multiple sentences in a single transaction and return their IDs.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of inserted document IDs
        """
        sentence_data = [(sentence.content, paragraph_id) for sentence, paragraph_id in sentences]
        
        self.cursor.execute("SELECT MAX(id) FROM MultiSentence")
        max_id_before = self.cursor.fetchone()[0] or 0
        
        self.cursor.executemany(
            "INSERT INTO MultiSentence (content, paragraph_id) VALUES (?, ?)",
            sentence_data
        )
        
        self.cursor.execute("SELECT MAX(id) FROM MultiSentence")
        max_id_after = self.cursor.fetchone()[0]
        ids = list(range(max_id_before + 1, max_id_after + 1))
        
        return ids

def main():
    from configs import load_configs
    
    config = load_configs()
    
    # Path to the SQLite database
    db_path = config['OriginalContentDatabase']['db_path']

    # Initialize the database handler
    db_handler = OriginalContentDatabaseHandler(db_path)

    # Create a new document
    document = Document(title="Sample Document")
    document_id = db_handler.insert_document(document)
    print(f"Inserted Document with ID: {document_id}")

    # Create a new section for the document
    section = Section(title="Introduction", document_id=document_id)
    section_id = db_handler.insert_section(section)
    print(f"Inserted Section with ID: {section_id}")

    # Create a new paragraph for the section
    paragraph = Paragraph(section_id=section_id)
    paragraph_id = db_handler.insert_paragraph(paragraph)
    print(f"Inserted Paragraph with ID: {paragraph_id}")

    # Create a new multi-sentence for the paragraph
    multi_sentence = MultiSentence(paragraph_id=paragraph_id, content="This is the first sentence. This is the second sentence.")
    multi_sentence_id = db_handler.insert_multi_sentence(multi_sentence)
    print(f"Inserted MultiSentence with ID: {multi_sentence_id}")

    # Create a summary for the document
    document_summary = Summary(document_id=document_id, content="This is a summary of the document.")
    document_summary_id = db_handler.insert_summary(document_summary)
    print(f"Inserted Document Summary with ID: {document_summary_id}")

    # Create a summary for the section
    section_summary = Summary(section_id=section_id, content="This is a summary of the section.")
    section_summary_id = db_handler.insert_summary(section_summary)
    print(f"Inserted Section Summary with ID: {section_summary_id}")

    # Retrieve the document and print its details
    retrieved_document = db_handler.get_document(document_id)
    if retrieved_document:
        print("\nRetrieved Document:")
        print(f"ID: {retrieved_document.id}")
        print(f"Title: {retrieved_document.title}")
        print(f"Summary: {retrieved_document.summary.content if retrieved_document.summary else 'No summary'}")

        # Print sections of the document
        print("\nSections:")
        for section in retrieved_document.sections:
            print(f"  Section ID: {section.id}")
            print(f"  Section Title: {section.title}")
            print(f"  Section Summary: {section.summary.content if section.summary else 'No summary'}")

            # Print paragraphs of the section
            print("\n  Paragraphs:")
            for paragraph in section.paragraphs:
                print(f"    Paragraph ID: {paragraph.id}")

                # Print multi-sentences of the paragraph
                print("\n    Multi-Sentences:")
                for multi_sentence in paragraph.multi_sentences:
                    print(f"      Multi-Sentence ID: {multi_sentence.id}")
                    print(f"      Content: {multi_sentence.content}")

    # Retrieve the summary target (document or section)
    summary_target = db_handler.get_summary_target(document_summary_id)
    if summary_target:
        target_type, target = summary_target
        print(f"\nSummary Target (Document): {target.title}")

    summary_target = db_handler.get_summary_target(section_summary_id)
    if summary_target:
        target_type, target = summary_target
        print(f"\nSummary Target (Section): {target.title}")

if __name__ == "__main__":
    main()