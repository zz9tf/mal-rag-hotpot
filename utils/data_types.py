# data_classes.py
class Document:
    def __init__(self, id=None, title=None, db_handler=None):
        self.id = id
        self.title = title
        self.sections = []
        self.summary = None
        self._db_handler = db_handler
        self._sections_loaded = False
        self._summary_loaded = False
    
    def add_section(self, section):
        self.sections.append(section)
        
    def get_sections(self):
        if not self._sections_loaded and self._db_handler and self.id:
            self.sections = self._db_handler.get_sections_by_document_id(self.id)
            self._sections_loaded = True
        return self.sections
    
    def get_summary(self):
        if not self._summary_loaded and self._db_handler and self.id:
            self.summary = self._db_handler.get_summary_by_document_id(self.id)
            self._summary_loaded = True
        return self.summary
    
    def set_summary(self, summary):
        self.summary = summary
        self._summary_loaded = True
    
    def get_full_text(self):
        """Get the complete document text for embedding generation"""
        text = f"Title: {self.title}\n\n"
        for section in self.get_sections():
            text += f"{section.get_full_text()}\n\n"
        return text


class Section:
    def __init__(self, id=None, title=None, document_id=None, db_handler=None):
        self.id = id
        self.title = title
        self.document_id = document_id
        self.paragraphs = []
        self.summary = None
        self._db_handler = db_handler
        self._paragraphs_loaded = False
        self._summary_loaded = False
    
    def add_paragraph(self, paragraph):
        self.paragraphs.append(paragraph)
        
    def get_paragraphs(self):
        if not self._paragraphs_loaded and self._db_handler and self.id:
            self.paragraphs = self._db_handler.get_paragraphs_by_section_id(self.id)
            self._paragraphs_loaded = True
        return self.paragraphs
    
    def get_summary(self):
        if not self._summary_loaded and self._db_handler and self.id:
            self.summary = self._db_handler.get_summary_by_section_id(self.id)
            self._summary_loaded = True
        return self.summary
    
    def set_summary(self, summary):
        self.summary = summary
        self._summary_loaded = True
    
    def get_full_text(self):
        """Get the complete section text for embedding generation"""
        text = f"Section: {self.title}\n"
        for paragraph in self.get_paragraphs():
            text += f"{paragraph.get_full_text()}\n"
        return text


class Paragraph:
    def __init__(self, id=None, section_id=None, db_handler=None):
        self.id = id
        self.section_id = section_id
        self.multi_sentences = []
        self._db_handler = db_handler
        self._sentences_loaded = False
    
    def add_multi_sentence(self, multi_sentence):
        self.multi_sentences.append(multi_sentence)
        
    def get_multi_sentences(self):
        if not self._sentences_loaded and self._db_handler and self.id:
            self.multi_sentences = self._db_handler.get_multi_sentences_by_paragraph_id(self.id)
            self._sentences_loaded = True
        return self.multi_sentences
    
    def get_full_text(self):
        """Get the complete paragraph text for embedding generation"""
        sentences = self.get_multi_sentences()
        return " ".join([s.get_content() for s in sentences])


class MultiSentence:
    def __init__(self, id=None, content=None, paragraph_id=None):
        self.id = id
        self.content = content
        self.paragraph_id = paragraph_id
    
    def get_content(self):
        return self.content


class Summary:
    def __init__(self, id=None, content=None, document_id=None, section_id=None):
        self.id = id
        self.content = content
        self.document_id = document_id
        self.section_id = section_id
    
    def get_content(self):
        return self.content