import os
import re
import json
import html
import argparse
import numpy as np
from tqdm import tqdm
from lxml import etree
from langdetect import detect
from xml.etree.ElementTree import XMLPullParser
from multiprocessing.pool import ThreadPool
from configs import load_configs
from generate_and_execute_slurm_job import generate_and_execute_slurm_job
from data_types import Document, Section, Paragraph, MultiSentence
from original_content_database import OriginalContentDatabaseHandler
from embedding_database import EmbeddingDatabaseHandler
from embedding_model import MultilingualE5LargeInstruct
from llama_index.core.node_parser import SentenceSplitter
import traceback

class WikipediaDumpProcessor:
    def __init__(
        self,
        input_file_path,
        cache_dir,
        db_original_content_path,
        db_embedding_path,
        workers=5,
        size_per_batch=5000000000, # 5GB
        device=None
    ):
        self.input_file_path = input_file_path
        self.cache_dir = cache_dir
        self.workers = workers
        self.size_per_batch = size_per_batch
        
        # Initialize database handlers
        self.db_original_content = OriginalContentDatabaseHandler(db_original_content_path)
        self.db_embedding = EmbeddingDatabaseHandler(db_embedding_path)
        
        # Initialize embedding model
        self.embedding_model = MultilingualE5LargeInstruct(device=device)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    # ----- XML PARSING METHODS -----
    def _processed_xml(self):
        """Check if XML has already been processed."""
        for filename in os.listdir(self.cache_dir):
            if "raw_page" in filename:
                return True
        return False
    
    def parse_xml_dump(self, verbose=True):
        """Parse the Wikipedia XML dump into individual pages."""
        # if self._processed_xml():
        #     print("XML already processed, skipping parsing step.")
        #     return
        
        print(f"Parsing Wikipedia XML dump: {self.input_file_path}")
        total_file_size = os.path.getsize(self.input_file_path)
        parser = XMLPullParser(events=('start', 'end'))
        raw_page = None
        end_page = False
        page_num = 0
        batch_id = 0
        current_batch = []
        all_batches = []
        update_len = 0
        
        # Load pages in batches
        with open(self.input_file_path, 'rb') as input_file:
            with tqdm(total=total_file_size, desc="Processing XML", unit="B", unit_scale=True) as pbar:
                for line_bytes in input_file:
                    line = line_bytes.decode('utf-8')  # Decode line
                    parser.feed(line)
                    for event, element in parser.read_events():
                        if event == 'start' and 'page' in element.tag:
                            raw_page = []
                        if event == 'end' and 'page' in element.tag:
                            end_page = True
                        element.clear()
                    
                    # Collect page content
                    if isinstance(raw_page, list):
                        raw_page.append(line)
                    
                    # Process completed page
                    if end_page:
                        raw_page = ''.join(raw_page)
                        current_batch.append(raw_page)
                        page_num += 1
                        raw_page = None
                        end_page = False
                    
                    # Update progress
                    update_len += len(line_bytes)
                    if update_len > self.size_per_batch:
                        # Add to processing queue
                        all_batches.append([batch_id, current_batch])
                        if verbose:
                            print(f"\nBatch {batch_id} has {[batch_id for batch_id, _ in all_batches]}")
                            print(f"len(current_batch): {len(current_batch)}")
                            print(f"update_len: {update_len}")
                        batch_id += 1
                        current_batch = []
                        
                        pbar.set_postfix_str(f"Pages: {page_num}")
                        pbar.update(update_len)
                        
                        update_len = 0
                    
                    # Process batches in parallel when we have enough
                    if len(all_batches) >= self.workers:
                        self._process_batches(all_batches)
                        all_batches = []
        
                # Process any remaining pages
                if current_batch:
                    all_batches.append([batch_id, current_batch])
                    pbar.set_postfix_str(f"Pages: {page_num}")
                    pbar.update(update_len)
        
        if all_batches:
            self._process_batches(all_batches)
    
    def _process_batches(self, batches):
        """Process batches of raw XML pages in parallel."""
        with ThreadPool(self.workers) as pool:
            for _ in pool.imap(lambda x: self._write_batch_to_disk(x[0], x[1]), batches):
                pass
    
    def _write_batch_to_disk(self, batch_id, batch):
        """Write a batch of raw pages to disk."""
        cache_file_path = os.path.join(self.cache_dir, f'raw_page_{batch_id}_not_finish.jsonl')
        with open(cache_file_path, 'w') as cache_file:
            for page in batch:
                cache_file.write(json.dumps({'page': page}) + '\n')
        # Rename to final filename when complete
        os.rename(cache_file_path, os.path.join(self.cache_dir, f'raw_page_{batch_id}.jsonl'))
    
    # ----- TEXT CLEANING METHODS -----
    
    def _clean_text(self, raw_text):
        """Clean Wikipedia markup from text."""
        if not raw_text:
            return ""
            
        cleaned_text = html.unescape(raw_text)
        
        # Remove <ref> tag without greedy
        cleaned_text = self._remove_self_closing_ref(cleaned_text)
        # Remove <ref /> tag without greedy
        cleaned_text = self._remove_ref_with_content(cleaned_text)
        
        # Remove { }
        cleaned_text = self._remove_braces_content(cleaned_text)
        # Remove <!-- ... -->
        cleaned_text = re.sub(r'<!--.*?-->', '', cleaned_text, flags=re.DOTALL)  # Comments
        
        # Remove ''' '''
        cleaned_text = re.sub(
            r"('{3,})(.*?)(\1)",  # Bold/italic formatting
            lambda m: m.group(2) if len(m.group(1)) == len(m.group(3)) else m.group(0),
            cleaned_text
        )
        
        # Remove [[ | target]] -> target
        cleaned_text = self._remove_square_brackets(cleaned_text)
        
        # Remove remaining HTML tags
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        
        # Clean up whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def _remove_self_closing_ref(self, text):
        """Remove <ref /> self-closing tags."""
        result = []
        i = 0
        
        while i < len(text):
            if text[i:i+4] == '<ref' and '/>' in text[i+4:]:
                end_idx = text.find('>', i)
                if end_idx > 0 and text[end_idx-1] == '/':
                    i = end_idx + 1  # Skip the entire tag
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)

    def _remove_ref_with_content(self, text):
        """Remove <ref>...</ref> tags and their content."""
        stack = []
        result = []
        i = 0
        
        while i < len(text):
            if text[i:i+4] == '<ref' and '>' in text[i+4:]:
                stack.append(i)
                i += 4
            elif text[i:i+6] == '</ref>' and stack:
                stack.pop()
                i += 6
            elif not stack:
                result.append(text[i])
                i += 1
            else:
                i += 1  # Skip characters within <ref> tags
        
        return ''.join(result)

    def _remove_braces_content(self, text):
        """Remove {{...}} templates, preserving some useful content."""
        stack = []
        result = []
        i = 0

        while i < len(text):
            char = text[i]
            if char == '{' and i + 1 < len(text) and text[i + 1] == '{':
                stack.append(len(result))
                i += 2
            elif char == '}' and i + 1 < len(text) and text[i + 1] == '}' and stack:
                start = stack.pop()
                block_content = ''.join(result[start:])
                if '|' in block_content and block_content.startswith('Blockquote'):
                    # Keep content after the pipe for blockquotes
                    parts = block_content.split('|', 1)
                    if len(parts) > 1:
                        result = result[:start] + list(parts[1])
                else:
                    result = result[:start]  # Remove the entire template
                i += 2
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)

    def _remove_square_brackets(self, text):
        """Process [[...]] wiki links, keeping display text."""
        square_stack = []
        result = []
        i = 0

        while i < len(text):
            char = text[i]
            if char == '[' and i + 1 < len(text) and text[i + 1] == '[':
                square_stack.append(len(result))
                i += 2
            elif char == ']' and i + 1 < len(text) and text[i + 1] == ']' and square_stack:
                start = square_stack.pop()
                link_content = ''.join(result[start:])
                
                # For links with display text: [[target|display]] -> display
                if '|' in link_content:
                    display_text = link_content.split('|', 1)[1]
                    result = result[:start] + list(display_text)
                # For simple links: [[target]] -> target
                else:
                    result = result[:start] + list(link_content)
                    
                i += 2
            else:
                result.append(char)
                i += 1

        return ''.join(result)
    
    # ----- DOCUMENT PARSING METHODS -----
    
    def extract_page_content(self, xml_data):
        """Extract title and text content from a Wikipedia page XML."""
        try:
            root = etree.fromstring(xml_data)
            
            # Skip redirect pages
            if root.find('redirect') is not None:
                return None
                
            title = root.findtext('title')
            text_element = root.find(".//text")
            
            # Skip pages without text
            if text_element is None or text_element.text is None:
                return None
                
            text = text_element.text.strip()
            
            # Skip non-article namespaces
            if title and (title.lower().startswith('file:') or 
                         title.lower().startswith('category:') or
                         title.lower().startswith('template:') or
                         title.lower().startswith('wikipedia:') or
                         title.lower().startswith('help:')):
                return None
                
            return {'title': title, 'text': text}
            
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return None
    
    def split_into_sections(self, page_text):
        """Split Wikipedia page text into sections."""
        # Split on section headings (== Section ==)
        raw_sections = re.split(r'(?m)^==\s*', page_text)
        section_title = None
        current_section = ""
        
        sections = []
        
        # First item is the lead section (before any headings)
        lead_section = self._clean_text(raw_sections[0])
        
        section_title = 'abstract'
        if len(lead_section) > 0:
            current_section = f'{lead_section}\n'
        should_skip = False
        
        for section in raw_sections[1:]:
            # Process body
            lines = section.split('\n')
            small_section_title = lines[0].strip('= ')
            section_content = self._clean_text("\n".join(lines[1:]))
            if not section.startswith('='):
                # Update accumulated contents
                if len(current_section.strip()) > 0:
                    sections.append({
                        'title': section_title,
                        'content': current_section
                    })
                    current_section = ""
                # Skip the content when the title is not very useful
               # Skip unwanted sections
                if small_section_title.lower() in ['references', 'external links', 'see also', 
                                           'further reading', 'notes', 'bibliography']:
                    should_skip = True
                else:
                    should_skip = False
                    # update status
                    section_title = small_section_title
            elif not should_skip:
                assert section_title is not None, 'prev_section_title shouldn\'t be None'
                if small_section_title.strip() == '':
                    current_section += f'{section_content}\n'
                else:
                    current_section += f'subtitle: {small_section_title}\n{section_content}\n'
        return sections
    
    def split_into_paragraphs(self, section_content):
        """Split section content into paragraphs."""
        paragraph_texts = re.split(r'\n', section_content)
        
        big_paragraphs = []  # List to store all big paragraphs
        current_big_paragraph = ""  # Current big paragraph being constructed
        current_subtitle = None  # Current subtitle for the big paragraph
        
        # Separate subtitles from other paragraphs
        for para in paragraph_texts:
            if para.startswith("subtitle:") and len(current_big_paragraph) > 0:
                big_paragraphs.append(current_big_paragraph)
                current_big_paragraph = ""
            elif len(current_big_paragraph) > 500:
                big_paragraphs.append(current_big_paragraph)
                current_big_paragraph = ""
            else:
                current_big_paragraph += f"{para}\n"
        
        return big_paragraphs
    
    def init_sentence_splitter(self):
        self.sentence_splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
    
    def split_into_sentences(self, paragraph_content):
        """Split paragraph content into sentences."""
        # Basic sentence splitting - can be improved with NLP libraries
        sentences = self.sentence_splitter.split_text(paragraph_content)
        
        return sentences
    
    # ----- PROCESSING METHODS -----
    def process_batch(self, batch_file, batch_id):
        """Process a single batch file and store in database."""
        pages = []
        with open(batch_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    pages.append(data['page'])
                except json.JSONDecodeError:
                    continue
        
        docs_added = 0
        english_docs = 0
        
        with tqdm(total=len(pages), desc=f"Batch {batch_id}", leave=False) as pbar:
            extracted_pages = [] # remove later
            for raw_page in pages:
                # Extract page content
                page = self.extract_page_content(raw_page)
                if not page or len(page['text']) < 250:  # Skip very short pages
                    pbar.update(1)
                    continue
                
                # Check if it's English (sample first 1000 chars)
                try:
                    is_english = detect(page['text'][:1000]) == 'en'
                except:
                    is_english = False
                
                if not is_english:
                    pbar.update(1)
                    continue
                
                # Process and store the document
                document_id = self.process_and_store_document(page)
                if document_id:
                    docs_added += 1
                    english_docs += 1
                
                pbar.update(1)
        
        # Mark the batch as processed
        processed_marker = os.path.join(self.cache_dir, f'processed_{batch_id}.done')
        with open(processed_marker, 'w') as f:
            f.write(f"Processed {docs_added} documents, {english_docs} in English")
        
        return docs_added, english_docs
    
    def process_and_store_document(self, page, is_store=True, is_embedding=True):
        """Process a Wikipedia page and store in database with embeddings."""
        # try:
        # Create the document
        document = Document(title=page['title'])
        
        # Split into sections
        sections_data = self.split_into_sections(page['text'])
        
        # Store the document
        if is_store:
            document.id = self.db_original_content.insert_document(document)
        
        # Embed the document
        if is_embedding:
            sections_content = [section['content'] for section in sections_data]
            document_embeddings = self.generate_embeddings_batch("\n".join(sections_content))
            self.db_embedding.insert_embedding(
                f"document_{document.id}",
                document_embeddings,
                metadata={
                    'type': 'document'
                }
            )
        
        # Process each section
        for section_data in sections_data:
            # Create section
            section = Section(title=section_data['title'], document_id=document.id)
            if is_store:
                section.id = self.db_original_content.insert_section(section)
            if is_embedding:
                section_embeddings = self.generate_embeddings_batch(section_data['content'])
                self.db_embedding.insert_embedding(
                    f"section_{section.id}",
                    section_embeddings,
                    metadata={
                        'type': 'section'
                    }
                )
            # Split into paragraphs
            paragraphs_data = self.split_into_paragraphs(section_data['content'])
            
            # Process each paragraph
            for paragraph_data in paragraphs_data:
                # Create paragraph
                paragraph = Paragraph(section_id=section.id)
                if is_store:
                    paragraph.id = self.db_original_content.insert_paragraph(paragraph)
                if is_embedding:
                    paragraph_embeddings = self.generate_embeddings_batch(paragraph_data)
                    self.db_embedding.insert_embedding(
                        f"paragraph_{paragraph.id}",
                        paragraph_embeddings,
                        metadata={
                            'type': 'paragraph'
                        }
                    )
                # Split into sentences
                sentences = self.split_into_sentences(paragraph_data)
                
                # Skip empty paragraphs
                if not sentences:
                    continue
                
                # Generate and store sentence embeddings in batch
                sentence_embeddings = self.generate_embeddings_batch(sentences)
                for i, multi_sentence in enumerate(sentences):
                    self.db_embedding.insert_embedding(
                        f"sentence_{multi_sentence.id}",
                        sentence_embeddings[i],
                        metadata={
                            'type': 'sentence'
                        }
                    )
        
        return document.id
    
    # ----- EMBEDDING METHODS -----
    
    def generate_embedding(self, text):
        """Generate embedding for a single text."""
        embedding_tensor = self.embedding_model.generate_embedding(
            input_text=text,
        )
        return embedding_tensor.cpu().numpy().tolist()
    
    def generate_embeddings_batch(self, texts):
        """Generate embeddings for multiple texts in batch."""
        embedding_tensors = self.embedding_model.generate_embeddings(
            input_texts=texts,
        )
        return embedding_tensors

    # ----- BATCH PROCESSING METHODS -----
    def process_batch_with_batch_mode(self, batch_file, batch_id, batch_size=2000, embedding_batch_size=100):
        """
        Process a batch of Wikipedia pages with optimized batch mode.
        
        Args:
            batch_file: Path to the batch file containing Wikipedia pages
            batch_id: ID of the batch for tracking
            batch_size: Number of pages to process in a single transaction
            
        Returns:
            Tuple of (total documents added, English documents added)
        """
        # Load pages from batch file
        pages = []
        with open(batch_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    pages.append(data['page'])
                except json.JSONDecodeError:
                    continue
        
        # Filter relevant pages and extract content
        pages = pages # TODO: remove this line later
        filtered_pages = []
        with tqdm(total=len(pages), desc=f"Filtering Batch {batch_id}", leave=False) as pbar:
            for raw_page in pages:
                # Extract page content
                page = self.extract_page_content(raw_page)
                if not page or len(page['text']) < 250:  # Skip very short pages
                    pbar.update(1)
                    continue
                
                # Check if it's English (sample first 1000 chars)
                try:
                    is_english = detect(page['text'][:1000]) == 'en'
                except:
                    is_english = False
                
                if not is_english:
                    pbar.update(1)
                    continue
                
                # Add to filtered pages
                filtered_pages.append(page)
                pbar.update(1)
        
        # Process the filtered pages using batch processing
        print(f"Processing {len(filtered_pages)} English pages in batch {batch_id}")
        document_ids = self.process_wikipedia_pages(filtered_pages, batch_size=batch_size, embedding_batch_size=embedding_batch_size)
        
        # Mark the batch as processed
        processed_marker = os.path.join(self.cache_dir, f'processed_{batch_id}.done')
        with open(processed_marker, 'w') as f:
            f.write(f"Processed {len(document_ids)} documents out of {len(filtered_pages)} English pages")
        
        return len(document_ids), len(filtered_pages)
    
    def process_wikipedia_pages(self, pages, batch_size=2000, embedding_batch_size=100):
        """
        Process multiple Wikipedia pages in batches for improved performance.
        
        Args:
            pages: List of page dictionaries with 'title' and 'text' keys
            batch_size: Number of pages to process in a single transaction
            
        Returns:
            List of document IDs created
        """
        document_ids = []
        
        # Configure database for maximum bulk performance
        try:
            self.db_original_content.optimize_for_bulk_insert()
        except AttributeError:
            print("Warning: Database handler does not support bulk optimizations")
        
        # Process in batches
        for batch_start in range(0, len(pages), batch_size):
            batch_end = min(batch_start + batch_size, len(pages))
            batch = pages[batch_start:batch_end]
            
            # Begin transaction for the entire batch
            self.db_original_content.begin_transaction()
            
            # First pass: Create all documents, sections, paragraphs, and sentences
            documents_batch = []
            sections_batch = []
            paragraphs_batch = []
            sentences_batch = []
            
            # Process each page in the batch
            for page in tqdm(batch, desc=f"Preparing pages {batch_start}-{batch_end}", leave=False):
                # Create document
                document = Document(title=page['title'])
                
                # Split into sections
                sections_data = self.split_into_sections(page['text'])
                document_text = "\n".join([section['content'] for section in sections_data])
                
                # Add to document batch
                documents_batch.append((document, document_text))
                
                # Process each section
                for section_data in sections_data:
                    section = Section(title=section_data['title'])
                    sections_batch.append((document, section, section_data['content']))
                    
                    # Split into paragraphs
                    paragraphs_data = self.split_into_paragraphs(section_data['content'])
                    
                    # Process each paragraph
                    for paragraph_text in paragraphs_data:
                        paragraph = Paragraph()
                        paragraphs_batch.append((section, paragraph, paragraph_text))
                        
                        # Split into sentences
                        sentence_texts = self.split_into_sentences(paragraph_text)
                        
                        # Process each sentence
                        for sentence_text in sentence_texts:
                            sentence = MultiSentence(content=sentence_text)
                            sentences_batch.append((paragraph, sentence, sentence_text))
            
            print(f"Batch statistics: {len(documents_batch)} documents, {len(sections_batch)} sections, " +
                    f"{len(paragraphs_batch)} paragraphs, {len(sentences_batch)} sentences")
            # Bulk insert all documents
            print("Inserting documents...")
            document_ids_batch = self.db_original_content.bulk_insert_documents([
                document for document, _ in documents_batch
            ])
            
            # Update document objects with IDs
            for i, doc_id in enumerate(document_ids_batch):
                documents_batch[i][0].id = doc_id
            
            # Insert sections with document IDs
            print("Inserting sections...")
            section_ids = self.db_original_content.bulk_insert_sections([
                (section, document.id) for document, section, _ in sections_batch
            ])
            
            # Update section objects with IDs
            for i, section_id in enumerate(section_ids):
                sections_batch[i][1].id = section_id
                # Also set document_id for later use
                sections_batch[i][1].document_id = sections_batch[i][0].id
            
            # Insert paragraphs with section IDs
            print("Inserting paragraphs...")
            paragraph_ids = self.db_original_content.bulk_insert_paragraphs([
                (paragraph, section.id) for section, paragraph, _ in paragraphs_batch
            ])
            
            # Update paragraph objects with IDs
            for i, paragraph_id in enumerate(paragraph_ids):
                paragraphs_batch[i][1].id = paragraph_id
                # Also set section_id for later use
                paragraphs_batch[i][1].section_id = paragraphs_batch[i][0].id
            
            # Insert sentences with paragraph IDs
            print("Inserting sentences...")
            sentence_ids = self.db_original_content.bulk_insert_sentences([
                (sentence, paragraph.id) for paragraph, sentence, _ in sentences_batch
            ])
            
            # Update sentence objects with IDs
            for i, sentence_id in enumerate(sentence_ids):
                sentences_batch[i][1].id = sentence_id
                # Also set paragraph_id for later use
                sentences_batch[i][1].paragraph_id = sentences_batch[i][0].id
        
            # Commit database transaction
            self.db_original_content.commit_transaction()
            
            # Collect all document IDs from this batch
            batch_document_ids = [document.id for document, _ in documents_batch]
            document_ids.extend(batch_document_ids)
            
            # Now process embeddings in smaller sub-batches
            self._process_batch_embeddings(documents_batch, sections_batch, paragraphs_batch, sentences_batch, embedding_batch_size=embedding_batch_size)

        # Restore normal database settings
        try:
            self.db_original_content.restore_normal_settings()
        except AttributeError:
            pass
        
        return document_ids

    def _process_batch_embeddings(self, documents_batch, sections_batch, paragraphs_batch, sentences_batch, embedding_batch_size = 100):
        """
        Process embeddings for a batch of content.
        
        This is separated to allow for better error handling - if embedding generation fails,
        we've already committed the content to the database.
        """
        # Process document embeddings
        print("Generating document embeddings...")
        document_texts = [document_text for _, document_text in documents_batch]
        document_embeddings = self.generate_embeddings_batch(document_texts)
        
        # Process in smaller batches for insertion
        for i in range(0, len(documents_batch), embedding_batch_size):
            batch_end = min(i + embedding_batch_size, len(documents_batch))
            embedding_batch = []
            
            for j in range(i, batch_end):
                document, _ = documents_batch[j]
                embedding_batch.append({
                    'id': f"document_{document.id}",
                    'embedding': document_embeddings[j],
                    'metadata': {
                        'type': 'document', 
                        'document_id': document.id,
                        'content': document.title
                    }
                })
            
            self.db_embedding.bulk_insert_embeddings(embedding_batch)
        
        # Process section embeddings
        print("Generating section embeddings...")
        section_texts = [content for _, _, content in sections_batch]
        section_embeddings = self.generate_embeddings_batch(section_texts)
        
        for i in range(0, len(sections_batch), embedding_batch_size):
            batch_end = min(i + embedding_batch_size, len(sections_batch))
            embedding_batch = []
            
            for j in range(i, batch_end):
                _, section, content = sections_batch[j]
                embedding_batch.append({
                    'id': f"section_{section.id}",
                    'embedding': section_embeddings[j],
                    'metadata': {
                        'type': 'section', 
                        'section_id': section.id,
                        'document_id': section.document_id,
                        'content': section.title
                    }
                })
            
            self.db_embedding.bulk_insert_embeddings(embedding_batch)
        
        # Process paragraph embeddings
        print("Generating paragraph embeddings...")
        paragraph_texts = [content for _, _, content in paragraphs_batch]
        paragraph_embeddings = self.generate_embeddings_batch(paragraph_texts)
        
        for i in range(0, len(paragraphs_batch), embedding_batch_size):
            batch_end = min(i + embedding_batch_size, len(paragraphs_batch))
            embedding_batch = []
            
            for j in range(i, batch_end):
                _, paragraph, content = paragraphs_batch[j]
                embedding_batch.append({
                    'id': f"paragraph_{paragraph.id}",
                    'embedding': paragraph_embeddings[j],
                    'metadata': {
                        'type': 'paragraph', 
                        'paragraph_id': paragraph.id,
                        'section_id': paragraph.section_id,
                        'content': content[:100] + '...' if len(content) > 100 else content
                    }
                })
            
            self.db_embedding.bulk_insert_embeddings(embedding_batch)
        
        # Process sentence embeddings in even smaller sub-batches to manage memory
        print("Generating sentence embeddings...")
        sentence_sub_batch_size = 500  # Even smaller batch size for sentences
        
        for i in range(0, len(sentences_batch), sentence_sub_batch_size):
            sub_batch_end = min(i + sentence_sub_batch_size, len(sentences_batch))
            sub_batch = sentences_batch[i:sub_batch_end]
            
            print(f"Processing sentence batch {i}-{sub_batch_end} of {len(sentences_batch)}")
            sentence_texts = [content for _, _, content in sub_batch]
            sentence_embeddings = self.generate_embeddings_batch(sentence_texts)
            
            embedding_batch = []
            for j, (_, sentence, content) in enumerate(sub_batch):
                embedding_batch.append({
                    'id': f"sentence_{sentence.id}",
                    'embedding': sentence_embeddings[j],
                    'metadata': {
                        'type': 'sentence',
                        'sentence_id': sentence.id,
                        'paragraph_id': sentence.paragraph_id,
                        'content': content
                    }
                })
            
            self.db_embedding.bulk_insert_embeddings(embedding_batch)

# ----- MAIN EXECUTION -----

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process Wikipedia dumps for research database.")
    parser.add_argument('--action', type=str, default='parse', 
                        choices=['parse', 'main', 'thread'],
                        help='Action to perform: parse XML, process batches, or all')
    parser.add_argument('--batch_id', type=int, default=None, 
                        help='ID of the specific batch to process')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for embedding model (cpu, cuda, cuda:0, etc.)')

    return parser.parse_args()

def submit_job(
    script_path: str,
    python_file_name: str,
    action: str,
    batch_id: int,
    gpu_num: str
):
    """Submit a job to Slurm and return the job ID."""
    job_name = f'wiki_{batch_id}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action {action} --batch_id {batch_id}",
            job_name=job_name,
            gpu="V100",
            num=gpu_num,
            log_file_path=log_file_path,
            script_path=script_path
        )
    print(f"[PID: {batch_id}] is submitted!")
    return job_name

def main():
    args = load_args()
    config = load_configs()
    
    processor = WikipediaDumpProcessor(
        input_file_path=config['DocumentReader']['input_file_path'],
        cache_dir=config['DocumentReader']['cache_dir'],
        db_original_content_path=config['OriginalContentDatabase']['db_path'],
        db_embedding_path=config['EmbeddingDatabase']['db_path'],
        size_per_batch=config['DocumentReader']['size_per_batch'],
        device=args.device,
        workers=config['DocumentReader']['worker']
    )
    
    if args.action == 'parse':
        processor.parse_xml_dump()
    
    elif args.action == 'main':
        python_file_name = 'preprocessing_wikipedia_dataset.py'
        
        batch_ids = []
        filenames = list(os.listdir(processor.cache_dir))
        for filename in filenames:
            batch_id = filename.split('.')[0].split('_')[-1]
            if 'raw_page' in filename and 'processed_{batch_id}.done' not in filenames:
                batch_ids.append(int(batch_id))
        batch_ids.sort()

        for batch_id in batch_ids:
            submit_job(
                script_path=os.getcwd(),
                python_file_name=python_file_name,
                action='thread',
                batch_id=batch_id,
                gpu_num=config['DocumentReader']['gpu_num']
            )

    elif args.action == 'thread':
        processor.init_sentence_splitter()
        batch_file = os.path.join(processor.cache_dir, f'raw_page_{args.batch_id}.jsonl')
        if os.path.exists(batch_file):
            processor.process_batch_with_batch_mode(batch_file, args.batch_id, config['DocumentReader']['store_batch_size'], config['DocumentReader']['embedding_batch_size'])
        else:
            print(f"Batch file not found: {batch_file}")
    
if __name__ == '__main__':
    main()