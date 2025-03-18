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

class WikipediaDumpProcessor:
    def __init__(
        self,
        input_file_path,
        cache_dir,
        db_original_content_path,
        db_embedding_path,
        workers=5,
        pages_per_batch=100,
        device=None
    ):
        self.input_file_path = input_file_path
        self.cache_dir = cache_dir
        self.workers = workers
        self.pages_per_batch = pages_per_batch
        
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
    
    def parse_xml_dump(self):
        """Parse the Wikipedia XML dump into individual pages."""
        if self._processed_xml():
            print("XML already processed, skipping parsing step.")
            return
        
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
        with open(self.input_file_path, 'r') as input_file:
            with tqdm(total=total_file_size, desc="Processing XML", unit="B", unit_scale=True) as pbar:
                for line in input_file:
                    parser.feed(line)
                    for event, element in parser.read_events():
                        if event == 'start' and 'page' in element.tag:
                            raw_page = []
                        if event == 'end' and 'page' in element.tag:
                            end_page = True
                    
                    # Collect page content
                    if isinstance(raw_page, list):
                        raw_page.append(line)
                    
                    # Process completed page
                    if end_page:
                        raw_page = ''.join(raw_page)
                        current_batch.append(raw_page)
                        
                        # When batch is full, add to processing queue
                        if len(current_batch) >= self.pages_per_batch:
                            all_batches.append([batch_id, current_batch])
                            batch_id += 1
                            current_batch = []
                        
                        page_num += 1
                        raw_page = None
                        end_page = False
                    
                    # Update progress
                    update_len += len(line)
                    if update_len > 10000000:  # Update every ~10MB
                        pbar.set_postfix_str(f"Pages: {page_num}")
                        pbar.update(update_len)
                        update_len = 0
                    
                    # Process batches in parallel when we have enough
                    if len(all_batches) >= 10:
                        # TODO: remove this later
                        if batch_id > 59:
                            self._process_batches(all_batches)
                        all_batches = []
        
        # Process any remaining pages
        if current_batch:
            all_batches.append([batch_id, current_batch])
        
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
        raw_sections = re.split(r'(?m)^==\s*(.*?)\s*==\s*$', page_text)
        
        # First item is the lead section (before any headings)
        lead_section = raw_sections[0]
        
        sections = []
        # Add the lead section (abstract)
        if lead_section.strip():
            sections.append({
                'title': 'Abstract',
                'content': self._clean_text(lead_section)
            })
        
        # Process the rest of the sections
        for i in range(1, len(raw_sections), 2):
            if i+1 < len(raw_sections):
                section_title = raw_sections[i]
                section_content = raw_sections[i+1]
                
                # Skip unwanted sections
                if section_title.lower() in ['references', 'external links', 'see also', 
                                           'further reading', 'notes', 'bibliography']:
                    continue
                
                if section_content.strip():
                    sections.append({
                        'title': section_title,
                        'content': self._clean_text(section_content)
                    })
        
        return sections
    
    def split_into_paragraphs(self, section_content):
        """Split section content into paragraphs."""
        # Split on blank lines
        paragraph_texts = re.split(r'\n\s*\n', section_content)
        paragraphs = []
        
        for para_text in paragraph_texts:
            if para_text.strip():
                paragraphs.append({
                    'content': para_text.strip()
                })
                
        return paragraphs
    
    def split_into_sentences(self, paragraph_content):
        """Split paragraph content into sentences."""
        # Basic sentence splitting - can be improved with NLP libraries
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, paragraph_content)
        
        return [sent.strip() for sent in sentences if sent.strip()]
    
    # ----- PROCESSING METHODS -----
    
    def process_batches(self):
        """Process all raw page batches and store in database."""
        # Get list of raw page files
        filenames = [f for f in os.listdir(self.cache_dir) if f.startswith('raw_page_') and f.endswith('.jsonl')]
        filenames.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        total_docs = 0
        english_docs = 0
        
        with tqdm(total=len(filenames), desc="Processing batches") as pbar:
            for filename in filenames:
                batch_id = int(filename.split('_')[2].split('.')[0])
                docs_added, en_docs = self.process_batch(os.path.join(self.cache_dir, filename), batch_id)
                
                total_docs += docs_added
                english_docs += en_docs
                
                pbar.update(1)
                pbar.set_postfix_str(f"Total: {total_docs}, English: {english_docs}")
        
        print(f"Finished processing. Total documents: {total_docs}, English documents: {english_docs}")
    
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
    
    def process_and_store_document(self, page):
        """Process a Wikipedia page and store in database with embeddings."""
        try:
            # Create the document
            document = Document(title=page['title'])
            document_id = self.db_original_content.insert_document(document)
            document.id = document_id
            
            # Split into sections
            sections_data = self.split_into_sections(page['text'])
            
            # Process each section
            section_texts = []
            for section_data in sections_data:
                # Create section
                section = Section(title=section_data['title'], document_id=document_id)
                section_id = self.db_original_content.insert_section(section)
                section.id = section_id
                
                # Split into paragraphs
                paragraphs_data = self.split_into_paragraphs(section_data['content'])
                
                # Process each paragraph
                paragraph_texts = []
                for paragraph_data in paragraphs_data:
                    # Create paragraph
                    paragraph = Paragraph(section_id=section_id)
                    paragraph_id = self.db_original_content.insert_paragraph(paragraph)
                    paragraph.id = paragraph_id
                    
                    # Split into sentences
                    sentences = self.split_into_sentences(paragraph_data['content'])
                    
                    # Skip empty paragraphs
                    if not sentences:
                        continue
                    
                    # Create all sentences first
                    multi_sentences = []
                    for sentence in sentences:
                        multi_sentence = MultiSentence(content=sentence, paragraph_id=paragraph_id)
                        multi_sentence_id = self.db_original_content.insert_multi_sentence(multi_sentence)
                        multi_sentence.id = multi_sentence_id
                        multi_sentences.append(multi_sentence)
                    
                    # Generate and store sentence embeddings in batch
                    sentence_embeddings = self.generate_embeddings_batch(sentences)
                    for i, multi_sentence in enumerate(multi_sentences):
                        self.db_embedding.insert_embedding(
                            f"sentence_{multi_sentence.id}",
                            sentence_embeddings[i],
                            metadata={
                                'type': 'sentence',
                                'document_id': document_id,
                                'section_id': section_id,
                                'paragraph_id': paragraph_id,
                                'content': multi_sentence.content
                            }
                        )
                    
                    # Generate and store paragraph embedding
                    paragraph_text = " ".join(sentences)
                    paragraph_texts.append(paragraph_text)
                    paragraph_embedding = self.generate_embedding(paragraph_text)
                    self.db_embedding.insert_embedding(
                        f"paragraph_{paragraph_id}",
                        paragraph_embedding,
                        metadata={
                            'type': 'paragraph',
                            'document_id': document_id,
                            'section_id': section_id,
                            'content': paragraph_text
                        }
                    )
                
                # Generate and store section embedding
                section_text = f"Section: {section_data['title']}\n" + "\n".join(paragraph_texts)
                section_texts.append(section_text)
                section_embedding = self.generate_embedding(section_text)
                self.db_embedding.insert_embedding(
                    f"section_{section_id}",
                    section_embedding,
                    metadata={
                        'type': 'section',
                        'document_id': document_id,
                        'content': section_text
                    }
                )
            
            # Generate and store document embedding
            document_text = f"Title: {page['title']}\n\n" + "\n\n".join(section_texts)
            document_embedding = self.generate_embedding(document_text)
            self.db_embedding.insert_embedding(
                f"document_{document_id}",
                document_embedding,
                metadata={
                    'type': 'document',
                    'content': document_text
                }
            )
            
            return document_id
            
        except Exception as e:
            print(f"Error processing document {page['title']}: {e}")
            return None
    
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
        return embedding_tensors.cpu().numpy().tolist()

# ----- MAIN EXECUTION -----

def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Process Wikipedia dumps for research database.")
    parser.add_argument('--action', type=str, default='parse', 
                        choices=['parse', 'process', 'process_batch', 'all'],
                        help='Action to perform: parse XML, process batches, or all')
    parser.add_argument('--batch_id', type=int, default=None, 
                        help='ID of the specific batch to process')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of worker threads')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for embedding model (cpu, cuda, cuda:0, etc.)')

    return parser.parse_args()

def main():
    args = load_args()
    config = load_configs()
    
    processor = WikipediaDumpProcessor(
        input_file_path=config['DocumentReader']['input_file_path'],
        cache_dir=config['DocumentReader']['cache_dir'],
        db_original_content_path=config['OriginalContentDatabase']['db_path'],
        db_embedding_path=config['EmbeddingDatabase']['db_path'],
        workers=args.workers,
        device=args.device
    )
    
    if args.action == 'parse' or args.action == 'all':
        processor.parse_xml_dump()
    
    if args.action == 'process_batch' and args.batch_id is not None:
        batch_file = os.path.join(processor.cache_dir, f'raw_page_{args.batch_id}.jsonl')
        if os.path.exists(batch_file):
            processor.process_batch(batch_file, args.batch_id)
        else:
            print(f"Batch file not found: {batch_file}")
    
    if args.action == 'process' or args.action == 'all':
        processor.process_batches()
    
if __name__ == '__main__':
    main()