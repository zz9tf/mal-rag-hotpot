import os
from tqdm import tqdm
import re
import json
from langdetect import detect
from lxml import etree
from xml.etree.ElementTree import XMLPullParser
import html
import argparse
from multiprocessing.pool import ThreadPool
from utils import generate_and_execute_slurm_job, load_configs

class WikipediaDumpReader():
    def __init__(
        self,
        input_file_path,
        cache_dir,
        worker: int=5,
        pages_per_batch: int=100
    ):
        self.input_file_path = input_file_path
        self.cache_dir = cache_dir
        self.worker = worker
        self.pages_per_batch = pages_per_batch
    
    def _write_to_disk(self, batch_id, batch):
        cache_file_path = os.path.join(self.cache_dir, f'raw_page_{batch_id}_not_finish.jsonl')
        with open(cache_file_path, 'w') as cache_file:
            for page in batch:
                cache_file.write(json.dumps({'page': page}) + '\n')
        os.rename(cache_file_path, os.path.join(self.cache_dir, f'raw_page_{batch_id}.jsonl'))
    
    def parse_file(self, input_file_path):
        total_file_size = os.path.getsize(input_file_path)
        parser = XMLPullParser(events=('start', 'end'))
        raw_page = None
        end_page = False
        page_num = 0
        batch_id = 0
        current_batch = []
        all_batches = []
        update_len = 0
        
        # Load all batches
        with open(input_file_path, 'r') as input_file:
            with tqdm(total=total_file_size, desc="Processing", unit="B", unit_scale=True) as pbar:
                for line in input_file:
                    parser.feed(line)
                    for event, element in parser.read_events():
                        if event == 'start' and 'page' in element.tag:
                            raw_page = []
                        if event == 'end' and 'page' in element.tag:
                            end_page = True
                    # add line
                    if isinstance(raw_page, list):
                        raw_page.append(line)
                    if end_page:
                        raw_page = ''.join(raw_page)
                        current_batch.append(raw_page)
                        if len(current_batch) > self.pages_per_batch:
                            all_batches.append([batch_id, current_batch])
                            batch_id += 1
                            current_batch = []
                        page_num += 1
                        raw_page = None
                        end_page = False
                    if update_len > 1e10:
                        pbar.set_postfix_str(f"page {page_num}")
                        pbar.update(update_len)
                        update_len = 0
                    else:
                        update_len += len(line)
                    if len(all_batches) >= 10:
                        # Write to disk using a ThreadPool for parallel processing of batches
                        with ThreadPool(self.worker) as pool:
                            # Wrap the pool.imap with tqdm to display a progress bar
                            for _ in pool.imap(lambda x: self._write_to_disk(x[0], x[1]), all_batches):
                                pass
                        all_batches = []
        # If there are remaining pages in the last batch, add them
        if current_batch:
            all_batches.append([batch_id, current_batch])

        if len(all_batches) > 0:
            # Write to disk using a ThreadPool for parallel processing of batches
            with ThreadPool(self.worker) as pool, tqdm(total=len(all_batches), desc="process batches") as pbar:
                # Wrap the pool.imap with tqdm to display a progress bar
                for _ in pool.imap(lambda x: self._write_to_disk(x[0], x[1]), all_batches):
                    pbar.update(1)  # Update the progress bar for each processed batch
    
    def _processed_xml(self):
        for filename in os.listdir(self.cache_dir):
            if "raw_page" in filename:
                return True
        return False
    
    def _remove_self_closing_ref(self, text):
        """
        Removes <ref /> self-closing tags.
        """
        result = []
        i = 0
        
        while i < len(text):
            # Check for the beginning of a self-closing <ref /> tag
            if text[i:i+4] == '<ref' and '/>' in text[i+4:]:
                end_idx = text.find('>', i)
                if text[end_idx - 1] == '/':  # Confirm it's self-closing
                    i = end_idx + 1  # Skip the entire self-closing tag
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)

    def _remove_ref_with_content(self, text):
        """
        Removes <ref*?.>...</ref> tags using a stack.
        """
        stack = []
        result = []
        i = 0
        
        while i < len(text):
            # Check for opening <ref> tag
            if text[i:i+4] == '<ref' and '>' in text[i+4:]:
                stack.append(i)  # Push position of opening tag onto stack
                i += 4  # Skip the opening tag
            # Check for closing </ref> tag
            elif text[i:i+6] == '</ref>' and len(stack) > 0:
                stack.pop()  # Pop the last opening tag position
                i += 6  # Skip the closing tag
            elif len(stack) == 0:
                result.append(text[i])  # Add to result if not inside a <ref ...>...</ref> block
                i += 1
            else:
                i += 1  # Skip characters within the <ref>...</ref> block
        
        return ''.join(result)

    def _remove_braces_content(self, text):
        """
        Removes { }
        """
        stack = []
        result = []
        i = 0

        while i < len(text):
            char = text[i]
            if char == '{' and i + 1 < len(text) and text[i + 1] == '{':
                # Push the current length of result onto the stack to remember where the brace started
                stack.append(len(result))
                i += 1  # Skip the second '{'
            elif char == '}' and i + 1 < len(text) and text[i + 1] == '}' and len(stack) > 0:
                start = stack.pop()
                block_content = ''.join(result[start:])
                if '|' in block_content and len(block_content.split('|')) == 2 and block_content[0] == 'Blockquote':
                    # Keep only the content is only two part splitted by '|'
                    keep_content = block_content.split('|')[1]
                    result = result[:start] + list(keep_content)
                else:
                    result = result[:start]
                i += 1  # Skip the second '}'
            else:
                # Append character to the result only if not within braces
                result.append(char)
            i += 1
        
        # Join the result back into a string
        return ''.join(result)

    def _remove_square_brackets(self, text):
        """
        Removes [[ | target]] -> target
        """
        square_stack = []
        result = []
        i = 0

        while i < len(text):
            char = text[i]

            if char == '[' and i + 1 < len(text) and text[i + 1] == '[':
                # Start of a square bracket block
                square_stack.append(len(result))  # Save the position in result
                i += 1  # Skip the second '['
            elif char == ']' and i + 1 < len(text) and text[i + 1] == ']' and len(square_stack) > 0:
                # End of a square bracket block
                start = square_stack.pop()
                block_content = ''.join(result[start:])
                if '|' in block_content:
                    # Keep only the content after '|' for other cases
                    keep_content = block_content.split('|')[-1]
                    result = result[:start] + list(keep_content)
                i += 1  # Skip the second ']'
            else:
                # Add the character to the result if not within brackets or after processing
                result.append(char)
            i += 1

        return ''.join(result)

    def _remove_tags(self, text):
        """
        Removes <*>
        """
        # Remove all HTML/XML tags using non-greedy regex
        cleaned_text = re.sub(r'<.*?>', '', text)
        return cleaned_text
        
    def _clean_text(self, raw_text):
        cleaned_text = html.unescape(raw_text)
        # Remove <ref> tag without greedy
        cleaned_text = self._remove_self_closing_ref(cleaned_text)
        # Remove <ref /> tag without greedy
        cleaned_text = self._remove_ref_with_content(cleaned_text)
        # Remove { }
        cleaned_text = self._remove_braces_content(cleaned_text)
        # Remove <!-- ... -->
        cleaned_text = re.sub(r'<!--.*?-->', '', cleaned_text, flags=re.DOTALL)
        # Remove ''' '''
        cleaned_text = re.sub(
            r"('{3,})(.*?)(\1)",  # Match balanced quotes
            lambda m: m.group(2) if len(m.group(1)) == len(m.group(3)) else m.group(0),
            cleaned_text
        )
        # Remove [[ | target]] -> target
        cleaned_text = self._remove_square_brackets(cleaned_text)
        # Remove all remaining HTML/XML tags
        cleaned_text = self._remove_tags(cleaned_text)
        cleaned_text = cleaned_text.strip() + '\n'
        
        return cleaned_text
    
    def _get_abstract(self, text, title, raw_text):
        # Split the text into lines
        abstract = []
        print_str = ""
        abstract = self._clean_text(text)
        if len(abstract) <= 0:
        #     print(raw_text[:10000])
        #     print(">>>>>>>>>>>>>>>")
        #     print(abstract.strip())
            print_str += f"Not find abstract at title {title}\n"
            
        # assert len(abstract) > 0, f"Not find abstract at title {title}"  
        
        return abstract, print_str

    def _read_page(self, page):
        if page['title'].lower().startswith('file:') or page['title'].lower().startswith('category:'):
            return None, None, ''
        file_dict = {}
        print_str = ""
        file_dict['title'] = page['title']
        page_text = page['text']
        file_dict['sections'] = {}
        
        paper_content = "Title: {}\n\n".format(file_dict['title'])
        start = len(paper_content)
        
        raw_sections = re.split(r'(?m)^==\s*', page_text)
        old_section_title = None
        current_section = ""
        # Get abstract
        abstract, abstract_print_str = self._get_abstract(raw_sections[0], page['title'], page['raw_page'])
        print_str += abstract_print_str
        old_section_title = 'abstract'
        if len(abstract) > 0:
            current_section = f'{abstract}\n'
        
        should_skip = False
        
        for section in raw_sections[1:]:
            # Process body
            lines = section.split('\n')
            section_title = lines[0].strip('= ')
            section_content = self._clean_text("\n".join(lines[1:]))
            if not section.startswith('='):
                # Update accumulated contents
                if len(current_section.strip()) > 0:
                    paper_content += current_section
                    file_dict['sections'][old_section_title] = [start, len(paper_content)-1]
                    current_section = ""
                # Skip the content when the title is not very useful
                if section_title.lower() in ['references', 'external links', 'see also', 'further reading', 'literature and sources']:
                    should_skip = True
                else:
                    should_skip = False
                    # update status
                    old_section_title = section_title
                    start = len(paper_content)
                    current_section = f'{section_content}\n'
            elif not should_skip:
                assert old_section_title is not None, 'old_section_title shouldn\'t be None'
                current_section += f'{section_content}\n'
        
        # Update accumulated contents
        if len(current_section.strip()) > 0:
            paper_content += current_section
            file_dict['sections'][old_section_title] = [start, len(paper_content)-1]
            current_section = ""
        print("raw_content:", page['raw_page'])
        print("paper_content:", paper_content)
        print("file_dict:", file_dict)
        exit()
        # if len(file_dict['sections']) == 0:
        #     print_str += f"[documetn reader] Detect invalided document with no sections {page['title']}\n{page['raw_page']}\n"
        #     return None, None, print_str
        # else:
        #     file_document = Document(
        #         text=paper_content,
        #         metadata=file_dict
        #     )
        #     return file_document, 'abstract' not in file_dict['sections'], print_str
        
    def _go_over_string_element(self, xml_data):
        # Parse the string as a file-like object
        root = etree.fromstring(xml_data)
        redirect = root.find('redirect')
        title = root.findtext('title')
        text_element = root.find(".//text")  # Find the text element (can be nested)
        if redirect is not None or text_element is None or text_element.text is None: return None
        text = text_element.text.strip()
        
        return {'title': title, 'text': text, 'raw_page': xml_data}
        
    def _process_batch(self, batch_id, page_chunk, break_num=None):
        break_num = len(page_chunk) if break_num is None else break_num
        documents = []
        no_abstract_num = 0
        print_str = f"Start batch id: {batch_id}\n"

        for raw_page in tqdm(page_chunk[:break_num], desc=f'reading page chunk {batch_id}'):
            page = self._go_over_string_element(raw_page)
            if page and len(page['text']) > 250:
                document, no_abstract, page_print_str = self._read_page(page)
                print_str += page_print_str
                if document:
                    documents.append(document)
                    if no_abstract: no_abstract_num += 1
        return documents, no_abstract_num, print_str
    
    def remove_duplicate_documents(self, documents):
        unique_document = []
        file_head_set = set()
        for document in documents:
            head = document.text[:800]
            if head not in file_head_set:
                file_head_set.add(head)
                unique_document.append(document)
        return unique_document

    def remove_non_english_documents(self, documents):
        english_documents = []
        for document in documents:
            head = document.text[:800]
            try:
                if detect(head) == 'en':
                    english_documents.append(document)
            except Exception as e:
                print(e)
        return english_documents
    
    def load_data(self):
        if not self._processed_xml():
            self._parse_files()
        
        filenames = [filename for filename in os.listdir(self.cache_dir) if 'raw_page' in filename]
        for filename in os.listdir(self.cache_dir):
            if 'finished_chunk' in filename:
                remove_name = f"raw_page_{filename.split('_')[-1]}"
                filenames.remove(remove_name)
        filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        if len(filenames) > 0:
            # TODO: comment [:4]
            # filenames = filenames[:4]
            
            no_abstract_num = 0
            document_num = 0
            
            with tqdm(total=len(filenames), desc="process file") as pbar:
                for filename in filenames:
                    current_batch = []
                    with open(os.path.join(self.cache_dir, filename), 'r') as cache_file:
                        for line in cache_file:
                            data = json.loads(line)
                            current_batch.append(data['page'])
                    pbar.update(1)
                    
                    batch_id = int(filename.split('.')[0].split('_')[-1])
                    # TODO: remove break number
                    documents, batch_no_abstract_num, print_str = self._process_batch(batch_id, current_batch)
                    # save_nodes_jsonl(os.path.join(self.cache_dir, f'finished_chunk_{batch_id}.jsonl'), documents)
                    
                    no_abstract_num += batch_no_abstract_num
                    document_num += len(documents)
                    
                    pbar.set_postfix_str(
                        f"{no_abstract_num}/{document_num}  {(no_abstract_num/document_num)*100:.2f}%"
                    )

        documents = []
        filenames = [filename for filename in os.listdir(self.cache_dir) if 'finished_chunk' in filename]
        filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        for filename in filenames:
                with open(os.path.join(self.cache_dir, filename), 'r', encoding='utf-8') as input_file:
                    file_size = os.path.getsize(os.path.join(self.cache_dir, filename))
                    with tqdm(total=file_size, desc=f'merging {filename}...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                        for i, line in enumerate(input_file):
                            try:
                                node_data = json.loads(line)
                                # node = TextNode.from_dict(node_data)
                                # documents.append(node)
                                # pbar.update(len(line))
                            except:
                                print(i, line)
        
        return documents
        
def load_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Create chromadbs.")
    parser.add_argument('--action', type=str, default='main', help='The action to preprocess wikipedia dump')
    parser.add_argument('--pid', type=str, default=None, help='The PID of the subtask to preprocess wikipedia dump')
    parser.add_argument('--filename', type=str, default=None, help='The filename that the subtask to preprocess')

    return parser.parse_args()

def submit_job(
    script_path: str,
    cpu_num: str,
    filename: str,
    python_file_name: str,
    action: str
):
    """Submit a job to Slurm and return the job ID."""
    pid = int(filename.split('.')[0].split('_')[-1])
    
    job_name = f'wiki_{pid}'
    log_file_path = os.path.abspath(os.path.join(script_path, 'out/{job_name}.out'))
    script_path = os.path.abspath(os.path.join(script_path, 'execute/execute.sh'))
    job_name = generate_and_execute_slurm_job(
            python_start_script=f"{python_file_name} --action {action} --pid {pid} --filename {filename}",
            account="guest",
            partition="guest-compute",
            job_name=job_name,
            qos='low',
            time="24:00:00",
            num=cpu_num,
            log_file_path=log_file_path,
            script_path=script_path
        )
    print(f"[PID: {pid}] is submitted!")
    return job_name

if __name__ == '__main__':
    config = load_configs()['DocumentReader']
    reader = WikipediaDumpReader(
        config['input_file_path'], 
        config["cache_dir"], 
        config['worker'], 
        config['pages_per_batch']
    )
    
    args = load_args()
    if args.action == 'main':
        cpu_num = 10
        python_file_name = 'preprocessing_wikipedia_dataset.py'
        
        if not reader._processed_xml():
            reader.parse_file(reader.input_file_path)
        exit()
        
        filenames = [filename for filename in os.listdir(reader.cache_dir) if 'raw_page' in filename]
        for filename in os.listdir(reader.cache_dir):
            if 'finished_chunk' in filename:
                remove_name = f"raw_page_{filename.split('_')[-1]}"
                filenames.remove(remove_name)
        filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        if len(filenames) == 0:
            documents = []
            filenames = [filename for filename in os.listdir(reader.cache_dir) if 'finished_chunk' in filename]
            filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
            for filename in filenames:
                with open(os.path.join(reader.cache_dir, filename), 'r', encoding='utf-8') as input_file:
                    file_size = os.path.getsize(os.path.join(reader.cache_dir, filename))
                    with tqdm(total=file_size, desc=f'merging {filename}...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                        for i, line in enumerate(input_file):
                            try:
                                node_data = json.loads(line)
                                # node = TextNode.from_dict(node_data)
                                # documents.append(node)
                                pbar.update(len(line))
                            except:
                                print(i, line)
            # save_nodes_jsonl(os.path.join(reader.cache_dir, f'final.jsonl'), documents)
            
        else:
            for filename in filenames:
                submit_job(
                    script_path=os.getcwd(),
                    cpu_num=cpu_num,
                    filename=filename,
                    python_file_name=python_file_name,
                    action='thread'
                )
    elif args.action == 'thread':
        current_batch = []
        file_size = os.path.getsize(os.path.join(reader.cache_dir, args.filename))
        with tqdm(total=file_size, desc=f'loading {args.filename}...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(os.path.join(reader.cache_dir, args.filename), 'r') as cache_file:
                for line in cache_file:
                    data = json.loads(line)
                    current_batch.append(data['page'])
                    pbar.update(len(line))
        documents, batch_no_abstract_num, print_str = reader._process_batch(args.pid, current_batch, 100)
        # save_nodes_jsonl(os.path.join(reader.cache_dir, f'finished_chunk_{args.pid}.jsonl'), documents)
        print(f"{batch_no_abstract_num}/{len(documents)}  {(batch_no_abstract_num/len(documents))*100:.2f}%")
    
    
    elif args.action == 'merge':
        documents = []
        filenames = [filename for filename in os.listdir(reader.cache_dir) if 'finished_chunk' in filename]
        filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        save_path = os.path.join(reader.cache_dir, f'final.jsonl')
        save_file = open(save_path, 'w')
        for filename in filenames:
            with open(os.path.join(reader.cache_dir, filename), 'r', encoding='utf-8') as input_file:
                file_size = os.path.getsize(os.path.join(reader.cache_dir, filename))
                with tqdm(total=file_size, desc=f'merging {filename}...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for line in input_file:
                        save_file.write(line)
                        pbar.update(len(line))
                        
        save_file.close()
        