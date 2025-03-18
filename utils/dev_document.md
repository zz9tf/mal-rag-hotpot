# Research Code Development Plan

## 1. Database Design

### 1.1 Original content database (SQL Database)

#### Tables

- **Document**

  - `id` (Primary Key)
  - `title` (Text)
  - `summary_id` (Foreign Key to Summary Table)

- **Section**

  - `id` (Primary Key)
  - `title` (Text)
  - `document_id` (Foreign Key to Document Table)
  - `summary_id` (Foreign Key to Summary Table)

- **Paragraph**

  - `id` (Primary Key)
  - `section_id` (Foreign Key to Section Table)

- **Multi-Sentences**

  - `id` (Primary Key)
  - `paragraph_id` (Foreign Key to Paragraph Table)
  - `content` (Text)

- **Summary**
  - `id` (Primary Key)
  - `document_id` (Foreign Key to Document Table, Nullable)
  - `section_id` (Foreign Key to Section Table, Nullable)
  - `content` (Text)

#### Relationships

- A `Document` can have multiple `Sections`.
- A `Section` can have multiple `Paragraphs`.
- A `Paragraph` can have multiple `Multi-Sentences`.
- `Summary` can be linked to either a `Document` or a `Section`.

### 1.2 Embedding database (Chroma Database)

#### Structure

- **Embedding Table**
  - `id` (Primary Key, Linked to Database-A)
  - `embedding_value` (Vector)

### 1.3 HotpotQA train database (HotpotQA Train Database)

#### Tables

- **TrainData**

  - `id` (Primary Key)
  - `title` (Text)
  - `content` (Text)

- **Question**
  - `id` (Primary Key)
  - `question` (Text)
  - `supporting_facts` (JSON)
  - `context` (JSON)
  - `answer` (Text)
  - `type` (Text)

## 2. Python Classes

### 2.1 Data Classes

- **Document**

  - Attributes: `id`, `title`, `sections`, `summary`
  - Methods: `add_section`, `get_sections`, `get_summary`

- **Section**

  - Attributes: `id`, `title`, `paragraphs`, `summary`
  - Methods: `add_paragraph`, `get_paragraphs`, `get_summary`

- **Paragraph**

  - Attributes: `id`, `multi_sentences`
  - Methods: `add_multi_sentence`, `get_multi_sentences`

- **MultiSentence**

  - Attributes: `id`, `content`
  - Methods: `get_content`

- **Summary**
  - Attributes: `id`, `content`, `document_id`, `section_id`
  - Methods: `get_content`

### 2.2 Database Interaction Classes

- **DatabaseAHandler**

  - Methods: `insert_document`, `get_document`, `insert_section`, `get_section`, `insert_paragraph`, `get_paragraph`, `insert_multi_sentence`, `get_multi_sentence`, `insert_summary`, `get_summary`

- **DatabaseBHandler**

  - Methods: `insert_embedding`, `get_embedding`, `search_by_embedding`

- **DatabaseCHandler**
  - Methods: `insert_train_data`, `get_train_data`, `insert_question`, `get_question`

## 3. Implementation Details

### 3.1 Database-A Operations

- **Insertion**: Insert data into respective tables maintaining relationships.
- **Retrieval**: Retrieve data using IDs and maintain hierarchical relationships.
- **Summary Linking**: Link summaries to respective documents or sections.

### 3.2 Database-B Operations

- **Embedding Storage**: Store embedding values linked to Database-A IDs.
- **Search**: Perform embedding search and retrieve corresponding Database-A data.

### 3.3 Database-C Operations

- **Data Matching**: Match titles and content with Database-A for text and embedding matching.
- **Question Handling**: Store and retrieve questions with supporting facts and context.

## 4. Additional Features

- **Text Matching**: Implement fuzzy text matching for sentences.
- **Embedding Matching**: Use embedding matching for sentence-level data.
- **Summary Retrieval**: Retrieve summaries and link them to original content.

## 5. Code Structure

```python
# database_a.py
class DatabaseAHandler:
    # Implementation for Database-A operations

# database_b.py
class DatabaseBHandler:
    # Implementation for Database-B operations

# database_c.py
class DatabaseCHandler:
    # Implementation for Database-C operations

# data_classes.py
class Document:
    # Implementation for Document class

class Section:
    # Implementation for Section class

class Paragraph:
    # Implementation for Paragraph class

class MultiSentence:
    # Implementation for MultiSentence class

class Summary:
    # Implementation for Summary class

# main.py
# Main script to orchestrate the operations
```

## 6. Future Enhancements

- **Performance Optimization**: Optimize database queries for large datasets.

- **Scalability**: Ensure the system can scale with increasing data.

- **Advanced Matching**: Implement more advanced text and embedding matching algorithms.
