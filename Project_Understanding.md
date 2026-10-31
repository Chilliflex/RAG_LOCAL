# Complete Workflow: Processing a 100-Page PDF and Answering Questions

Imagine you have a 100-page environmental research report containing:

- Printed text and scanned pages.
- Tables spread across multiple pages.
- Mathematical formulas and equations.
- Images, diagrams, and charts.
- Handwritten notes in margins.
- Handwritten text inside scanned images.
- Bookmarks, references, and URLs.

Your system processes this PDF in two main phases:

1. **Ingestion:** Reads the PDF, understands its structure, and builds a searchable knowledge database.
2. **Question Answering:** Searches that database and generates answers with page citations.

Everything runs locally on your PC.

---

# Part 1: Ingestion – Processing the 100-Page PDF

**Input:** `environmental_report.pdf`

**Total pages:** 100

One PDF containing mixed document formats.

## 10 Processing Stages

Each stage contributes to the final searchable document.

## Stage 1. File Fingerprinting

**Tool:** Python standard library – SHA-256

### What happens

- Calculates the PDF's SHA-256 hash.
- Uses the first 16 hexadecimal characters as its document ID.
- Checks whether the PDF has already been ingested.

### Example

text

```text
environmental_report.pdf
       ↓
SHA-256 fingerprint
       ↓
Document ID: a82f19c...
       ↓
data/a82f19c.../
```

**Purpose:** Prevents unnecessary reprocessing of an already ingested document.

## Stage 2. PDF Triage

**Tool:** PyMuPDF

### What happens

- Examines all 100 pages.
- Identifies digital versus scanned pages.
- Counts images.
- Extracts bookmarks.
- Detects hyperlink URLs.

### Example

text

```text
Total pages:       100
Digital pages:      75
Scanned pages:      25
Images:             42
Bookmarks:           8
URLs:               15
```

**Purpose:** Understands the PDF before deeper processing begins.

## Stage 3. Document Understanding

**Main tool:** Docling

**Additional components:** Heron, TableFormer, RapidOCR, and SmolVLM-256M.

### Models and components

| Component | Responsibility |
| --- | --- |
| Heron | Detects document layout |
| TableFormer | Understands table structure |
| RapidOCR | Reads printed text |
| SmolVLM-256M | Classifies and describes pictures |

### What happens

Docling acts as the main document understanding engine.

For example, page 12 contains a table, paragraph, and diagram.

Docling attempts to identify each separately instead of extracting everything as one block of text.

### Output includes

- Paragraphs and headings.
- Tables with rows and columns.
- Picture descriptions.
- Reading order.
- OCR text.

**Purpose:** Converts visually complex PDF pages into structured information.

## Stage 4. Handwriting Recognition

**Tool/model:** TrOCR-base-handwritten

### What happens

- Checks OCR lines with confidence below 0.95.
- Re-reads text-like pictures.
- Processes uncovered images.
- Retains TrOCR results above 0.6 confidence.

### Example

Original handwritten note:

text

```text
Station C showed unusual nitrate levels
```

TrOCR output:

text

```text
Station C showed unusual nitrate levels
```

**Purpose:** Recovers handwritten information that ordinary printed-text OCR might miss.

## Stage 5. Document Stitching

**Tool:** Python

### What happens

The system organizes extracted information.

It:

- Removes repeated headers and footers.
- Joins paragraphs that continue across pages.
- Attaches captions to relevant figures and tables.
- Attaches margin notes to nearby paragraphs.
- Builds a section tree.

### Section hierarchy

The hierarchy uses the following priority:

1. Bookmarks.
2. Heading numbers.
3. Layout levels.

### Multi-page table handling

Multi-page tables are merged only under specific conditions:

- The continuation must be the first item on the next page.
- It must have the same number of columns.
- It must have no caption of its own.

**Purpose:** Reconstructs the logical structure of the original document.

## Stage 6. Enrichment

**Tool:** Python regex

### What happens

Finds:

- References.
- URLs.
- Cross-references.

### Example

text

```text
Page 25:

"See Table 7 for nitrate measurements."
```

The system identifies the reference to Table 7 and creates a cross-reference connection.

**Purpose:** Connects related information throughout the PDF.

## Stage 7. Summarization

**Model:** Qwen2.5-3B-Instruct

**Quantization:** Q4\_K\_M GGUF

**Inference engine:** llama.cpp

### What happens

Qwen generates:

- Section summaries.
- One whole-document summary.

### Example

Original section: 12 pages discussing water quality.

Generated summary:

text

```text
The section examines water quality across
five monitoring stations, focusing on nitrate,
turbidity and pH measurements.
```

**Purpose:** Makes broad document questions easier to answer later.

## Stage 8. Chunking

**Tool:** Python

### What happens

The extracted document is divided into searchable units called chunks.

### Example

text

```text
Section 1 → Paragraph chunks
Section 2 → Paragraph chunks
Section 3 → Table chunks
Section 4 → Figure chunks
Section 5 → Formula chunks
```

### Important rules

- Structure-first chunking.
- Tables, figures, and formulas remain atomic where possible.
- Large tables are split by rows with headers repeated.
- Section and document summaries are also added as chunks.

**Purpose:** Prepares information for efficient searching.

## Stage 9. Embedding and Indexing

**Embedding model:** BGE-M3

**Vector database:** Qdrant

### What happens

BGE-M3 converts chunks into numerical embeddings for semantic search and also supports sparse representations.

Qdrant stores the searchable vectors and associated metadata in embedded, on-disk mode.

### Example

text

```text
"Station C nitrate level = 11.3 mg/L"
                 ↓
             BGE-M3
                 ↓
          Numerical vectors
                 ↓
          Qdrant database
```

The system uses:

- Dense retrieval.
- Sparse retrieval.

**Purpose:** Finds relevant information even when the user's question uses different words from the PDF.

## Stage 10. Completeness Audit

**Tool:** Python

### What happens

Checks:

- Pages processed.
- Pages containing extracted items.
- Scanned pages.
- Sections and TOC entries.
- References, URLs, and cross-references.
- Counts of tables, figures, formulas, margin notes, and handwriting.
- Low-confidence regions.
- Processing gaps.

### Example

text

```text
Pages:              100
Sections:            12
Tables:              18
Figures:             42
Formulas:            11
Gaps:                 []
Complete:          true
```

These are illustrative numbers, not actual results.

### Final files

text

```text
data/<doc_id>/
    document.json
    audit.json
    timings.json
    pages/
        0001.json
        0002.json
        ...
        0100.json
```

**Ingestion complete\!**

Your 100-page PDF is now searchable.

---

# Part 2: Asking Questions – How the AI Answers

Now imagine you ask five different questions.

### Example command

bash

```bash
python -m rag ask "What was the nitrate level at station C?"
```

## Stage 11. Question Routing

**Tool:** Python regex rules

### What happens

The system identifies the question type.

There are five routes:

| Route | Example |
| --- | --- |
| Lookup | How many tables are there? |
| Whole-document | What is this report about? |
| Section | Summarize the water quality section |
| Multi-hop | Compare nitrate levels and explain their relationship with water quality |
| Fact | What was the nitrate level at station C? |

**Purpose:** Decides how to process the question.

## Stage 12. Retrieval and Reranking

For a fact question, the system searches the indexed document.

### Tools and models

- BGE-M3: Dense and sparse retrieval.
- RRF: Combines both result rankings.
- Qdrant: Retrieves relevant chunks.
- bge-reranker-base: Ranks retrieved chunks by relevance.

### Example

Question:

text

```text
"What was the nitrate level at station C?"
```

Retrieval:

text

```text
20 candidate chunks
        ↓
     RRF fusion
        ↓
 bge-reranker-base
        ↓
  Top 6 chunks
```

Suppose the table containing the answer gets the highest score:

text

```text
Table chunk:       0.010
Paragraph chunk:   0.008
Other chunks:      0.003
```

The system checks the best score against `0.005`.

- If below 0.005: Returns "Not found in the documents."
- Otherwise: Continues to answer generation.

**Important:** This threshold is a retrieval gate, not proof of correctness.

## Stage 13. Context Expansion and Answer Generation

**Tool/model:** Python + Qwen2.5-3B-Instruct through llama.cpp

### What happens

The system:

1. Adds the neighbouring item.
2. Adds up to two linked cross-references.
3. Packs passages into a 2,500-token budget using the real tokenizer.
4. Sends the context and question to Qwen.
5. Generates a one- or two-sentence answer.
6. Normalizes citations to `(p. N)`.
7. Logs the question to `data/queries.jsonl`.

### Example

Retrieved table:

| Station | Turbidity | Nitrate | pH |
| --- | --- | --- | --- |
| A | 8.2 | 9.5 | 7.1 |
| B | 7.4 | 10.2 | 7.0 |
| C | 9.6 | 11.3 | 6.8 |

Qwen receives the table and question.

### Generated answer

> The mean nitrate level at station C was 11.3 mg/L. (p. 1)

_Illustrative answer from the example table._

---

# Part 3: Five Practical Questions and What Happens Internally

| Your question | What the system does |
| --- | --- |
| How many tables are there? | Reads `document.json` directly. No Qwen. |
| What is this report about? | Uses whole-document summary and Qwen. |
| Summarize the water quality section. | Uses the relevant section summary and Qwen. |
| What was the nitrate level at station C? | Retrieves and reranks relevant table chunks, then Qwen answers. |
| What does the handwritten note on page 38 say? | Uses the normal fact route; relies on the handwritten chunk being retrieved. No special page filter. |

### Multi-hop questions

For a multi-hop question, Qwen can:

1. Split the question into up to three sub-questions.
2. Retrieve relevant information for each sub-question.
3. Generate a combined answer using the retrieved information.

---

# Part 4: Using the System Through Your Browser

**Tools:** FastAPI + Uvicorn

### Start the server

bash

```bash
python -m rag serve
```

### Open the browser

text

```text
http://127.0.0.1:8000
```

### Available features

You can:

- Upload PDFs.
- Monitor processing.
- View available documents.
- Ask questions.

Everything runs locally on your PC.

---

# Part 5: Testing the System

**Tool:** Python evaluation module

### Run evaluation

bash

```bash
python -m rag eval
```

Uses your `samples/golden.json` questions to measure:

| Metric | What it measures |
| --- | --- |
| Recall@5 | Whether the correct gold page appears in the top five chunks |
| Accuracy | Whether the generated answer contains a gold phrase |
| Handwriting CER | Character error rate against handwriting ground truth |

### Results location

text

```text
data/eval_<time>.json
```

---

# Final Overview: The Entire Architecture

## Your Local PDF AI System

text

```text
                  INPUT
                    |
                    v
        100-Page Mixed-Format PDF
                    |
                    v
          DOCUMENT PROCESSING
                    |
                    v
  PyMuPDF · Docling · RapidOCR · TrOCR
              SmolVLM
                    |
                    v
       Stitching · Enrichment
       Qwen Summarization
             Chunking
                    |
                    v
          KNOWLEDGE DATABASE
                    |
                    v
             BGE-M3
                +
             Qdrant
                    |
                    v
         QUESTION PROCESSING
                    |
                    v
              Routing
                 ↓
             Retrieval
                 ↓
              RRF
                 ↓
             Reranking
                 ↓
             Expansion
                    |
                    v
            FINAL ANSWER
                    |
                    v
             Qwen + Citations
```

## Technology Stack

| Layer | Technology |
| --- | --- |
| Programming language | Python |
| PDF processing | PyMuPDF |
| Document understanding | Docling |
| Printed-text OCR | RapidOCR |
| Handwriting OCR | TrOCR-base-handwritten |
| Layout detection | Heron |
| Table extraction | TableFormer |
| Image understanding | SmolVLM-256M |
| Summarization and answering | Qwen2.5-3B-Instruct |
| Local inference | llama.cpp |
| Embeddings | BGE-M3 |
| Vector database | Qdrant |
| Reranking | bge-reranker-base |
| Backend | FastAPI |
| Server | Uvicorn |
| Hardware | GTX 1650 |

---

# The Key Concept

**Ingestion converts your PDF into structured, searchable knowledge.**

**Question answering retrieves the relevant knowledge and gives it to Qwen to generate an answer.**

Qwen does not need to reread the original 100-page PDF every time you ask a question.

Instead, the system performs the expensive document processing once, stores the extracted knowledge in a searchable database, and retrieves only the relevant information for each new question.

This makes it possible to build a completely local PDF question-answering system with:

- Structured document understanding.
- Printed and handwritten text recognition.
- Table, figure, and formula extraction.
- Semantic and keyword-based retrieval.
- Context-aware answer generation.
- Page-level citations.
- Local processing and storage.

**End result:** A local AI-powered research assistant that can understand, search, and answer questions about complex 100-page environmental research reports without sending the documents to cloud services.

