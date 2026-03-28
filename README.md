# IA-Chatway-RAG

Clone: gh repo clone Itsyehia/IA-Chatway-RAG

This project is a simple Retrieval-Augmented Generation (RAG) application built around a local PDF processing pipeline and a minimal Streamlit frontend.

The workflow is:
- extract text from a PDF
- preserve page boundaries in a raw text file
- split the content into chunks
- generate embeddings for those chunks
- store the vectors in a local FAISS index
- retrieve relevant chunks for a user question
- send the retrieved context to a language model
- display the final answer in a Streamlit page

## Technology Stack

The project currently uses the following technologies:

- Python: main programming language for the full pipeline
- Streamlit: minimal web frontend for question input and answer display
- Groq API via `groq.Groq`: LLM inference and answer generation
- Sentence Transformers: embedding model generation with `paraphrase-multilingual-MiniLM-L12-v2`
- FAISS: local vector index and similarity search
- LangChain text splitters: chunking the extracted document text
- PyMuPDF (`fitz`): PDF text extraction
- pytesseract + Pillow: OCR fallback for image-based PDF pages
- python-dotenv: environment variable loading from `.env`
- pickle: local persistence for chunk metadata

## Project Structure

`main.py`
- Main application entry point.
- Loads the FAISS index and stored chunks.
- Retrieves relevant chunks for a question.
- Calls the Groq model to generate an answer.
- Runs the Streamlit interface.

`build_embeddings.py`
- Reads the raw extracted text file.
- Splits the document into page-aware chunks.
- Generates embeddings using Sentence Transformers.
- Saves the vectors to `embeddings.index`.
- Saves chunk text and page metadata to `stored_chunks.pkl`.

`Extract_Text_from_PDF.py`
- Extracts text from the source PDF.
- Uses native PDF text extraction first.
- Falls back to OCR with Tesseract when a page has too little extracted text.
- Writes the output to `carbone4_raw_text.txt` with page separators.

`carbone4_raw_text.txt`
- Intermediate extracted text file created from the PDF.
- Keeps explicit page markers such as `===== PAGE X =====`.
- Serves as the input to the embedding build step.

`embeddings.index`
- Local FAISS vector index built from the chunk embeddings.
- Used at runtime for semantic retrieval.

`stored_chunks.pkl`
- Local serialized chunk store.
- Contains chunk text plus page metadata used for citations.

`requirements.txt`
- Python dependency list for the project.

`.env`
- Stores environment variables such as `GROQ_API_KEY`.


`.gitignore`
- Prevents sensitive files from being tracked.

`answers.txt`
- Answers of the required questions

## Prerequisites

Before running the project, make sure you have:

- Python 3.10+ installed
- `pip` available
- Tesseract OCR installed on your machine
- a valid Groq API key

On Windows, also make sure the Tesseract executable path in `Extract_Text_from_PDF.py` matches your local installation.

## Installation and Setup

1. Clone or open the project locally.
2. Move into the project directory:

```powershell
cd IA-Chatway\IA-Chatway-RAG
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_api_key_here
```

5. If needed, update the Tesseract path in `Extract_Text_from_PDF.py`.

## Running the Application

If you need to regenerate the extracted text from the PDF:

```powershell
python Extract_Text_from_PDF.py
```

If you need to rebuild the embeddings and chunk metadata:

```powershell
python build_embeddings.py
```

Run the Streamlit application with:

```powershell
python -m streamlit run main.py
```

## How It Works

1. `Extract_Text_from_PDF.py` extracts or OCRs text from the PDF and writes a page-separated text file.
2. `build_embeddings.py` reads that text, creates page-aware chunks, embeds them, and stores both vectors and chunk metadata.
3. `main.py` embeds the user question, retrieves the most relevant chunks from FAISS, and sends them to the LLM.
4. The LLM answers in French using only the retrieved chunks, and the app displays the answer in Streamlit.

## Future Work and Improvements

Ps: The tools mentioned here just blueplrints And it is not restricted to these specific provider, as the core logic can be implemented using any similar technology

### 1. Replace Groq with Azure OpenAI Service

Current implementation:
- The project currently uses `groq.Groq` with the `llama-3.1-8b-instant` model for answer generation.
- This is the current "brain" of the system.

Possible improvement:
- Replace the Groq client with `openai.AzureOpenAI`.
- This would move text generation and orchestration to Azure OpenAI Service.
- This will allow better responses faster processing.

### 2. Replace Manual Local Retrieval with Azure AI Search

Current implementation:
- Retrieval is currently done manually in Python.
- `SentenceTransformer` generates embeddings locally.
- FAISS stores and searches vectors locally in `embeddings.index`.
- `main.py` retrieves chunks directly from the FAISS index and then sends them to the LLM.

Possible improvement:
- Replace the local FAISS retrieval layer with Azure AI Search.
- Instead of searching vectors manually in Python, the application could use Azure AI Search as the retrieval engine.
- In an Azure-native architecture, retrieved chunks could be provided through Azure search integration instead of local index files.
- This would make indexing, filtering, scaling, and operational monitoring easier, faster and automated for production usage.

Note:
- The current code does not use `data_sources`, Azure Search endpoints, Azure Search keys, or `SearchIndexerClient`.
- Those would be part of a future cloud implementation, not the current local version.

### 3. Replace Local File Storage with Azure Blob Storage

Current implementation:
- The PDF is expected to be available locally in the project folder.
- The extracted text, embeddings, and chunk metadata are also stored locally as files.

Possible improvement:
- Store source PDFs in Azure Blob Storage instead of keeping them in the local project folder.
- This would allow centralized document storage and easier document updates.
- The ingestion pipeline could then read documents directly from cloud storage before OCR, text extraction, chunking, and embedding.

### 4. Add CI/CD and Automated Document Processing

Current implementation:
- OCR, text extraction, chunk generation, and embedding generation are manual steps.
- When a document changes, the developer must rerun the scripts manually.

Possible improvement:
- Add a CI/CD pipeline to automate ingestion and deployment.
- Trigger processing whenever a new document is uploaded or an existing one is updated.
- Automate the sequence:
  - document upload
  - OCR / text extraction
  - text transformation and chunking
  - embedding generation
  - index refresh
  - application deployment or refresh
- In an Azure-based version, this could be connected to Blob Storage events, Azure Functions, Azure AI Search indexing, and Azure OpenAI inference endpoints.

## Notes

- If `main.py` says embeddings are missing, run `python build_embeddings.py` first.
- If OCR fails, verify that Tesseract is installed correctly and the configured path is valid.
- If the Streamlit command is not recognized, use `python -m streamlit run main.py`.
