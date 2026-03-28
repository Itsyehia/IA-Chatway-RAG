import os
import pickle
import re

import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_TEXT_PATH = os.path.join(BASE_DIR, "carbone4_raw_text.txt")
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "stored_chunks.pkl")


with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

pages = re.split(r"===== PAGE (\d+) =====", raw_text)

documents = []
for i in range(1, len(pages), 2):
    page_number = int(pages[i])
    page_text = pages[i + 1].strip()

    if page_text:
        documents.append(
            Document(
                page_content=page_text,
                metadata={"page": page_number}
            )
        )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_records = [
    {
        "text": chunk.page_content,
        "page": chunk.metadata.get("page")
    }
    for chunk in chunks
]

# to support multiple languages ( french in this case )
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunk_records, f)

print(f"Saved FAISS index to: {INDEX_PATH}")
print(f"Saved chunks to: {CHUNKS_PATH}")
