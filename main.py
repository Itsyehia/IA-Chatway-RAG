# !pip install langchain langchain-openai chroma db
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_core.documents import Document

# pip install langchain-core langchain-text-splitters

import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load raw text
with open("carbone4_raw_text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# 2. Parse pages using markers
pages = re.split(r"===== PAGE (\d+) =====", raw_text)

documents = []

# pages format: ["", "1", "text...", "2", "text...", ...]
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


# 3. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)


# Testing splitting chunks 
# 4. Print summary
# print(f"Total pages parsed: {len(documents)}")
# print(f"Total chunks created: {len(chunks)}")


# 5. Inspect first chunks
# for i, chunk in enumerate(chunks[:5], start=1):
#     print("\n" + "="*60)
#     print(f"Chunk {i}")
#     print(f"Page: {chunk.metadata['page']}")
#     print(chunk.page_content[:300])  # preview only


# 3. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(chunk_texts, convert_to_numpy=True).astype("float32")


# 4 Store in vector database
# Initialize a FAISS index with the embedding dimension.
index = faiss.IndexFlatL2(embeddings.shape[1])

# Add all chunk vectors to the index in the same order as chunk_texts.
index.add(embeddings)
stored_chunks = chunk_texts


def retrieve_chunks(question, top_k=3, distance_threshold=120.0):
    question_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(question_embedding, top_k)

    # If the best match is too far away, treat it as out of scope.
    if len(indices[0]) == 0 or indices[0][0] == -1 or distances[0][0] > distance_threshold:
        return ""

    retrieved = []
    for idx in indices[0]:
        if idx != -1:
            retrieved.append(f"[Chunk {idx}]\n{stored_chunks[idx]}")

    return "\n\n".join(retrieved)

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# Handle if API key is not present
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


def generate_answer(question, retrieved_text):
    if not retrieved_text:
        print("This information is not available in the document.")
        return

    # system_prompt = (
    #     "Answer the question using ONLY the provided retrieved chunks. "
    #     "Include citations pointing to the information in the text. "
    #     'If the retrieved chunks do not contain the answer, or if the retrieval confidence is too low, output exactly: "This information is not available in the document." '
    #     "Do not guess."
    #     "Output the answer in French without any conversational filters."
    # )
    system_prompt = (
    "You are a strict data extraction assistant. "
    "Answer the question using ONLY the facts explicitly stated in the provided retrieved chunks. "
    "Include citations pointing to the information in the text. "
    "CRITICAL RULES: "
    "1. NO DEDUCTION: Do not use your pre-trained world knowledge to deduce or infer answers. If the text mentions a word (like 'Paris') but does not explicitly answer the user's question about it, you cannot draw conclusions. "
    "2. FALLBACK: If the exact answer is not explicitly written in the chunks, you MUST output EXACTLY and ONLY this phrase: "
    "\"Cette information n'est pas disponible dans le document.\" "
    "3. NO CHAT: Do not explain your reasoning. Do not write things like 'Puisque le document ne contient pas...' or 'La réponse est'. Just output the facts or the fallback phrase. "
    "4. Output the final answer in French."
)

    user_prompt = f"Retrieved chunks:\n{retrieved_text}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

    print()


questions = [
    "quelle est la capitale de la France ?",
    "Quelle est l’autonomie maximale d’un camion électrique de 16 tonnes en 2022 dans le cas d’étude ?",
    "Quel est le coût de maintenance par kilomètre d’un camion électrique par rapport à un camion diesel, selon le modèle TCO présenté dans l’étude ?",
    "En combinant le dispositif de suramortissement et le bonus écologique, quel est l’écart de TCO entre un camion électrique et un camion diesel ?",
    "Quel est l’impact du camion électrique sur le transport longue distance, au-delà de 500 km ?",
    "Quels sont les trois principaux freins au déploiement du camion électrique identifiés dans l’étude, et quelles solutions sont recommandées pour chacun d’entre eux ?",
]


for i, question in enumerate(questions, start=1):
    print(f"\nQ{i}: {question}")
    context = retrieve_chunks(question, top_k=3)
    generate_answer(question, context)
