import pickle
import re
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

from langchain_core.documents import Document


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "stored_chunks.pkl")

if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        "Saved embeddings not found. Run build_embeddings.py first."
    )

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    stored_chunks = pickle.load(f)

STOPWORDS = {
    "quel", "quelle", "quels", "quelles", "dans", "pour", "avec", "selon",
    "est", "sont", "une", "des", "les", "du", "de", "par", "rapport",
    "sur", "aux", "cas", "etude", "camion", "electrique"
}


def get_chunk_text(chunk):
    return chunk["text"] if isinstance(chunk, dict) else chunk


def get_chunk_page(chunk):
    return chunk.get("page") if isinstance(chunk, dict) else None


def extract_page_labels(retrieved_text):
    seen = []
    for page in re.findall(r"\[Page (\d+)\]", retrieved_text):
        label = f"[Page {page}]"
        if label not in seen:
            seen.append(label)
    return seen


def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def keyword_chunk_indices(question, top_k=3):
    normalized_question = normalize_text(question)
    keywords = [
        word for word in re.findall(r"[a-z]+", normalized_question)
        if len(word) >= 4 and word not in STOPWORDS
    ]

    scored_chunks = []
    for idx, chunk in enumerate(stored_chunks):
        normalized_chunk = normalize_text(get_chunk_text(chunk))
        score = sum(1 for word in keywords if word in normalized_chunk)
        if score > 0:
            scored_chunks.append((score, idx))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in scored_chunks[:top_k]]


def retrieve_chunks(question, top_k=9, distance_threshold=120.0):
    question_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(question_embedding, top_k)

    selected_indices = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or dist > distance_threshold:
            continue  # skip invalid or low-quality matches
        if idx not in selected_indices:
            selected_indices.append(idx)

    for idx in keyword_chunk_indices(question):
        if idx not in selected_indices:
            selected_indices.append(idx)
        next_idx = idx + 1
        if next_idx < len(stored_chunks) and next_idx not in selected_indices:
            selected_indices.append(next_idx)

    formatted_chunks = []
    for idx in selected_indices:
        chunk = stored_chunks[idx]
        page = get_chunk_page(chunk)
        page_label = f"Page {page}" if page is not None else f"Chunk {idx}"
        formatted_chunks.append(f"[{page_label}]\n{get_chunk_text(chunk)}")

    return "\n\n".join(formatted_chunks)

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# Handle if API key is not present
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


def generate_answer(question, retrieved_text):
    if not retrieved_text:
        yield "This information is not available in the document."
        return

    available_pages = extract_page_labels(retrieved_text)
    system_prompt = (
    "You are a strict data extraction assistant. "
    "Answer the question using ONLY the facts explicitly stated in the provided retrieved chunks. "
    "Every answer must include source citations using the exact page format [Page X]. "
    "CRITICAL RULES: "
    "1. NO DEDUCTION: Do not use your pre-trained world knowledge to deduce or infer answers. If the text mentions a word (like 'Paris') but does not explicitly answer the user's question about it, you cannot draw conclusions. "
    "2. FALLBACK: If the answer cannot be found or assembled only from explicit facts written in the chunks, you MUST output EXACTLY and ONLY this phrase: "
    "\"Cette information n'est pas disponible dans le document.\" "
    "3. NO CHAT: Do not explain your reasoning. Do not write things like 'Puisque le document ne contient pas...' or 'La réponse est'. Just output the facts or the fallback phrase. "
    "If you provide an answer, never append the fallback phrase after it. "
    "4. Output the final answer in French. "
    "5. Do not use chunk numbers or invent citations. Only cite the page labels provided in the retrieved chunks. "
    "6. If you provide an answer, at least one [Page X] citation is mandatory."
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
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )

    answer_parts = []
    for chunk in completion:
        answer_parts.append(chunk.choices[0].delta.content or "")

    answer = "".join(answer_parts).strip()

    if (
        answer
        and answer != "Cette information n'est pas disponible dans le document."
        and not re.search(r"\[Page \d+\]", answer)
        and available_pages
    ):
        answer = f"{answer} {' '.join(available_pages)}"

    yield answer


question = st.text_input("Question")

if st.button("Submit") and question:
    context = retrieve_chunks(question, top_k=5)
    st.write_stream(generate_answer(question, context))
