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

# EN: Define absolute paths
# FR: Définition des chemins absolus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "stored_chunks.pkl")

# EN: Fail-safe to prevent running the app without the required vector database.
# FR: Sécurité pour empêcher le lancement de l'application sans la base vectorielle.
if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        "Saved embeddings not found. Run build_embeddings.py first."
    )


# EN: Load the multilingual embedding model and the FAISS index.
# FR: Chargement du modèle d'embedding multilingue et de l'index FAISS.
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index(INDEX_PATH)

# EN: Load the serialized chunk metadata (text + page numbers).
# FR: Chargement des métadonnées des chunks sérialisés (texte + numéros de page).
with open(CHUNKS_PATH, "rb") as f:
    stored_chunks = pickle.load(f)

# EN: Define stopwords for keyword chunking.
# FR: Définition des mots-stop pour le chunking par mots-clés.
STOPWORDS = {
    "quel", "quelle", "quels", "quelles", "dans", "pour", "avec", "selon",
    "est", "sont", "une", "des", "les", "du", "de", "par", "rapport",
    "sur", "aux", "cas", "etude", "camion", "electrique"
}


# EN: Helper function to extract chunk text.
# FR: Fonction auxiliaire pour extraire le texte du chunk.
def get_chunk_text(chunk):
    return chunk["text"] if isinstance(chunk, dict) else chunk


# EN: Helper function to extract chunk page number.
# FR: Fonction auxiliaire pour extraire le numéro de page du chunk.
def get_chunk_page(chunk):
    return chunk.get("page") if isinstance(chunk, dict) else None


def extract_page_labels(retrieved_text):
    seen = []
    for page in re.findall(r"\[Page (\d+)\]", retrieved_text):
        label = f"[Page {page}]"
        if label not in seen:
            seen.append(label)
    return seen

# EN: Normalize text by removing accents and converting to lowercase for better matching.
# FR: Normalisation du texte en supprimant les accents et en convertissant en minuscules pour une meilleure correspondance.
def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()

# EN: Extract keywords from the question for keyword chunking.
# FR: Extraction des mots-clés de la question pour le chunking par mots-clés.
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

# EN: Core retrieval logic: Hybrid Search (Semantic/Vector + Lexical).
# FR: Logique de recherche centrale: Recherche hybride (Sémantique/Vectorielle + Lexicale).
def retrieve_chunks(question, top_k=9, distance_threshold=120.0):
    question_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(question_embedding, top_k)

    selected_indices = []
    for dist, idx in zip(distances[0], indices[0]):
        # EN: Filter out low-quality matches based on L2 distance threshold.
        # FR: Filtrer les correspondances de qualité faible basées sur le seuil de distance L2.
        if idx == -1 or dist > distance_threshold:
            continue  # skip invalid or low-quality matches
        if idx not in selected_indices:
            selected_indices.append(idx)

    # EN:  Inject Lexical Search Results for better context.
    # FR: Injection des résultats de la recherche lexicale pour un meilleur contexte.
    for idx in keyword_chunk_indices(question):
        if idx not in selected_indices:
            selected_indices.append(idx)

        # EN: Add the immediate next chunk to preserve trailing context.
        # FR: Ajout du chunk immédiatement suivant pour préserver le contexte de fin de phrase.
        next_idx = idx + 1
        if next_idx < len(stored_chunks) and next_idx not in selected_indices:
            selected_indices.append(next_idx)
    # EN: Format the retrieved chunks with  page labels for the LLM.
    # FR: Formatage des chunks récupérés avec des labels de page pour le LLM.
    formatted_chunks = []
    for idx in selected_indices:
        chunk = stored_chunks[idx]
        page = get_chunk_page(chunk)
        page_label = f"Page {page}" if page is not None else f"Chunk {idx}"
        formatted_chunks.append(f"[{page_label}]\n{get_chunk_text(chunk)}")

    return "\n\n".join(formatted_chunks)

# EN: Load API key from .env file
# FR: Chargement de la clé API depuis le fichier .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Handle if API key is not present
# FR: Gérer le cas où la clé API n'est pas présente
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

# EN: Generate answer using the LLM.
# FR: Générer la réponse en utilisant le LLM.
def generate_answer(question, retrieved_text):
    if not retrieved_text:
        # EN: If no chunks are retrieved, yield a fallback message.
        # FR: Si aucun chunk n'est récupéré, générer un message de secours.
        yield "This information is not available in the document."
        return

    # EN: Extract page labels from the retrieved chunks.
    # FR: Extraction des labels de page des chunks récupérés.
    available_pages = extract_page_labels(retrieved_text)


    # EN: Define the system prompt for the LLM.
    # FR: Définition du prompt système pour le LLM.
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
    # EN: Define the user prompt for the LLM.
    # FR: Définition du prompt utilisateur pour le LLM.
    user_prompt = f"Retrieved chunks:\n{retrieved_text}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        # EN: Define the model to use for the LLM.
        # FR: Définition du modèle à utiliser pour le LLM.
        model="llama-3.1-8b-instant",
        messages=[
            # EN: Define the system message for the LLM.
            # FR: Définition du message système pour le LLM.
            {
                "role": "system",
                "content": system_prompt
            },
            # EN: Define the user message for the LLM.
            # FR: Définition du message utilisateur pour le LLM.
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        # EN: Define the temperature for the LLM.
        # FR: Définition de la température pour le LLM.
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        # EN: Define the stream for the LLM.
        # FR: Définition du stream pour le LLM.
        stream=True,
        # EN: Define the stop for the LLM.
        # FR: Définition du stop pour le LLM.
        stop=None
    )

    # EN: Collect the answer parts from the LLM.
    # FR: Collecte des parties de la réponse du LLM.
    answer_parts = []
    for chunk in completion:
        # EN: Append the content of the chunk to the answer parts.
        # FR: Ajout du contenu du chunk à la réponse.
        answer_parts.append(chunk.choices[0].delta.content or "")

    # EN: Join the answer parts and strip any whitespace.
    # FR: Joindre les parties de la réponse et supprimer les espaces blancs.
    answer = "".join(answer_parts).strip()

    # EN: If the answer is not empty, and not the fallback phrase, and has available pages, append the pages to the answer.
    # FR: Si la réponse n'est pas vide, et pas le message de secours, et a des pages disponibles, ajouter les pages à la réponse.
    if (
        answer
        and answer != "Cette information n'est pas disponible dans le document."
        and not re.search(r"\[Page \d+\]", answer)
        and available_pages
    ):
        answer = f"{answer} {' '.join(available_pages)}"

    yield answer


# EN: Define the question input for the user.
# FR: Définition de la question entrée par l'utilisateur.
question = st.text_input("Question")

# EN: Define the submit button for the user.
# FR: Définition du bouton soumettre pour l'utilisateur.
if st.button("Submit") and question:
    # EN: Retrieve the chunks from the FAISS index.
    # FR: Récupération des chunks de la base vectorielle.
    context = retrieve_chunks(question, top_k=5)
    st.write_stream(generate_answer(question, context))
