import os
import pickle
import re

import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# EN: Define absolute paths
# FR: Définition des chemins absolus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_TEXT_PATH = os.path.join(BASE_DIR, "carbone4_raw_text.txt")
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "stored_chunks.pkl")

# EN: Open the raw text file and read the content.
# FR: Ouvrir le fichier texte brut et lire le contenu.
with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

# EN: Split the raw text into pages.
# FR: Séparer le texte brut en pages.
pages = re.split(r"===== PAGE (\d+) =====", raw_text)

documents = []
# EN: Create a list of documents for each page.
# FR: Créer une liste de documents pour chaque page.
for i in range(1, len(pages), 2):
    page_number = int(pages[i])
    page_text = pages[i + 1].strip()
    # EN: If the page text is not empty, add the page to the documents list.
    # FR: Si le texte de la page n'est pas vide, ajouter la page à la liste des documents.
    if page_text:
        documents.append(
            Document(
                page_content=page_text,
                metadata={"page": page_number}
            )
        )

# EN: Define the text splitter for the chunks.
# FR: Définition du splitteur de texte pour les chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# EN: Split the documents into chunks.
# FR: Séparer les documents en chunks.
chunks = text_splitter.split_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]

# EN: Create a list of chunk records for each chunk.
# FR: Créer une liste de records de chunk pour chaque chunk.
chunk_records = [
    {
        "text": chunk.page_content,
        "page": chunk.metadata.get("page")
    }
    for chunk in chunks
]

# EN: Create a model for the embeddings.
# FR: Créer un modèle pour les embeddings.
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# EN: Create an index for the embeddings.
# FR: Créer un index pour les embeddings.
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# EN: Save the index to the file.
# FR: Enregistrer l'index dans le fichier.
faiss.write_index(index, INDEX_PATH)

# EN: Save the chunk records to the file.
# FR: Enregistrer les records de chunk dans le fichier.
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunk_records, f)

# EN: Print the paths of the saved files.
# FR: Afficher les chemins des fichiers enregistrés.
print(f"Saved FAISS index to: {INDEX_PATH}")
print(f"Saved chunks to: {CHUNKS_PATH}")
