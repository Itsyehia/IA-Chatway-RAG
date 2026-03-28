import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# EN: Define the path to the PDF file.
# FR: Définition du chemin vers le fichier PDF.
PDF_PATH = "MOBILITY-CHATWAY_data_rapport-camion-elec-carbone4.pdf"
# EN: Define the path to the output text file.
# FR: Définition du chemin vers le fichier texte de sortie.
OUTPUT_TXT = "carbone4_raw_text.txt"

# EN: Optional: set this manually on Windows if pytesseract can't find tesseract
# FR: Optionnel: définir manuellement sur Windows si pytesseract ne peut pas trouver tesseract
pytesseract.pytesseract.tesseract_cmd = r"F:\Tessercat\tesseract.exe"

# EN: OCR language:
# FR: Langue OCR:
OCR_LANG = "fra"

# If direct extracted text is shorter than this, fallback to OCR
MIN_TEXT_LENGTH = 50

# EN: Extract text from the PDF file.
# FR: Extraire le texte du fichier PDF.
def extract_text_from_pdf(pdf_path: str, output_txt: str) -> str:
    doc = fitz.open(pdf_path)
    # EN: Create a list of all pages text.
    # FR: Créer une liste de tous les textes des pages.
    all_pages_text = []

    for page_index in range(len(doc)):
        # EN: Load the page.
        # FR: Charger la page.
        page = doc.load_page(page_index)
        page_number = page_index + 1
        
        # EN: Extract the text from the page.
        # FR: Extraire le texte de la page.

        # 1) Try native PDF text extraction first, keeping block order for tables/layout
        # FR: Essayer d'abord l'extraction de texte native PDF, en conservant l'ordre des blocs pour les tables/layout
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda block: (block[1], block[0]))
        text = "\n".join(block[4].strip() for block in blocks if block[4].strip()).strip()

        # 2) Fallback to OCR if page is image-based / nearly empty
        if len(text) < MIN_TEXT_LENGTH:
            pix = page.get_pixmap(dpi=300)
            mode = "RGB" if pix.alpha == 0 else "RGBA"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            # OCR
            text = pytesseract.image_to_string(img, lang=OCR_LANG).strip()

        # Add clear page separators so later you can keep track of source pages
        page_text = f"\n\n===== PAGE {page_number} =====\n{text}\n"
        all_pages_text.append(page_text)

        print(f"Processed page {page_number}/{len(doc)}")

    full_text = "".join(all_pages_text)

    # EN: Save the full text to the output file.
    # FR: Enregistrer le texte complet dans le fichier de sortie.
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    # EN: Close the document.
    # FR: Fermer le document.
    doc.close()
    # EN: Return the full text.
    # FR: Retourner le texte complet.
    return full_text

# EN: Main function.
# FR: Fonction principale.
if __name__ == "__main__":
    # EN: Extract the text from the PDF file.
    # FR: Extraire le texte du fichier PDF.
    raw_text = extract_text_from_pdf(PDF_PATH, OUTPUT_TXT)
    # EN: Print the path to the output file.
    # FR: Afficher le chemin vers le fichier de sortie.
    print(f"\nDone. Text saved to: {OUTPUT_TXT}")
    # EN: Print the number of extracted characters.
    # FR: Afficher le nombre de caractères extraits.
    print(f"Extracted characters: {len(raw_text)}")