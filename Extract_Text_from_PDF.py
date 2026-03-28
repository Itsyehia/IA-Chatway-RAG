
# pip install pymupdf pytesseract pillow
# Also install Tesseract OCR on your system:
# Windows: install Tesseract and set the path below if needed
# Linux: sudo apt install tesseract-ocr tesseract-ocr-fra
# Mac: brew install tesseract tesseract-lang

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image


PDF_PATH = "MOBILITY-CHATWAY_data_rapport-camion-elec-carbone4.pdf"
OUTPUT_TXT = "carbone4_raw_text.txt"

# Optional: set this manually on Windows if pytesseract can't find tesseract
pytesseract.pytesseract.tesseract_cmd = r"F:\Tessercat\tesseract.exe"

# OCR language:
# "fra" for French only
# "fra+eng" if you want French + English
OCR_LANG = "fra"

# If direct extracted text is shorter than this, fallback to OCR
MIN_TEXT_LENGTH = 50


def extract_text_from_pdf(pdf_path: str, output_txt: str) -> str:
    doc = fitz.open(pdf_path)
    all_pages_text = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_number = page_index + 1

        # 1) Try native PDF text extraction first, keeping block order for tables/layout
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

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    doc.close()
    return full_text


if __name__ == "__main__":
    raw_text = extract_text_from_pdf(PDF_PATH, OUTPUT_TXT)
    print(f"\nDone. Text saved to: {OUTPUT_TXT}")
    print(f"Extracted characters: {len(raw_text)}")