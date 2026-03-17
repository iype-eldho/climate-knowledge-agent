import fitz
from pathlib import Path

DATA_DIR = Path("data/papers")

def load_pdfs():
    texts = []
    for pdf in DATA_DIR.glob("*.pdf"):
        doc = fitz.open(pdf)
        text = ""
        for page in doc:
            text += page.get_text()
        texts.append(text)
        print(f"Loaded {pdf.name}, chars={len(text)}")
    return texts

if __name__ == "__main__":
    load_pdfs()
