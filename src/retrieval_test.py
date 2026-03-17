import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_DIR = Path("data/papers")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_chunks():
    chunks = []

    for pdf in DATA_DIR.glob("*.pdf"):
        doc = fitz.open(pdf)
        text = ""
        for page in doc:
            text += page.get_text()

        words = text.split()

        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding chunks...")
    embeddings = model.encode(chunks)

    query = input("Ask question: ")

    q_emb = model.encode([query])[0]

    scores = np.dot(embeddings, q_emb)

    top_k = np.argsort(scores)[-3:][::-1]

    print("\nTop results:\n")

    for i in top_k:
        print("----")
        print(chunks[i][:500])
        print()


if __name__ == "__main__":
    main()
