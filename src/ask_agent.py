import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

DATA_DIR = Path("data/papers")

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60


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
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "source": pdf.name
            })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def query_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

    data = response.json()

    # Debug print (temporary)
    # print("\nDEBUG OLLAMA RESPONSE:\n", data)

    # Handle different formats
    if "response" in data:
        return data["response"]
    elif "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    else:
        return str(data)


def main():
    print("Preparing knowledge base...")

    chunks = load_chunks()
    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding chunks...")
    embeddings = model.encode([c["text"] for c in chunks])

    print("\nResearch agent ready.\n")

    while True:
        query = input("Ask question (type exit to quit): ")

        if query.lower() == "exit":
            break

        # embed query
        q_emb = model.encode([query])[0]

        scores = np.dot(embeddings, q_emb)

        top_k = np.argsort(scores)[-8:][::-1]

        # ⭐ similarity confidence gating
        max_score = scores[top_k[0]]

        print("DEBUG similarity:", max_score)

        if max_score < 0.35:
            print("\nNo relevant information found in research papers.\n")
            continue

        print("\nTop sources:\n")
        for i in top_k:
            print(chunks[i]["source"])

        context = "\n\n".join([chunks[i]["text"] for i in top_k])

        # build prompt
        prompt = f"""
You are a scientific research assistant.

Use ONLY the context below to answer the question.
If the answer is not contained, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

        # query local LLM
        answer = query_llm(prompt)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
