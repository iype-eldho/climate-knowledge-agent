import os
import pickle
import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import faiss
import re


DATA_DIR = Path("data/papers")

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"


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
        "temperature": 0.1,
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
    
def is_valid_answer(answer):
    import re

    sentences = [s.strip() for s in answer.split(".") if s.strip()]

    if not (2 <= len(sentences) <= 4):
        return False

    for s in sentences:
        # MUST contain [number]
        if not re.search(r"\[\d+\]", s):
            return False

        # MUST NOT contain author-style citations
        if re.search(r"\([A-Za-z].*\d{4}.*\)", s):
            return False

    return True

def main():
    print("Preparing knowledge base...")

    chunks = load_chunks()
    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing FAISS index...")
        index = faiss.read_index(INDEX_FILE)

        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)

    else:
        print("Embedding chunks...")
        embeddings = model.encode([c["text"] for c in chunks])

        # normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        print("Saving FAISS index...")
        faiss.write_index(index, INDEX_FILE)

        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)

    print("\nResearch agent ready.\n")

    while True:
        query = input("Ask question (type exit to quit): ")

        if query.lower() == "exit":
            break

        # embed + normalize query
        q_emb = model.encode([query])[0]
        q_emb = q_emb / np.linalg.norm(q_emb)

        # FAISS search
        scores, indices = index.search(
            q_emb.reshape(1, -1).astype(np.float32), 8
        )

        top_k = indices[0]
        scores = scores[0]

        # similarity gating
        max_score = scores[0]
        print("DEBUG similarity:", max_score)

        if max_score < 0.35:
            print("\nNo relevant information found in research papers.\n")
            continue

        print("\nTop sources:\n")

        seen = set()
        for i, score in zip(top_k, scores):
            src = chunks[i]["source"]
            if src not in seen:
                print(f"{src}  (score: {score:.3f})")
                seen.add(src)

        numbered_chunks = []
        for idx, i in enumerate(top_k):
            numbered_chunks.append(f"[{idx+1}] ({chunks[i]['source']})\n{chunks[i]['text']}")

        context = ""
        for idx, i in enumerate(top_k):
            context += f"[{idx+1}] {chunks[i]['text']}\n\n"

        prompt = f"""
        You are a scientific research assistant.

        STRICT RULES:
        1. Use ONLY the provided context.
        2. You MUST use ONLY citation format [1], [2], etc.
        3. DO NOT use author-year citations like (Smith et al., 2020).
        4. Every sentence MUST contain at least one [number] citation.
        5. If ANY sentence violates this → output is invalid.
        6. If answer not found: write exactly "Not found in provided papers."

        OUTPUT FORMAT:
        - 2–4 sentences ONLY
        - Each sentence MUST end with citation(s)

        GOOD EXAMPLE:
        IMERG is a satellite product [1].

        BAD EXAMPLE (DO NOT DO THIS). This answer is invalid because it contains sentences without citations ([1] is present):
        IMERG is a satellite product.   

        You will be graded strictly. Answers without citations will be rejected.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        answer = query_llm(prompt)

        if not is_valid_answer(answer):
            print("\n⚠️ Invalid format (missing citations). Retrying...\n")

            retry_prompt = prompt + "\nREMEMBER: EVERY sentence MUST include [1] style citations."

            answer = query_llm(retry_prompt)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")

        print("\nSources used:\n")
        seen = set()
        for idx, i in enumerate(top_k):
            src = chunks[i]["source"]
            if src not in seen:
                print(f"[{idx+1}] {src}")
                seen.add(src)    

if __name__ == "__main__":
    main()
