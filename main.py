import pathway as pw
from sentence_transformers import SentenceTransformer
import openai
import os

# ---------------------------
# CONFIG
# ---------------------------
openai.api_key = "YOUR_OPENAI_API_KEY"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

embedder = SentenceTransformer(EMBED_MODEL)

# ---------------------------
# STEP 1: INGEST LONG DOCUMENTS (PATHWAY)
# ---------------------------
documents = pw.io.fs.read(
    path="data/",
    format="text"
)

# ---------------------------
# STEP 2: CHUNKING (ROBUST LONG-CONTEXT HANDLING)
# ---------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = documents.select(
    chunk=pw.apply(chunk_text, documents.text)
).flatten()

# ---------------------------
# STEP 3: EMBEDDINGS (VECTOR STORE)
# ---------------------------
def embed(text):
    return embedder.encode(text).tolist()

embedded_chunks = chunks.select(
    text=chunks.chunk,
    embedding=pw.apply(embed, chunks.chunk)
)

# ---------------------------
# STEP 4: VECTOR INDEX (PATHWAY)
# ---------------------------
index = pw.index.KNNIndex(
    embedded_chunks.embedding,
    embedded_chunks.text
)

# ---------------------------
# STEP 5: QUERY + RETRIEVAL
# ---------------------------
def retrieve_evidence(query):
    query_vec = embed(query)
    results = index.query(query_vec, k=TOP_K)
    return [r[1] for r in results]

# ---------------------------
# STEP 6: EVIDENCE-GROUNDED GENERATION
# ---------------------------
def answer_query(query):
    evidence = retrieve_evidence(query)

    prompt = f"""
You are an evidence-grounded reasoning assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in provided documents."

Context:
{chr(10).join(evidence)}

Question:
{query}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response["choices"][0]["message"]["content"]

# ---------------------------
# STEP 7: DEMO QUERY
# ---------------------------
if __name__ == "__main__":
    query = "Describe the main conflict in the story."
    print("Answer:\n", answer_query(query))
