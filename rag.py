import os
import uuid
from collections import deque

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from endee import Endee, Precision


class SimpleRAG:
    def __init__(self):
        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Local LLM (Mistral via ctransformers)
        self.llm = AutoModelForCausalLM.from_pretrained(
            "models",
            model_file="mistral.gguf",
            model_type="mistral"
        )

        # Endee client setup
        self.endee = Endee()
        self.endee.set_base_url("http://localhost:8080/api/v1")

        self.index_name = "rag_index"
        self.vector_dim = 384

        self._init_index()
        self.index = self.endee.get_index(name=self.index_name)

        # Keep last 3 Q&A turns
        self.memory = deque(maxlen=3)

    # -------------------------
    # Index initialization
    # -------------------------
    def _init_index(self):
        try:
            self.endee.create_index(
                name=self.index_name,
                dimension=self.vector_dim,
                space_type="cosine",
                precision=Precision.INT8
            )
        except Exception:
            pass

    # -------------------------
    # Document ingestion
    # -------------------------
    def ingest(self, docs_path="data/docs"):
        combined_text = ""

        for fname in os.listdir(docs_path):
            if fname.lower().endswith(".pdf"):
                reader = PdfReader(os.path.join(docs_path, fname))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text

        chunks = self._split_text(combined_text)
        vectors = self.embedding_model.encode(chunks)

        payload = []
        for vec, chunk in zip(vectors, chunks):
            payload.append({
                "id": str(uuid.uuid4()),
                "vector": vec.tolist(),
                "meta": {"text": chunk}
            })

        self.index.upsert(payload)

    # -------------------------
    # Query vector search
    # -------------------------
    def retrieve(self, query, k=2):
        query_vec = self.embedding_model.encode([query])[0]

        results = self.index.query(
            vector=query_vec.tolist(),
            top_k=k
        )

        return "\n".join(r["meta"]["text"] for r in results)

    # -------------------------
    # Ask with RAG
    # -------------------------
    def ask(self, question):
        context = self.retrieve(question)

        history = "\n".join(
            f"Q: {h['q']}\nA: {h['a']}" for h in self.memory
        )

        prompt = f"""
Answer the question strictly using the context below.
If the answer is not found, say "I don't know based on the document".

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.llm(prompt, max_new_tokens=120)

        self.memory.append({
            "q": question,
            "a": answer
        })

        return answer

    # -------------------------
    # Text chunking
    # -------------------------
    def _split_text(self, text, size=400, overlap=50):
        parts = []
        i = 0
        while i < len(text):
            parts.append(text[i:i + size])
            i += size - overlap
        return parts


if __name__ == "__main__":
    rag = SimpleRAG()

    print("Type 'ingest' to load documents or ask a question.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() == "ingest":
            rag.ingest()
            print("Documents ingested.")
        elif user_input.lower() in ("exit", "quit"):
            break
        else:
            response = rag.ask(user_input)
            print("\n", response)