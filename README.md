# Offline RAG System with Endee Vector Database

## Project Overview and Problem Statement
Retrieval-Augmented Generation (RAG) models have become essential for allowing Large Language Models (LLMs) to answer questions based on specific knowledge bases. However, doing so while maintaining privacy and keeping costs low is a challenge. This project aims to solve this by building a completely offline RAG application that reads local PDF documents, indexes their contents utilizing a high-performance vector database, and generates knowledgeable responses using an offline open-source LLM.

## System Design and Technical Approach
The application follows a standard modular RAG architecture but is designed to run entirely offline on standard consumer hardware:
1. **Document Ingestion**: The system uses `pypdf` to parse text from local PDF files situated in the `data/docs` directory.
2. **Chunking and Embedding**: The parsed text is split into overlapping chunks (size 400, overlap 50) and vectorized using the lightweight `sentence_transformers` (`all-MiniLM-L6-v2`).
3. **Storage and Retrieval**: The vector embeddings are upserted into an **Endee** vector database running locally. Upon receiving a query, Endee returns the top-k most semantically similar chunks based on cosine distance.
4. **Generation**: We use `ctransformers` to load an offline `.gguf` quantized LLM (like Mistral) to formulate an answer strictly employing the returned context and memory of the last three interactions.

## Explanation of How Endee is Used
Endee serves as the core vector search engine for the RAG pipeline. It allows extremely fast and efficient storage and retrieval of vector embeddings:
- **Initialization**: A local Endee client connects to the Endee server running gracefully at `http://localhost:8080/api/v1`.
- **Index Creation**: The code automatically initializes an `INT8` precision index `rag_index` with a dimension size of `384` using `cosine` distance metrics.
- **Upsertion**: During ingestion, vectors mapped with their textual metadata are batch-upserted cleanly into the Endee index.
- **Querying**: Endee performs high-speed Top-K querying against the user's encoded prompt vector to retrieve the most suitable context.

## Setup and Execution Instructions

### Prerequisites
- Python 3.9+
- Docker (to run the Endee server)
- A downloaded GGUF model (e.g., Mistral-7B) placed in the `models` directory and named `mistral.gguf`.
- Relevant PDF documents placed in the `data/docs` directory.

### Quickstart
1. **Prepare the Endee Server**:
   ```bash
   # Run the Endee official Docker image
   docker run -p 8080:8080 endee/server:latest
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install pypdf sentence_transformers ctransformers endee
   ```

3. **Run the Application**:
   ```bash
   python rag.py
   ```

4. **Usage**:
   - Type `ingest` to process your local PDFs.
   - Start asking questions to interact with the offline LLM!
   - Type `exit` or `quit` to stop the session.
