import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Configuration ---
DATA_PATH = r"D:\skill-dp\rag-agentic-ai\news-nexus\data\raw_pdfs"
DB_PATH = r"D:\skill-dp\rag-agentic-ai\news-nexus\data\chroma_db"

def ingest_documents():
    # 1. Load Documents (Day 3: Data Loading)
    print(f"Loading PDFs from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # 2. Split Text (Day 3: Chunking Strategies)
    # We use overlapping chunks to maintain context across boundaries [cite: 45]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Balance between context and precision
        chunk_overlap=50,    # Overlap to prevent data loss at edges
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Initialize Embeddings (Switched to Ollama for stability)
    from langchain_ollama import OllamaEmbeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Create Vector Store with Batching (Day 2: Vector Databases)
    print("Initializing Vector Store (this may take a few minutes for large PDFs)...")
    
    # Initialize an empty vector store first
    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )

    # Batch size (Smaller batches are safer for Ollama)
    BATCH_SIZE = 100
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"   > Processing batch {i // BATCH_SIZE + 1} of {(total_chunks-1) // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        vector_db.add_documents(batch)
    
    print("Vector Store created successfully.")
    return len(raw_documents), len(chunks)

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Create a dummy PDF if none exists (for testing)
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"No PDFs found in {DATA_PATH}. Please add files to enable RAG features.")
    else:
        ingest_documents()