import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuration
MEMORY_DB_PATH = r"D:\skill-dp\rag-agentic-ai\news-nexus\data\archive_memory"
COLLECTION_NAME = "newsletter_archive"

class MemoryStore:
    def __init__(self):
        # Initialize Ollama Embeddings for stability (nomic-embed-text)
        from langchain_ollama import OllamaEmbeddings
        self.embedding_fn = OllamaEmbeddings(model="nomic-embed-text")
        
        # Connection to Archive Database
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            persist_directory=MEMORY_DB_PATH
        )

    def save_memory(self, topic: str, content: str):
        """Saves a finished newsletter to the vector store."""
        print(f"\n[Memory] Archiving newsletter on '{topic}'...")
        
        doc = Document(
            page_content=content,
            metadata={"topic": topic, "timestamp": str(datetime.now())}
        )
        
        self.vector_store.add_documents([doc])
        print("[Memory] Successfully saved.")

    def check_memory(self, query: str, k=1) -> str:
        """
        Searches past newsletters to see if we've covered this recently.
        Returns a summary string to inject into the Agent's context.
        """
        # Similarity search returns documents and scores
        # Note: LangChain Chroma's similarity_search handles embedding automatically
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        if results:
            doc, score = results[0]
            # In Chroma (Distance), lower score is more similar. 
            # 0.4 is a reasonable threshold for "same topic"
            if score < 0.4:
                past_content = doc.page_content
                metadata = doc.metadata
                date = metadata.get("timestamp", "unknown date")
                return f"WARNING: We already wrote a newsletter on this topic on {date}. \nSummary of past content: {past_content[:300]}..."
        
        return "No prior newsletters found on this topic. You are clear to proceed."

# Test block
if __name__ == "__main__":
    mem = MemoryStore()
    mem.save_memory("Test Topic", "This is a test newsletter content.")
    print(mem.check_memory("Test Topic"))