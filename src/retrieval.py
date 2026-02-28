import sys
import os

# Add the project root to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_chroma import Chroma

# --- Configuration ---
DB_PATH = r"D:\skill-dp\rag-agentic-ai\news-nexus\data\chroma_db"

def retrieve_documents(query, k=4, keyword_filter=True):
    """
    Retrieves documents using vector similarity and optionally applies
    a simple keyword boosting filter (Hybrid Search Logic).
    """
    
    # 1. Initialize the Embedding Model (Switched to Ollama for stability)
    from langchain_ollama import OllamaEmbeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. Connect to the existing Vector Store
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    
    # 3. Perform Vector Search (Semantic Retrieval)
    # k+2 fetches a bit more to allow for filtering
    print(f"Searching for: '{query}'...")
    results = vector_store.similarity_search_with_score(query, k=k+2)
    
    # 4. Hybrid Logic: Keyword Boosting (Simple Implementation)
    # If a document contains the exact query terms, we prioritize it.
    final_results = []
    
    if keyword_filter:
        query_terms = set(query.lower().split())
        
        for doc, score in results:
            content = doc.page_content.lower()
            term_matches = sum(1 for term in query_terms if term in content)
            
            # Artificial Score Boost: Lower score is better in Chroma (Distance)
            # We subtract a small value for every match to make it "closer"
            boosted_score = score - (term_matches * 0.05)
            
            final_results.append((doc, boosted_score))
            
        # Re-sort based on new boosted scores
        final_results.sort(key=lambda x: x[1])
        
        # Trim back to requested 'k'
        final_results = final_results[:k]
    else:
        final_results = results[:k]

    return final_results

# --- Test Block ---
if __name__ == "__main__":
    # Test query
    test_query = "What is the impact of GenAI on productivity?"
    
    retrieved_docs = retrieve_documents(test_query)
    
    print(f"\n--- Top {len(retrieved_docs)} Results ---")
    for i, (doc, score) in enumerate(retrieved_docs):
        print(f"\n[Result {i+1}] (Score: {score:.4f})")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content Snippet: {doc.page_content[:200]}...")