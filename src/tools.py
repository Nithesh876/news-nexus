import json
import os
from langchain.tools import tool
from langchain_ollama import ChatOllama
from retrieval import retrieve_documents

# --- Tool 1: The RAG Tool (Enhanced with Deep Links) ---
@tool
def lookup_policy_docs(query: str) -> str:
    """
    Useful for finding specific details, statistics, or sections from the uploaded 
    industry reports (PDFs). Use this when you need factual grounding.
    """
    # Clean the query if it comes in as a dictionary string
    if isinstance(query, str) and "{" in query:
        query = query.replace("{", "").replace("}", "").replace("value:", "")
        
    docs = retrieve_documents(query, k=3)
    if not docs:
        return f"RAG: No relevant internal documents found for query: '{query}'."
        
    results = []
    for doc, score in docs:
        source_name = doc.metadata.get('source', 'Unknown PDF')
        basename = os.path.basename(source_name)
        safe_source_path = source_name.replace('\\', '/')
        results.append(f"Content: {doc.page_content}\nSource Link: [{basename}](file:///{safe_source_path})")
    
    return "\n\n".join(results)

# --- Tool 2: The Live Web Search Tool (Enhanced with Citations) ---
@tool
def web_search_stub(query: str) -> str:
    """
    Useful for finding the 'latest' or 'current' news from the internet.
    Use this for recent trends, real-time events, or any information not in the PDFs.
    """
    from duckduckgo_search import DDGS
    import re
    
    clean_query = str(query)
    if "{" in clean_query:
        match = re.search(r'["\']query["\']:\s*["\']([^"\']+)["\']', clean_query)
        if match:
            clean_query = match.group(1)
        else:
            clean_query = clean_query.replace("{", "").replace("}", "").replace("value:", "").strip()

    print(f"\n[Tool Called] Live Web Search for: '{clean_query}'")
    
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Try text search first
            results = list(ddgs.text(clean_query, max_results=10))
            
            # If empty, try a news search as fallback
            if not results:
                print("   > No text results, trying news search...")
                results = list(ddgs.news(clean_query, max_results=10))
                
            if results:
                formatted_results = []
                for res in results:
                    title = res.get("title", "Source")
                    link = res.get("href", res.get("url", "#"))
                    snippet = res.get("body", res.get("snippet", "No Snippet"))
                    formatted_results.append(f"Title: {title}\nLink: [{title}]({link})\nSnippet: {snippet}")
                return "\n\n---\n".join(formatted_results)
            else:
                return f"WEB: DuckDuckGo returned 0 results for query: '{clean_query}'."
    except Exception as e:
        return f"WEB: Error during search: {e}"
        
    return "WEB: Unexpected failure in web search tool."

# --- Tool 3: RSS Feed Connector (Enhanced) ---
@tool
def rss_feed_search(query: str) -> str:
    """
    Useful for finding high-quality, targeted news from specific industry RSS feeds.
    """
    import feedparser
    
    FEEDS = [
        "https://www.technologyreview.com/feed/",
        "https://openai.com/news/rss.xml",
        "https://machinelearning.apple.com/rss.xml",
        "https://feeds.feedburner.com/TheHackersNews",
        "https://techcrunch.com/feed/"
    ]
    
    results = []
    # Broaden keyword search
    keywords = query.lower().split()
    print(f"\n[Tool Called] RSS Search for keywords: {keywords}")
    
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                text_to_search = (entry.title + " " + entry.get("summary", "")).lower()
                # Match if ANY keyword exists (Hybrid match)
                if any(kw in text_to_search for kw in keywords):
                    results.append(f"Source: {feed.feed.get('title', 'Industry News')}\nTitle: {entry.title}\nLink: [{entry.title}]({entry.link})\nSummary: {entry.get('summary', '')[:250]}...")
        except Exception as e:
            continue
            
    return "\n\n---\n".join(results) if results else "No matching recent RSS entries found."

def get_llm_with_tools():
    llm = ChatOllama(model="llama3.2", temperature=0) 
    tools = [lookup_policy_docs, web_search_stub, rss_feed_search]
    llm_with_tools = llm.bind_tools(tools)
    return llm, llm_with_tools, tools

if __name__ == "__main__":
    # Test Block
    base_llm, agent, tools_list = get_llm_with_tools()
    query = "Latest AI models by Apple or OpenAI"
    print(f"Testing tools... {query}")
    print(agent.invoke(query).tool_calls)
