import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup Client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Setup ChromaDB Connection
chroma_client = chromadb.PersistentClient(path="chroma_storage") # Ensure this matches your path
collection_semantic = chroma_client.get_collection(name="legal_reports")

def search_semantic_vault(query_text: str) -> str:
    """
    Tool B: Queries the original Semantic Chunking database.
    Retrieves highly dense, isolated sentence fragments directly.
    Best for finding a specific data point, individual number, or footnote metric.
    """
    try:
        results = collection_semantic.query(
            query_texts=[query_text],
            n_results=5
        )
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        if not documents:
            return "Observation: No matching records found in the Semantic vault."
            
        contexts = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get("source", "Unknown Source") if meta else "Unknown Source"
            contexts.append(f"[{source}]: {doc}")
            
        return "\n\n---\n\n".join(contexts)
        
    except Exception as e:
        return f"Observation Error (Semantic Tool): {str(e)}"