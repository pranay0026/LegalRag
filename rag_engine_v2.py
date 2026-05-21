import os
import chromadb
from google import genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# 1. Load Secure Credentials
load_dotenv()

# 2. Initialize the Official Gemini Client
# It automatically picks up GEMINI_API_KEY from your .env file
client = genai.Client()

# 3. Connect to the V2 Parent-Child Database
chroma_client = chromadb.PersistentClient(path="chroma_storage_v2")

# Re-initialize the identical local embedding function so string matching coordinates perfectly
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection_parent= chroma_client.get_collection(
    name="legal_reports_v2", 
    embedding_function=local_ef
)

def search_parent_child_vault(query_text:str)-> str:
    """
    Tool A: Queries the Parent-Child database.
    Retrieves small precise child fragments but returns rich macro parent text walls
    with inline document citations. Best for structural context and trend analysis.
    """
    try:
        results=collection_parent.query(
            query_texts=[query_text],
            n_results=4,
        )
        metadatas=results['metadatas'][0] if results['metadatas'] else []
        seen_parents=set()
        contexts=[]
        for meta in metadatas:
            p_id=meta.get('parent_id')
            p_text=meta.get('parent_text')
            source=meta.get('source', 'unknown source')
            if p_id not in seen_parents:
                seen_parents.add(p_id)
                contexts.append(f"From {source}:\n{p_text}\n")
        return "\n---\n".join(contexts) 
    except Exception as e:
        return f"Error during search: {str(e)}"