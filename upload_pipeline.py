import os
import re
import uuid
import gc
import chromadb
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Load environment configs
load_dotenv()
Groq_API_KEY = os.getenv("GROQ_API_KEY")
orchestrator_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
AGENT_MODEL = "llama-3.1-8b-instant"

# Initialize local embedding alignment matching your permanent vaults
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def process_and_analyze_upload(uploaded_file, user_query: str, log_list: list = None) -> str:
    """
    Pipeline that intercepts a Streamlit file upload, builds in-memory 
    Semantic and Parent-Child vector indexes on the fly, and runs the ReAct agent loop.
    Optimized for strict RAM management and batch vector calculations.
    """
    # 0. Setup the internal dual-logging utility
    def log_and_print(message_text: str):
        print(message_text)  # Keep logging to terminal console
        if log_list is not None:
            log_list.append(message_text)  # Send directly to your Streamlit expander

    log_and_print(f"Pipeline: Extracting text from uploaded file: {uploaded_file.name}")
    
    # 1. Extract Raw Text using PyPDF
    try:
        reader = PdfReader(uploaded_file)
        full_text_segments = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text_segments.append((i + 1, text))
    except Exception as e:
        return f"Pipeline Ingestion Error: Failed to parse PDF structure. {str(e)}"
    
    if not full_text_segments:
        return "Pipeline Ingestion Error: The uploaded file appears to be empty or unscannable."

    # 2. Spin up an Ephemeral (In-Memory Only) Chroma Client
    ephemeral_client = chromadb.EphemeralClient()
    
    temp_semantic_col = ephemeral_client.create_collection(
        name="temp_semantic", embedding_function=local_ef
    )
    temp_parent_child_col = ephemeral_client.create_collection(
        name="temp_parent_child", embedding_function=local_ef
    )

    # Allocating low-overhead batch arrays to minimize transactional memory spikes
    semantic_batch = {"documents": [], "metadatas": [], "ids": []}
    child_batch = {"documents": [], "metadatas": [], "ids": []}
    
    # Isolate large parent texts in a native dictionary to protect vector engine memory space
    parent_macro_vault = {}

    try:
        # 3. Perform Live Dual-Chunking Strategies
        log_and_print("Pipeline: Running Dual-Chunking strategies on memory stream...")
        filename = uploaded_file.name
        
        for page_num, page_text in full_text_segments:
            # Strategy A: Pure Semantic/Sentence Splitting
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', page_text) if s.strip()]
            for idx, sentence in enumerate(sentences):
                if len(sentence) > 10:
                    semantic_batch["documents"].append(sentence)
                    semantic_batch["metadatas"].append({"source": filename, "page": page_num})
                    semantic_batch["ids"].append(f"sem_{page_num}_{idx}_{uuid.uuid4().hex[:6]}")

            # Strategy B: Parent-Child Splitting
            parent_text = page_text.strip()
            parent_id = f"parent_{page_num}_{uuid.uuid4().hex[:6]}"
            
            # Map macro string text tracking to native memory address
            parent_macro_vault[parent_id] = parent_text
            
            child_size = 150
            overlap = 30
            for i in range(0, len(parent_text), child_size - overlap):
                child_chunk = parent_text[i:i + child_size]
                if len(child_chunk.strip()) > 15:
                    child_batch["documents"].append(child_chunk)
                    child_batch["metadatas"].append({
                        "parent_id": parent_id,
                        "source": filename,
                        "page": page_num
                    })
                    child_batch["ids"].append(f"child_{uuid.uuid4().hex[:6]}")

        # Execute single batch execution to insert items efficiently into system memory
        if semantic_batch["documents"]:
            temp_semantic_col.add(
                documents=semantic_batch["documents"],
                metadatas=semantic_batch["metadatas"],
                ids=semantic_batch["ids"]
            )
        if child_batch["documents"]:
            temp_parent_child_col.add(
                documents=child_batch["documents"],
                metadatas=child_batch["metadatas"],
                ids=child_batch["ids"]
            )

        log_and_print(f"In-Memory Indexes Ready. Executing Targeted Agent Reasoning...")

        # 4. Define Dynamic Real-Time Tools accessing the Ephemeral Collections
        def temp_search_semantic(query: str) -> str:
            res = temp_semantic_col.query(query_texts=[query], n_results=3)
            docs = res['documents'][0] if res['documents'] else []
            metas = res['metadatas'][0] if res['metadatas'] else []
            return "\n\n---\n\n".join([f"[{m.get('source')} Page {m.get('page')}]: {d}" for d, m in zip(docs, metas)]) if docs else "No semantic records found."

        def temp_search_parent_child(query: str) -> str:
            res = temp_parent_child_col.query(query_texts=[query], n_results=2)
            metas = res['metadatas'][0] if res['metadatas'] else []
            if not metas: 
                return "No macro context records found."
            seen = set()
            contexts = []
            for m in metas:
                pid = m.get("parent_id")
                if pid not in seen:
                    seen.add(pid)
                    # Resolve full block content from local dictionary structure
                    resolved_macro_text = parent_macro_vault.get(pid, "Macro context missing.")
                    contexts.append(f"[{m.get('source')} Page {m.get('page')}]: {resolved_macro_text}")
            return "\n\n---\n\n".join(contexts)

        # 5. The Dynamic ReAct State Machine Execution Bounded Loop
        system_prompt = f"""
        You are an advanced legal agent investigating an uploaded document: {filename}.
        You resolve user requests using an iterative loop of Thought, Action, and Observation.

        AVAILABLE TOOLS:
        1. search_semantic_vault[query]: Queries individual text slices. Best for small numbers or specific sentences.
        2. search_parent_child_vault[query]: Queries full page/paragraph blocks. Best for structural context and trend summaries.

        STRICT TURN FORMAT:
        Thought: Your analysis about what data is missing from your context.
        Action: Call a tool using exact brackets, like: search_semantic_vault[specific keyword]
        Observation: Extracted text results will appear here.
        ... (Max 3 turns)
        Final Answer: Your final synthesis backed explicitly by page numbers and citations.

        RULES: Rely ONLY on observations. If the information is not present, state that it is unavailable.
        """

        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this user query against the uploaded document: {user_query}"}
        ]

        agent_final_output = "Agent timeout reached while investigating uploaded document."

        for turn in range(3):
            print(f"[Upload Loop Turn {turn + 1}/3] Processing...")
            try:
                response = orchestrator_client.chat.completions.create(
                    model=AGENT_MODEL, messages=conversation_history, temperature=0.0
                )
                assistant_output = response.choices[0].message.content.strip()
                print(f"\n[INTERNAL MONOLOGUE]:\n{assistant_output}\n")
                conversation_history.append({"role": "assistant", "content": assistant_output})

                if "Final Answer:" in assistant_output:
                    agent_final_output = assistant_output.split("Final Answer:")[-1].strip()
                    break

                tool_match = re.search(r"(search_semantic_vault|search_parent_child_vault)\[(.*?)\]", assistant_output)
                if tool_match:
                    tool_name = tool_match.group(1)
                    search_query = tool_match.group(2).strip()
                    
                    if tool_name == "search_semantic_vault":
                        obs = temp_search_semantic(search_query)
                    else:
                        obs = temp_search_parent_child(search_query)
                    
                    print(f"[Observation Grabbed]: {len(obs)} characters.")
                    conversation_history.append({"role": "user", "content": f"Observation: {obs}"})
                else:
                    if "Final Answer:" in assistant_output:
                        agent_final_output = assistant_output.split("Final Answer:")[-1].strip()
                    else:
                        agent_final_output = f"Structure Break: {assistant_output}"
                    break
            except Exception as e:
                agent_final_output = f"Error running real-time pipeline: {str(e)}"
                break

        return agent_final_output

    finally:
        # Crucial clean up layer to safely deallocate internal vector records and dictionaries from RAM
        log_and_print("Pipeline: Evicting ephemeral vector instances and cleaning heap space...")
        try:
            ephemeral_client.delete_collection("temp_semantic")
            ephemeral_client.delete_collection("temp_parent_child")
        except Exception:
            pass
        
        # Free structure arrays explicitly
        del semantic_batch
        del child_batch
        del parent_macro_vault
        
        # Force garbage collector to immediately release unreferenced memory frames
        gc.collect()
# Wrap your testing code at the bottom of agent_orchestrator.py like this:
if __name__ == "__main__":
    test_logs = []
    # This will now only execute if you run python agent_orchestrator.py directly
    print(legal_agent_loop("What is Article 21?", log_list=test_logs))