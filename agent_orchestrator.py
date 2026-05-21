import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from rag_engine import search_semantic_vault
from rag_engine_v2 import search_parent_child_vault

load_dotenv()
Groq_API_KEY=os.getenv("GROQ_API_KEY3")
orchestrator_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
AGENT_MODEL="llama-3.1-8b-instant"  

AGENT_SYSTEM_PROMPT="""
You are an advanced Agentic Legal Scholar specializing in the Constitution of India. You resolve complex constitutional, structural, and legal query paths by using an iterative reasoning loop of Thought, Action, and Observation.

AVAILABLE TOOLS:
1. search_semantic_vault[query]: Queries precise article slices. Best for locating exact article numbers, specific clauses, unique legal terminology, or pinpoint sub-clauses.
2. search_parent_child_vault[query]: Queries deep macro legal blocks with surrounding structural context. Best for reviewing the complete text of an entire Part, Chapter, or multi-article legal framework.

YOUR STRICT TURN EXECUTION FORMAT:
You must output text strictly following this progression. Do not mix patterns or omit keys:

Thought: Your analytical reasoning about what constitutional context is missing, or which legal search tool is best suited next.
Action: Call one of the tools above using exact bracket notation. Example: search_semantic_vault[Article 21 Protection of Life and Personal Liberty]

(After you output an Action block, the system will pause execution, run your local tool, and return the data directly inside an Observation block)

Observation: The raw context returned by the tool will be injected here.

... (You can repeat the Thought/Action/Observation cycle up to 3 times maximum)

Once you have gathered all necessary constitutional context or verified the text is unavailable, output your conclusion in this exact format:

Final Answer: Your ultimate concise legal synthesis, explicitly citing the specific Article, Clause, Chapter, and Part of the Constitution found in your observations.

CRITICAL LEGAL AGENT RULES:
1. Do not invent provisions, guess interpretations, or extrapolate laws. Only use legal text explicitly returned in an Observation block.
2. If both tools yield insufficient or empty data after searching, state exactly what constitutional provision is missing in your Final Answer.
3. Always explicitly name the source file or document section present in the observations in your final legal synthesis.
4. Never assume historical intent. If the context does not explicitly specify a legal exception or provision, treat it as unavailable.
5. ZERO OUTSIDE KNOWLEDGE: You must operate as if you have total amnesia regarding real-world legal frameworks, landmark case laws, or outside political facts. You only know what is explicitly stated within a returned Observation block.
6. ABSOLUTE REFUSAL: If an Observation block returns "No matching records found" or is empty, you are strictly FORBIDDEN from using your pre-trained training data to answer. You must immediately exit the loop and output: "Final Answer: I'm sorry, my local vault does not contain specific data to answer this."
"""
def legal_agent_loop(user_query: str, log_list: list = None) -> str:
    """
    Main Autonomous ReAct Controller.
    Accepts an optional log_list to send terminal prints back to the Streamlit UI.
    """
    # Create a small internal helper function to handle dual-logging
    def log_and_print(message_text: str):
        print(message_text)  # Keep printing to your terminal
        if log_list is not None:
            log_list.append(message_text)  # Send directly to Streamlit chat UI

    # Initialize the core conversation history ledger
    conversation_history = [
        {
            "role": "system", 
            "content": AGENT_SYSTEM_PROMPT
        },
        {
            "role": "user", 
            "content": f"Execute audit analysis for this query: {user_query}"
        }
    ]
    
    max_turns = 10
    
    log_and_print(f"Initializing Autonomous Analyst for Query: '{user_query}'")
    log_and_print("-" * 60)

    for turn in range(max_turns):
        log_and_print(f"[Turn {turn + 1}/{max_turns}] Generating Agent Reasoning Matrix...")
        
        try:
            # Call Groq's API
            response = orchestrator_client.chat.completions.create(
                model=AGENT_MODEL,
                messages=conversation_history,
                temperature=0.0
            )
            
            assistant_output = response.choices[0].message.content.strip()
            log_and_print(f"\n[INTERNAL MONOLOGUE]:\n{assistant_output}\n")
            
            # Log the agent's reasoning turn into the memory ledger
            conversation_history.append({"role": "assistant", "content": assistant_output})
            
            if "Final Answer:" in assistant_output:
                final_synthesis = assistant_output.split("Final Answer:")[-1].strip()
                return final_synthesis
            
            # Search for a Tool call execution token using Regex
            tool_match = re.search(r"(search_semantic_vault|search_parent_child_vault)\[(.*?)\]", assistant_output)
            
            if tool_match:
                tool_name = tool_match.group(1)
                search_query = tool_match.group(2).strip()
                
                log_and_print(f"[System Intercept]: Executing {tool_name} with query '{search_query}'...")
                
                if tool_name == "search_semantic_vault":
                    observation_result = search_semantic_vault(search_query)
                elif tool_name == "search_parent_child_vault":
                    observation_result = search_parent_child_vault(search_query)
                else:
                    observation_result = "Observation Error: Invoked tool does not match the registry."
                
                log_and_print(f"[Observation Received]: {len(observation_result)} characters extracted.")
                log_and_print("-" * 60)
                
                conversation_history.append({
                    "role": "user", 
                    "content": f"Observation: {observation_result}"
                })
                
            else:
                log_and_print("[System Warning]: Agent drifted from formatting structure.")
                conversation_history.append({
                    "role": "user",
                    "content": "System Alert: Please conclude with a 'Final Answer:' now."
                })
                
            time.sleep(1)
            
        except Exception as e:
            return f"Orchestrator Internal Error during turn iteration: {str(e)}"
    log_and_print("[System Alert]: Turn budget exhausted. Forcing final legal synthesis from collected observations...")
    try:
        # Append an explicit command forcing the model to stop searching and summarize
        conversation_history.append({
            "role": "user", 
            "content": "System Override: You have run out of search turns. Do not call any more tools. Read your previous observations immediately and output your absolute conclusion starting with 'Final Answer:' now."
        })
        
        final_run = orchestrator_client.chat.completions.create(
            model=AGENT_MODEL,
            messages=conversation_history,
            temperature=0.0
        )
        final_output = final_run.choices[0].message.content.strip()
        
        if "Final Answer:" in final_output:
            return final_output.split("Final Answer:")[-1].strip()
        return final_output
        
    except Exception as fallback_error:
        return f"Agent timeout reached, and fallback synthesis failed: {str(fallback_error)}"
            
    return "Agent timeout reached: The system exhausted its execution budget without finding a definitive answer."
# Wrap your testing code at the bottom of agent_orchestrator.py like this:
if __name__ == "__main__":
    test_logs = []
    # This will now only execute if you run python agent_orchestrator.py directly
    print(legal_agent_loop("What is Article 21?", log_list=test_logs))