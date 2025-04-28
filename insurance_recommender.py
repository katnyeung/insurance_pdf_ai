from langchain_ollama import OllamaLLM
from langgraph.graph import Graph, START, END
from prompts import question_refinement_template, recommendation_template, profile_template
from langgraph.checkpoint.memory import MemorySaver
from vector_search_neo4j import document_retrieval, get_datasets
import re
from config import settings

# Initialize the LLM
llm = OllamaLLM(model="deepseek-r1:latest")

# Define all_categories globally
ALL_CATEGORIES = [
    "Company size/employee count", "Industry/business type", "Annual revenue",
    "Risk profile/concerns", "Budget constraints", "Country of the company",
    "Crypto coverage needs", "Preferred grace period (days) for new subsidiary cover"
]

# Define suggested answers for each category
SUGGESTED_ANSWERS = {
    "Company size/employee count": ["Small (1-50 employees)", "Medium (51-250 employees)", "Large (251-1000 employees)", "Enterprise (1001+ employees)"],
    "Industry/business type": ["Technology", "Finance", "Manufacturing", "Retail", "Healthcare", "Other"],
    "Annual revenue": ["< $1M", "$1M - $5M", "$5M - $25M", "$25M - $100M", "> $100M"],
    "Risk profile/concerns": ["Cyberattacks/Data Breaches", "Regulatory Investigations", "Employment Disputes", "Securities Claims", "Environmental Issues", "Other"],
    "Budget constraints": ["< $10K", "$10K - $50K", "$50K - $100K", "$100K - $500K", "> $500K"],
    "Country of the company": ["Hong Kong", "United States", "United Kingdom", "Canada", "Other Non-US", "Other US Territory"],
    "Crypto coverage needs": ["Yes", "No", "Maybe"],
    "Preferred grace period (days) for new subsidiary cover": ["30 days", "60 days", "90 days", "No preference"]
}

def get_initial_input(state):
    state["company_info"] = []
    state["collected_categories"] = []
    state["question_attempts"] = 0
    return state

def refine_question(state):
    if state.get("question_attempts", 0) >= 5:
        state["next_step"] = "COMPLETE"
        return state
    
    # Check if we have enough information to retrieve policies
    if len(state.get("company_info", [])) > 0:
        # Generate a temporary company profile for policy retrieval
        current_info_text = "\n".join(state.get("company_info", []))
        
        # Create a temporary state for policy retrieval
        temp_state = state.copy()
        temp_state["company_profile"] = current_info_text
        
        # Retrieve relevant policies based on current information
        temp_state = retrieve_relevant_policies(temp_state)
        
        # Get the number of retrieved chunks
        retrieved_chunks = temp_state.get("retrieved_chunks", [])
        num_chunks = len(retrieved_chunks)
        print(f"DEBUG - retrieved_chunks {num_chunks}")
        
        # Store the retrieved chunks in the main state for future reference
        state["retrieved_chunks"] = retrieved_chunks
        
        # If we have a sufficient number of relevant policies, move to recommendation
        if num_chunks >= 400:  # Updated threshold to 600 chunks
            state["next_step"] = "COMPLETE"
            return state
    
    # If we don't have enough policies yet, continue gathering information
    missing_info = [cat for cat in ALL_CATEGORIES if cat not in state.get("collected_categories", [])]
    if not missing_info:
        # Even if we have all categories, we still need enough chunks
        if state.get("retrieved_chunks", []) and len(state["retrieved_chunks"]) >= 600:
            state["next_step"] = "COMPLETE"
        else:
            # We have all categories but not enough chunks, continue with more specific questions
            state["next_step"] = "CONTINUE"
        return state
    
    current_info_text = "\n".join(state.get("company_info", [])) if state.get("company_info") else "No information yet."
    
    # Include information about retrieved policies if available
    policy_context = ""
    if "retrieved_chunks" in state and state["retrieved_chunks"]:
        num_chunks = len(state["retrieved_chunks"])
        policy_context = f"\nCurrently found {num_chunks} chunks. We need at least 600 chunk to make a recommendation."
    
    missing_with_suggestions = [f"Note: You have a maximum of 5 attempts to gather missing information. This is attempt {state['question_attempts'] + 1}."]
    for cat in missing_info:
        options = ", ".join(SUGGESTED_ANSWERS[cat])
        missing_with_suggestions.append(f"{cat} (e.g., {options})")
    
    question_chain = question_refinement_template | llm
    state["next_question"] = question_chain.invoke({
        "current_info": current_info_text,
        "missing_info": "\n".join(missing_with_suggestions),
        "policy_context": policy_context
    })
    state["question_attempts"] = state.get("question_attempts", 0) + 1
    return state

def process_user_input(state):
    if "user_input" in state and state["user_input"]:
        state["company_info"] = state.get("company_info", []) + [state["user_input"]]
        for category in ALL_CATEGORIES:
            if category not in state.get("collected_categories", []):
                state["collected_categories"] = state.get("collected_categories", []) + [category]
                break
        state["user_input"] = ""  # Clear input after processing
    return state

def generate_company_profile(state):
    if not state.get("company_info"):
        state["company_profile"] = "No company information available."
    else:
        collected_info = "\n".join(state["company_info"])
        profile_chain = profile_template | llm
        company_profile = profile_chain.invoke({"collected_info": collected_info})
        import re
        state["company_profile"] = re.sub(r'<think>.*?</think>', '', company_profile, flags=re.DOTALL).strip()
    return state

def retrieve_relevant_policies(state):
    """Retrieve relevant policy information from RAGFlow using the company profile"""
    print("\nDEBUG - Starting retrieve_relevant_policies function")
    
    # Log company info from state
    print(f"DEBUG - Company info in state: {state.get('company_info', [])}")
    
    company_profile = state.get("company_profile", "")
    
    if not company_profile or company_profile == "No company information available.":
        print("DEBUG - No company profile available")
        state["retrieved_chunks"] = []
        state["policy_context"] = "No specific policy information available."
        return state
    
    # Generate search query based on company profile
    search_query = f"{company_profile}"
    
    try:
        # Get all available datasets
        datasets = get_datasets()
        
        # Extract dataset IDs
        dataset_ids = []
        
        if datasets:
            for dataset in datasets:
                if isinstance(dataset, dict) and 'id' in dataset:
                    dataset_ids.append(dataset['id'])
                elif isinstance(dataset, str):
                    dataset_ids.append(dataset)
        
        print(f"DEBUG - Final dataset_ids: {dataset_ids}")
        
        # If no datasets are found, return no results
        if not dataset_ids:
            print("WARNING: No datasets found for retrieval.")
            state["retrieved_chunks"] = []
            state["policy_context"] = "No policy information found. Please check if datasets are available."
            return state
        
        # Retrieve relevant chunks from RAGFlow
        chunks = document_retrieval(dataset_ids, search_query)
        
        # Store retrieved chunks in state
        state["retrieved_chunks"] = chunks if chunks else []
        
        # Also build a formatted context string for the recommendation
        policy_texts = []
        if chunks:
            for chunk in chunks:
                content = chunk.get("content", "").strip()
                source = chunk.get("source", "Unknown Policy")
                if content:
                    policy_texts.append(f"Policy: {source}\nContent: {content}\n")
        
        if policy_texts:
            state["policy_context"] = "\n".join(policy_texts)
        else:
            state["policy_context"] = "No policy information found matching your company profile."
        
    except Exception as e:
        print(f"ERROR retrieving policy information: {e}")
        state["retrieved_chunks"] = []
        state["policy_context"] = "Error retrieving policy information."
    
    return state

def generate_recommendation(state):
    company_profile = state.get("company_profile", "\n".join(state.get("company_info", [])))
    
    # Use retrieved policy information if available, otherwise use generic text
    policy_info = state.get("policy_context", "No specific policy information available.")
    
    recommendation_chain = recommendation_template | llm
    recommendation = recommendation_chain.invoke({
        "company_profile": company_profile,
        "relevant_policies": policy_info
    })
    
    state["recommendation"] = recommendation
    return state

# Define and compile the LangGraph workflow
checkpointer = MemorySaver()
workflow = Graph()

workflow.add_node("get_initial_input", get_initial_input)
workflow.add_node("refine_question", refine_question)
workflow.add_node("process_user_input", process_user_input)
workflow.add_node("generate_company_profile", generate_company_profile)
workflow.add_node("retrieve_relevant_policies", retrieve_relevant_policies)
workflow.add_node("generate_recommendation", generate_recommendation)

workflow.add_edge(START, "get_initial_input")
workflow.add_edge("get_initial_input", "refine_question")
workflow.add_conditional_edges(
    "refine_question",
    lambda state: "generate_company_profile" if state.get("next_step") == "COMPLETE" else "process_user_input"
)
workflow.add_edge("process_user_input", "refine_question")
workflow.add_edge("generate_company_profile", "retrieve_relevant_policies")
workflow.add_edge("retrieve_relevant_policies", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)

# Compile the workflow
insurance_recommender = workflow.compile(checkpointer=checkpointer)