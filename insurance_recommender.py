from langchain_ollama import OllamaLLM
from langgraph.graph import Graph, START, END
from prompts import question_refinement_template, recommendation_template, profile_template
from vector_search import document_retrieval, get_datasets
from langgraph.checkpoint.memory import MemorySaver

# Initialize the LLM
llm = OllamaLLM(model="deepseek-r1:1.5b")

def get_initial_input(state):
    print("Please provide a brief description of your company: Company size, industry, annual revenue, risk profile, and budget constraints.")
    user_input = input("> ")
    state["company_info"] = [user_input]
    state["collected_categories"] = []
    return state

def refine_question(state):
    # Initialize state if needed
    if "company_info" not in state:
        state["company_info"] = []
    if "collected_categories" not in state:
        state["collected_categories"] = []
    
    # Rest of the function remains the same
    all_categories = [
        "Company size/employee count",
        "Industry/business type",
        "Annual revenue",
        "Risk profile/concerns",
        "Budget constraints"
    ]
    
    missing_info = [cat for cat in all_categories if cat not in state["collected_categories"]]
    
    if not missing_info:
        state["next_step"] = "COMPLETE"
        return state
    
    current_info_text = "\n".join(state["company_info"]) if state["company_info"] else "No information yet."
    
    question_chain = question_refinement_template | llm
    response = question_chain.invoke({
        "current_info": current_info_text,
        "missing_info": "\n".join(missing_info)
    })
    
    state["next_question"] = response
    return state

def ask_user(state):
    # Display the question and get user input
    question = state["next_question"]
    print(f"\nQuestion: {question}")
    user_input = input("> ")
    
    # Store the user input
    state["user_input"] = user_input
    
    # For debugging
    print(f"DEBUG - Received input: {user_input}")
    
    return state

def process_user_input(state):
    if "user_input" in state and state["user_input"]:
        # Add the user input to company_info
        if "company_info" not in state:
            state["company_info"] = []
        user_response = state["user_input"]
        state["company_info"].append(user_response)

        # Mark the first category as collected (for simplicity)
        if "collected_categories" not in state:
            state["collected_categories"] = []
        all_categories = [
            "Company size/employee count",
            "Industry/business type",
            "Annual revenue",
            "Risk profile/concerns",
            "Budget constraints"
        ]
        # Find the first category that hasn't been collected yet
        for category in all_categories:
            if category not in state["collected_categories"]:
                state["collected_categories"].append(category)
                print(f"DEBUG - Marked category as collected: {category}")
                break

        # Clear the user input
        state["user_input"] = ""
    return state

def generate_company_profile(state):
    """Use LLM to generate a structured company profile from collected information"""
    if "company_info" not in state or not state["company_info"]:
        state["company_profile"] = "No company information available."
        return state
    
    # Format the collected information for the LLM
    collected_info = "\n".join(state["company_info"])
    
    # Generate the company profile
    profile_chain = profile_template | llm
    company_profile = profile_chain.invoke({"collected_info": collected_info})
    
    # Remove <think> </think> tags if present
    import re
    company_profile = re.sub(r'<think>.*?</think>', '', company_profile, flags=re.DOTALL)
    company_profile = company_profile.strip()
    
    # Store the cleaned profile in the state
    state["company_profile"] = company_profile
    print(f"DEBUG - Generated company profile: {company_profile}")
    
    return state

def generate_recommendation(state):
    # Use the generated company profile instead of joining company_info
    company_profile = state.get("company_profile", "\n".join(state["company_info"]))
    print(f"DEBUG - Using company profile: {company_profile}")
    
    try:
        # Get dataset ID
        datasets = get_datasets(name="insurance_kb")
        if datasets and len(datasets) > 0:
            dataset_id = datasets[0].get("id")
            print(f"DEBUG - Dataset ID: {dataset_id}")
            
            # Use document_retrieval with the company profile
            chunks = document_retrieval([dataset_id], company_profile)
            
            if chunks:
                # Include document metadata with each chunk
                formatted_chunks = []
                for chunk in chunks:
                    doc_name = chunk.get("document_keyword", "Unknown Document")
                    doc_id = chunk.get("document_id", "Unknown ID")
                    content = chunk.get("content", "")
                    formatted_chunk = f"DOCUMENT: {doc_name} (ID: {doc_id})\nCONTENT: {content}"
                    formatted_chunks.append(formatted_chunk)
                
                relevant_policies = "\n\n---\n\n".join(formatted_chunks)
                print(f"DEBUG - Retrieved {len(chunks)} relevant policy chunks")
                
                # Use the recommendation template
                recommendation_chain = recommendation_template | llm
                recommendation = recommendation_chain.invoke({
                    "company_profile": company_profile,
                    "relevant_policies": relevant_policies
                })
                
                state["recommendation"] = recommendation
                print("DEBUG - Generated recommendation using retrieved policy information")
                return state
    
    except Exception as e:
        print(f"DEBUG - Error retrieving documents: {e}")
    
    print("DEBUG - Falling back to local recommendation generation")
    
    # Fallback to local recommendation generation
    policies_text = "Mock insurance policy data for demonstration purposes."
    recommendation_chain = recommendation_template | llm
    recommendation = recommendation_chain.invoke({
        "company_profile": company_profile,
        "relevant_policies": policies_text
    })
    state["recommendation"] = recommendation
    
    print("\nGenerated recommendation based on:")
    print(f"- Company profile: {company_profile}")
    return state


# Now modify the LangGraph flow to include the new function
checkpointer = MemorySaver()
workflow = Graph()

# Add nodes
workflow.add_node("get_initial_input", get_initial_input)
workflow.add_node("refine_question", refine_question)
workflow.add_node("ask_user", ask_user)
workflow.add_node("process_user_input", process_user_input)
workflow.add_node("generate_company_profile", generate_company_profile)
workflow.add_node("generate_recommendation", generate_recommendation)

# Define the flow
workflow.add_edge(START, "get_initial_input")
workflow.add_edge("get_initial_input", "refine_question")
workflow.add_conditional_edges(
    "refine_question",
    lambda state: "generate_company_profile" if state.get("next_step") == "COMPLETE" else "ask_user"
)
workflow.add_edge("ask_user", "process_user_input")
workflow.add_edge("process_user_input", "refine_question")
workflow.add_edge("generate_company_profile", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)

# Compile the graph with the checkpointer
insurance_recommender = workflow.compile(checkpointer=checkpointer)