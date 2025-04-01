from langchain_ollama import OllamaLLM
from langgraph.graph import Graph, START, END
from prompts import question_refinement_template, recommendation_template, profile_template
from langgraph.checkpoint.memory import MemorySaver

# Initialize the LLM
llm = OllamaLLM(model="deepseek-r1:1.5b")

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
    
    missing_info = [cat for cat in ALL_CATEGORIES if cat not in state.get("collected_categories", [])]
    if not missing_info:
        state["next_step"] = "COMPLETE"
        return state
    
    current_info_text = "\n".join(state.get("company_info", [])) if state.get("company_info") else "No information yet."
    missing_with_suggestions = [f"Note: You have a maximum of 5 attempts to gather missing information. This is attempt {state['question_attempts'] + 1}."]
    for cat in missing_info:
        options = ", ".join(SUGGESTED_ANSWERS[cat])
        missing_with_suggestions.append(f"{cat} (e.g., {options})")
    
    question_chain = question_refinement_template | llm
    state["next_question"] = question_chain.invoke({
        "current_info": current_info_text,
        "missing_info": "\n".join(missing_with_suggestions)
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

def generate_recommendation(state):
    company_profile = state.get("company_profile", "\n".join(state.get("company_info", [])))
    recommendation_chain = recommendation_template | llm
    recommendation = recommendation_chain.invoke({
        "company_profile": company_profile,
        "relevant_policies": "Mock insurance policy data for demonstration purposes."
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
workflow.add_node("generate_recommendation", generate_recommendation)

workflow.add_edge(START, "get_initial_input")
workflow.add_edge("get_initial_input", "refine_question")
workflow.add_conditional_edges(
    "refine_question",
    lambda state: "generate_company_profile" if state.get("next_step") == "COMPLETE" else "process_user_input"
)
workflow.add_edge("process_user_input", "refine_question")
workflow.add_edge("generate_company_profile", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)

insurance_recommender = workflow.compile(checkpointer=checkpointer)