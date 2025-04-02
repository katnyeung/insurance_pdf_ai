from flask import Flask, request, jsonify, render_template, session
import uuid
import os
from insurance_recommender import insurance_recommender, ALL_CATEGORIES, SUGGESTED_ANSWERS
from prompts import profile_template
from langchain_ollama import OllamaLLM
from vector_search import document_retrieval, get_datasets
from config import settings

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize the LLM
llm = OllamaLLM(model=settings.LLM_MODEL)

@app.route('/')
def index():
    """Render the main application page"""
    # Initialize a new session
    session['thread_id'] = str(uuid.uuid4())
    session['company_info'] = []
    session['collected_categories'] = []
    return render_template('index.html')

@app.route('/update_requirements', methods=['POST'])
def update_requirements():
    """Process user input and update requirements"""
    data = request.json
    user_input = data.get('input', '')
    current_info = data.get('current_info', [])
    
    if not user_input:
        return jsonify({"error": "No input provided"})
    
    # Add the new input to the current info
    updated_info = current_info + [user_input]
    
    # Create a state object to use with LangGraph
    state = {
        "company_info": updated_info,
        "collected_categories": [],
        "question_attempts": len(updated_info)
    }
    
    # We can't directly access the node functions in the compiled workflow
    # Let's use the refine_question function directly instead
    from insurance_recommender import refine_question
    state = refine_question(state)
    
    # Check if we have a next question or if we're complete
    if state.get("next_step") == "COMPLETE":
        next_question = "Great! You've provided all the necessary information. You can now generate a recommendation."
        completed = True
    else:
        next_question = state.get("next_question", "Please provide more information about your company.")
        completed = False
    
    # Generate a profile summary using the profile template
    profile_chain = profile_template | llm
    profile = profile_chain.invoke({"collected_info": "\n".join(updated_info)})
    
    # Remove any thinking tags
    import re
    profile = re.sub(r'<think>.*?</think>', '', profile, flags=re.DOTALL).strip()
    
    # Determine which categories have been detected in the input
    # This is a simplified version - the actual function would be more sophisticated
    all_detected = []
    for category in ALL_CATEGORIES:
        category_lower = category.lower()
        input_text = " ".join(updated_info).lower()
        
        # Check if category keywords are in the input
        for keyword in category.lower().split('/'):
            if keyword.strip() in input_text:
                all_detected.append(category)
                break
        
        # Check if any suggested answers for this category are in the input
        if category not in all_detected:
            for answer in SUGGESTED_ANSWERS[category]:
                if answer.lower() in input_text:
                    all_detected.append(category)
                    break
    
    # Get remaining missing categories
    missing_categories = [cat for cat in ALL_CATEGORIES if cat not in all_detected]
    
    return jsonify({
        "updated_info": updated_info,
        "all_detected_categories": all_detected,
        "next_question": next_question,
        "profile": profile,
        "missing_categories": missing_categories,
        "completed": completed
    })

@app.route('/generate_recommendation', methods=['POST'])
def generate_recommendation():
    """Generate insurance recommendations based on company information"""
    print("\nDEBUG - Starting generate_recommendation endpoint")
    
    data = request.json
    company_info = data.get('company_info', [])
    thread_id = session.get('thread_id', str(uuid.uuid4()))
    
    print(f"DEBUG - Company info received: {company_info}")
    print(f"DEBUG - Thread ID: {thread_id}")
    
    if not company_info:
        print("DEBUG - No company information provided")
        return jsonify({"error": "No company information provided."})
    
    # Start with initial state
    state = {
        "company_info": company_info,
        "collected_categories": [],
        "question_attempts": 5,  # Set to max attempts to skip questioning
        "next_step": "COMPLETE"  # Skip to profile generation
    }
    
    try:
        print("DEBUG - Importing functions from insurance_recommender")
        # Import the functions directly
        from insurance_recommender import generate_company_profile, retrieve_relevant_policies, generate_recommendation
        
        # Generate company profile
        print("DEBUG - Calling generate_company_profile")
        profile_state = generate_company_profile(state)
        print(f"DEBUG - Company profile generated: {profile_state.get('company_profile', '')[:100]}...")
        
        # Retrieve relevant policies
        print("DEBUG - Calling retrieve_relevant_policies")
        retrieval_state = retrieve_relevant_policies(profile_state)
        
        # Check if we got any policy information
        policy_context = retrieval_state.get("policy_context", "")
        if policy_context and not policy_context.startswith("No specific policy information"):
            print(f"DEBUG - Retrieved policy context length: {len(policy_context)}")
            print(f"DEBUG - Policy context sample: {policy_context[:200]}...")
        else:
            print("DEBUG - No specific policy information retrieved")
        
        # Generate recommendation
        print("DEBUG - Calling generate_recommendation")
        final_state = generate_recommendation(retrieval_state)
        
        # Check if we have a recommendation
        recommendation = final_state.get("recommendation", "")
        if recommendation:
            print(f"DEBUG - Generated recommendation length: {len(recommendation)}")
            print(f"DEBUG - Recommendation sample: {recommendation[:200]}...")
        else:
            print("DEBUG - No recommendation generated")
            # Create a fallback recommendation
            final_state["recommendation"] = """
            <h2>Insurance Recommendation</h2>
            
            <p>Based on the information provided about your company, we recommend considering the following insurance policies:</p>
            
            <h3>POLICY RECOMMENDATIONS</h3>
            <ul>
                <li><strong>General Liability Insurance</strong> - Essential coverage for all businesses</li>
                <li><strong>Professional Liability Insurance</strong> - Protects against claims of errors or negligence</li>
                <li><strong>Cyber Liability Insurance</strong> - Coverage for data breaches and cyber attacks</li>
                <li><strong>Commercial Property Insurance</strong> - Protects your physical assets</li>
                <li><strong>Business Interruption Insurance</strong> - Covers lost income during disruptions</li>
            </ul>
            
            <h3>WHY THESE POLICIES</h3>
            <p>These policies provide a foundation of protection for most businesses. For more tailored recommendations, please provide additional details about your specific industry, size, and risk concerns.</p>
            
            <h3>COST CONSIDERATIONS</h3>
            <p>Insurance costs vary widely based on your business specifics. We recommend consulting with a licensed insurance broker who can provide detailed quotes based on your exact needs.</p>
            """
        
        # Return the recommendation
        print("DEBUG - Returning recommendation response")
        return jsonify({
            "recommendation": final_state.get("recommendation", "No recommendation available."),
            "company_profile": final_state.get("company_profile", ""),
            "chunk_count": len(final_state.get("retrieved_chunks", []))
        })
    except Exception as e:
        print(f"ERROR generating recommendation: {e}")
        import traceback
        print(f"DEBUG - Exception traceback: {traceback.format_exc()}")
        
        # Return a more useful error response
        return jsonify({
            "error": str(e),
            "recommendation": """
            <h2>Insurance Recommendation</h2>
            
            <p>We encountered an issue while generating your personalized recommendation, but we can still provide some general guidance.</p>
            
            <h3>GENERAL RECOMMENDATIONS</h3>
            <ul>
                <li><strong>General Liability Insurance</strong> - Essential coverage for all businesses</li>
                <li><strong>Professional Liability Insurance</strong> - Protects against claims of errors or negligence</li>
                <li><strong>Cyber Liability Insurance</strong> - Coverage for data breaches and cyber attacks</li>
            </ul>
            
            <p>For more tailored recommendations, please try again or consult with a licensed insurance broker.</p>
            """
        })

@app.route('/search', methods=['POST'])
def direct_search():
    """Allow direct search of policies"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Get all available datasets
    datasets = get_datasets()
    dataset_ids = [d['id'] for d in datasets] if datasets else []
    
    if not dataset_ids:
        return jsonify({"error": "No datasets available for search"}), 404
    
    # Retrieve relevant chunks from RAGFlow
    chunks = document_retrieval(dataset_ids, query)
    
    return jsonify({"results": chunks})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(
        host=settings.APP_HOST, 
        port=settings.APP_PORT, 
        debug=settings.APP_DEBUG
    )