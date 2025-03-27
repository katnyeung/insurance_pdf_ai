from insurance_recommender import insurance_recommender

def main():
    print("Welcome to the Insurance Recommendation System!")
    print("Starting up with the questions...")
    
    # Initialize state
    state = {"company_info": [], "collected_categories": []}

    # Create a thread_id for the conversation
    thread_id = "insurance_thread"
    
    # Initial invocation with thread_id
    result = insurance_recommender.invoke(
        state, 
        config={"configurable": {"thread_id": thread_id}}
    )
    
    while True:
        if "recommendation" in result:
            print("\nBased on the information provided, here's our recommendation:")
            print(result["recommendation"])
            break
        elif "next_question" in result:
            user_input = input(f"{result['next_question']}: ")
            result["user_input"] = user_input
            
            # Continue with the same thread_id
            result = insurance_recommender.invoke(
                result,
                config={"configurable": {"thread_id": thread_id}}
            )
        else:
            print("Error: Unexpected state returned from the graph")
            print(result)  # Print the result for debugging
            break

if __name__ == "__main__":
    main()
