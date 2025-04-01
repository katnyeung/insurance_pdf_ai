from app import app
from config import settings
import os

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs(settings.TEMPLATES_DIR, exist_ok=True)
    os.makedirs(settings.STATIC_DIR, exist_ok=True)
    
    print(f"Starting Insurance Recommendation System...")
    print(f"RAGFlow API URL: {settings.RAGFLOW_API_URL}")
    print(f"LLM Model: {settings.LLM_MODEL}")
    print(f"Running on {settings.APP_HOST}:{settings.APP_PORT} (Debug: {settings.APP_DEBUG})")
    
    # Check if RAGFlow is accessible
    import requests
    try:
        response = requests.get(f"{settings.RAGFLOW_API_URL}/api/v1/datasets", 
                                headers={"Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"}, 
                                timeout=5)
        if response.status_code == 200:
            datasets = response.json().get("data", {}).get("items", [])
            print(f"Connected to RAGFlow successfully. Found {len(datasets)} datasets.")
        else:
            print(f"Warning: RAGFlow returned status code {response.status_code}. The system may not function correctly.")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to RAGFlow at {settings.RAGFLOW_API_URL}. Error: {e}")
        print("The application will still start, but RAGFlow-related features may not work.")

    # Run the Flask app
    app.run(
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        debug=settings.APP_DEBUG
    )