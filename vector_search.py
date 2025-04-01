import requests
import uuid
from config import settings

def document_retrieval(dataset_ids, query, similarity_threshold=0.3, vector_similarity_weight=0.5, top_k=15):
    """Retrieve relevant chunks from datasets based on a query"""
    url = f"{settings.RAGFLOW_API_URL}/api/v1/retrieval"
    headers = {
        "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "question": query,
        "dataset_ids": dataset_ids,
        "similarity_threshold": similarity_threshold,
        "vector_similarity_weight": vector_similarity_weight,
        "top_k": top_k,
        "highlight": True,
        "page_size": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and "chunks" in result["data"]:
                # Extract the content from each chunk
                chunks = result["data"]["chunks"]
                return chunks
            else:
                print(f"DEBUG - No chunks found in response: {result}")
                return []
        else:
            print(f"DEBUG - Error retrieving chunks: {response.text}")
            return []
    except Exception as e:
        print(f"DEBUG - Exception in document_retrieval: {e}")
        return []

def get_datasets(name=None):
    """Get all datasets or filter by name"""
    url = f"{settings.RAGFLOW_API_URL}/api/v1/datasets"
    if name:
        url += f"?name={name}"
    
    headers = {
        "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"DEBUG - Datasets: {response.json()}")
            return response.json().get("data", [])
        else:
            print(f"Error getting datasets: {response.text}")
            return []
    except Exception as e:
        print(f"Exception in get_datasets: {e}")
        return []
