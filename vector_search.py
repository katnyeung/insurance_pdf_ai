import requests
import uuid
import time
from config import settings

def document_retrieval(dataset_ids, query, similarity_threshold=0.3, vector_similarity_weight=0.5, top_k=15, max_retries=3, retry_delay=1):
    """Retrieve relevant chunks from datasets based on a query with retry logic"""
    url = f"{settings.RAGFLOW_API_URL}/api/v1/datasets/retrieval"
    headers = {
        "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Handle single dataset_id as string
    if isinstance(dataset_ids, str):
        dataset_ids = [dataset_ids]
    
    # Handle empty dataset_ids
    if not dataset_ids:
        print("Warning: No dataset_ids provided for retrieval")
        return []
    
    data = {
        "question": query,
        "dataset_ids": dataset_ids,
        "similarity_threshold": similarity_threshold,
        "vector_similarity_weight": vector_similarity_weight,
        "top_k": top_k,
        "highlight": True,
        "page_size": 200
    }
    
    # Implement retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if "data" in result and "chunks" in result["data"]:
                    # Process and format the chunks
                    processed_chunks = []
                    for chunk in result["data"]["chunks"]:
                        processed_chunk = {
                            "content": chunk.get("content", ""),
                            "source": chunk.get("source", "Unknown"),
                            "score": chunk.get("score", 0),
                            "highlighted_content": chunk.get("highlighted_content", "")
                        }
                        processed_chunks.append(processed_chunk)
                    return processed_chunks
                else:
                    print(f"DEBUG - No chunks found in response: {result}")
                    return []
            elif response.status_code == 429:  # Rate limit
                print(f"DEBUG - Rate limited, retrying after delay ({attempt+1}/{max_retries})")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                print(f"DEBUG - Error retrieving chunks: {response.status_code}, {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return []
        except Exception as e:
            print(f"DEBUG - Exception in document_retrieval: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return []
    
    # If we've exhausted all retries
    return []

def get_datasets(name=None, max_retries=3, retry_delay=1):
    """Get all datasets or filter by name with retry logic"""
    url = f"{settings.RAGFLOW_API_URL}/api/v1/datasets"
    if name:
        url += f"?name={name}"
    
    headers = {
        "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"
    }
    
    # Implement retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                datasets = result.get("data", {}).get("items", [])
                print(f"DEBUG - Found {len(datasets)} datasets")
                return datasets
            elif response.status_code == 429:  # Rate limit
                print(f"DEBUG - Rate limited, retrying after delay ({attempt+1}/{max_retries})")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                print(f"Error getting datasets: {response.status_code}, {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return []
        except Exception as e:
            print(f"Exception in get_datasets: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return []
    
    # If we've exhausted all retries
    return []

def search_documents_with_context(query, context=None, dataset_ids=None):
    """
    Search documents with context enhancement
    
    Args:
        query (str): The search query
        context (str, optional): Additional context to enhance the search
        dataset_ids (list, optional): List of dataset IDs to search in
    
    Returns:
        list: List of relevant document chunks
    """
    # Enhance the query with context if provided
    enhanced_query = query
    if context:
        enhanced_query = f"{query} (Context: {context})"
    
    # Get all datasets if none specified
    if not dataset_ids:
        datasets = get_datasets()
        dataset_ids = [d.get("id") for d in datasets if d.get("id")]
    
    # Retrieve documents
    chunks = document_retrieval(dataset_ids, enhanced_query)
    
    return chunks