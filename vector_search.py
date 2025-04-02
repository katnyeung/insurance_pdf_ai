import requests
import uuid
import time
from config import settings

def document_retrieval(dataset_ids, query, similarity_threshold=0.2, vector_similarity_weight=0.3, top_k=1024, max_retries=3, retry_delay=1):
    """
    Retrieve relevant chunks from datasets based on a query with retry logic.
    
    Parameters match the RAGFlow API /api/v1/retrieval endpoint:
    - query: The user query or query keywords
    - dataset_ids: List of dataset IDs to search
    - similarity_threshold: Minimum similarity score (default: 0.2)
    - vector_similarity_weight: Weight of vector cosine similarity (default: 0.3)
    - top_k: Number of chunks engaged in vector cosine computation (default: 1024)
    """
    print("\nDEBUG - Starting document_retrieval function")
    
    # Use the correct endpoint
    url = f"{settings.RAGFLOW_API_URL}/api/v1/retrieval"
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
    
    print(f"DEBUG - Retrieving with dataset_ids: {dataset_ids}")
    
    # Prepare request body according to API documentation
    data = {
        "question": query,  # Using 'question' as specified in the API docs
        "dataset_ids": dataset_ids,
        "similarity_threshold": similarity_threshold,
        "vector_similarity_weight": vector_similarity_weight,
        "top_k": top_k,
        "highlight": True,
        "page_size": 200,
        "keyword": False
    }
    
    print(f"DEBUG - Request payload: {data}")
    
    # Implement retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            print(f"DEBUG - Retrieval response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Debug the response structure
                if isinstance(result, dict):
                    print(f"DEBUG - Response structure: {list(result.keys())}")
                
                # Process response according to the API documentation structure
                chunks = []
                
                if isinstance(result, dict):
                    if result.get("code") == 0 and "data" in result:
                        if result["data"] is None:
                            print("DEBUG - Response data field is None")
                            return []
                        elif isinstance(result["data"], dict) and "chunks" in result["data"]:
                            chunks = result["data"]["chunks"]
                            print(f"DEBUG - Found {len(chunks)} chunks in result.data.chunks")
                
                # Process and format the chunks
                processed_chunks = []
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                        
                    processed_chunk = {
                        "content": chunk.get("content", ""),
                        "source": chunk.get("document_keyword", "Unknown"),
                        "score": chunk.get("similarity", 0),
                        "highlighted_content": chunk.get("highlight", "")
                    }
                    processed_chunks.append(processed_chunk)
                
                print(f"DEBUG - Processed {len(processed_chunks)} chunks")
                return processed_chunks
            
            elif response.status_code == 429:  # Rate limit
                print(f"DEBUG - Rate limited, retrying after delay ({attempt+1}/{max_retries})")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"DEBUG - Error retrieving chunks: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        except Exception as e:
            print(f"DEBUG - Exception in document_retrieval: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    # If all retries fail, return empty list - NO DEFAULT CONTENT
    print("DEBUG - All retrieval attempts failed, returning empty list")
    return []


def get_datasets(name=None, max_retries=3, retry_delay=1):
    """Get all datasets or filter by name with retry logic"""
    url = f"{settings.RAGFLOW_API_URL}/api/v1/datasets"
    if name:
        url += f"?name={name}"
    
    headers = {
        "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"
    }
    
    print(f"DEBUG - Attempting to get datasets from: {url}")
    print(f"DEBUG - Using API key ending with: {settings.RAGFLOW_API_KEY[-5:] if settings.RAGFLOW_API_KEY else 'None'}")
    
    # Implement retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"DEBUG - Get datasets response status: {response.status_code}")
            
            if response.status_code == 200:
                # Dump full response for debugging
                result = response.json()
                print(f"DEBUG - Raw response: {result}")
                
                # Handle different response structures
                datasets = []
                
                if isinstance(result, dict):
                    if "data" in result and isinstance(result["data"], dict) and "items" in result["data"]:
                        datasets = result["data"]["items"]
                        print(f"DEBUG - Found datasets in result.data.items")
                    elif "data" in result and isinstance(result["data"], list):
                        datasets = result["data"]
                        print(f"DEBUG - Found datasets in result.data (list)")
                    elif "code" in result and result["code"] == 0 and "data" in result:
                        # This appears to be your actual response structure
                        datasets = result["data"]
                        print(f"DEBUG - Found datasets in result.data with code 0")
                elif isinstance(result, list):
                    datasets = result
                    print(f"DEBUG - Result is already a list of datasets")
                
                print(f"DEBUG - Found {len(datasets)} datasets")
                if datasets:
                    print(f"DEBUG - First dataset sample: {datasets[0]}")
                
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
            print(f"DEBUG - Exception type: {type(e).__name__}")
            import traceback
            print(f"DEBUG - Traceback: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return []
    
    # If we've exhausted all retries
    print("DEBUG - Exhausted all retries in get_datasets")
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