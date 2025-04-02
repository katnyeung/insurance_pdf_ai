import requests
import json
import os
from config import settings

def check_ragflow_connection():
    """Check if RAGFlow API is accessible and print diagnostic information"""
    print(f"\n=== RAGFlow Diagnostic ===")
    print(f"RAGFlow API URL: {settings.RAGFLOW_API_URL}")
    print(f"API Key ends with: {settings.RAGFLOW_API_KEY[-5:] if settings.RAGFLOW_API_KEY else 'None'}")
    
    try:
        # Check basic connectivity
        response = requests.get(
            f"{settings.RAGFLOW_API_URL}/api/v1/datasets", 
            headers={"Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"},
            timeout=5
        )
        
        print(f"Connection test status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response type: {type(result)}")
            print(f"Response structure: {json.dumps(result, indent=2)[:300]}...")
            
            # Depending on response structure, get datasets
            if isinstance(result, dict) and "data" in result and "items" in result["data"]:
                datasets = result["data"]["items"]
            elif isinstance(result, list):
                datasets = result
            else:
                datasets = []
            
            print(f"Found {len(datasets)} datasets")
            for i, dataset in enumerate(datasets[:3]):  # Show up to 3 datasets
                print(f"  Dataset {i+1}: {dataset}")
        else:
            print(f"Error response: {response.text}")
            
        # Check required endpoints
        endpoints = [
            "/api/v1/datasets",
            "/api/v1/datasets/retrieval"
        ]
        
        print("\nEndpoint availability:")
        for endpoint in endpoints:
            try:
                response = requests.head(
                    f"{settings.RAGFLOW_API_URL}{endpoint}",
                    headers={"Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"},
                    timeout=2
                )
                print(f"  {endpoint}: {'Available' if response.status_code < 400 else 'Not available'} ({response.status_code})")
            except Exception as e:
                print(f"  {endpoint}: Error - {str(e)}")
        
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
    
    print("=== End of Diagnostic ===\n")

def test_pdf_upload(file_path):
    """Test uploading a PDF file to RAGFlow"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    print(f"\n=== Testing PDF Upload ===")
    print(f"File: {file_path}")
    
    # Step 1: Upload the file
    url = f"{settings.RAGFLOW_API_URL}/api/v1/files"
    headers = {"Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"}
    
    try:
        with open(file_path, "rb") as file:
            files = {"file": file}
            response = requests.post(url, headers=headers, files=files)
        
        print(f"Upload status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            file_id = result.get("id")
            print(f"File uploaded successfully with ID: {file_id}")
            return file_id
        else:
            print(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return None
    
    print("=== End of Test ===\n")

def test_create_dataset(name="Test Dataset"):
    """Test creating a dataset in RAGFlow"""
    url = f"{settings.RAGFLOW_API_URL}/api/v1/datasets"
    headers = {
        "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "name": name,
        "description": "Created for testing purposes"
    }
    
    print(f"\n=== Testing Dataset Creation ===")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Creation status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            dataset_id = result.get("id")
            print(f"Dataset created successfully with ID: {dataset_id}")
            return dataset_id
        else:
            print(f"Creation failed: {response.text}")
            return None
    except Exception as e:
        print(f"Creation error: {str(e)}")
        return None
    
    print("=== End of Test ===\n")

if __name__ == "__main__":
    check_ragflow_connection()
    
    # Uncomment to test dataset creation
    # dataset_id = test_create_dataset("Insurance Policies")
    
    # Uncomment to test PDF upload (specify path to a PDF file)
    # test_pdf_upload("/path/to/your/file.pdf")