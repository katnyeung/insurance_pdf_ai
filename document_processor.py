import requests
from config import settings

def upload_and_index_pdf(file_path: str, knowledge_base_id: str):
    url = f"{settings.RAGFLOW_API_URL}/api/v1/files"
    headers = {"Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"}
    
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        file_id = response.json()["id"]
        index_url = f"{settings.RAGFLOW_API_URL}/api/v1/knowledge_bases/{knowledge_base_id}/index"
        index_data = {"file_ids": [file_id]}
        index_response = requests.post(index_url, headers=headers, json=index_data)
        
        if index_response.status_code == 200:
            print(f"File uploaded and indexed successfully: {file_id}")
        else:
            print(f"Error indexing file: {index_response.text}")
    else:
        print(f"Error uploading file: {response.text}")
