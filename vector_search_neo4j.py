import time
from neo4j import GraphDatabase
from config import settings

# Neo4j connection parameters
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "****"  # Replace with actual password if different

class Neo4jConnection:
    """Class to manage Neo4j database connection"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnection, cls).__new__(cls)
            cls._instance.driver = None
            cls._instance.connect()
        return cls._instance
    
    def connect(self, max_retries=3, retry_delay=1):
        """Establish connection to Neo4j database with retry logic"""
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    NEO4J_URI, 
                    auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                # Verify connection
                self.driver.verify_connectivity()
                print("DEBUG - Successfully connected to Neo4j database")
                return True
            except Exception as e:
                print(f"DEBUG - Neo4j connection error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print("DEBUG - Failed to connect to Neo4j after all retries")
                    return False
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def execute_query(self, query, params=None, max_retries=3, retry_delay=1):
        """Execute a Cypher query with retry logic"""
        if not self.driver:
            if not self.connect():
                return None
        
        for attempt in range(max_retries):
            try:
                with self.driver.session() as session:
                    result = session.run(query, params)
                    return [record for record in result]
            except Exception as e:
                print(f"DEBUG - Neo4j query error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    # Try to reconnect
                    self.connect()
                else:
                    print("DEBUG - Failed to execute Neo4j query after all retries")
                    return None

def document_retrieval(dataset_ids, query, similarity_threshold=0.2, vector_similarity_weight=0.3, top_k=1024, max_retries=3, retry_delay=1):
    """
    Retrieve relevant chunks from Neo4j database based on a query.
    
    Parameters:
    - query: The user query or query keywords
    - dataset_ids: List of policy IDs to search (used as filter)
    - similarity_threshold: Minimum similarity score (default: 0.2) - used for filtering
    - vector_similarity_weight: Weight parameter (not used in Neo4j implementation)
    - top_k: Maximum number of results to return (default: 1024)
    """
    print("\nDEBUG - Starting document_retrieval function with Neo4j")
    
    # Handle single dataset_id as string
    if isinstance(dataset_ids, str):
        dataset_ids = [dataset_ids]
    
    # Handle empty dataset_ids
    if not dataset_ids:
        print("Warning: No dataset_ids provided for retrieval")
        return []
    
    print(f"DEBUG - Retrieving with dataset_ids: {dataset_ids}")
    
    # Connect to Neo4j
    neo4j_conn = Neo4jConnection()
    
    # Prepare Cypher query for text search
    # This query searches for Clause nodes that contain text similar to the query
    # and are connected to policies specified in dataset_ids
    cypher_query = """
    CALL db.index.fulltext.queryNodes('clause_text_idx', $query_text) 
    YIELD node, score
    WHERE score >= $threshold
    MATCH (p:Policy)-[:CONTAINS_CLAUSE]->(node)
    WHERE p.policyId IN $policy_ids
    RETURN 
        node.text AS content,
        p.insurer + ' - ' + p.policyName AS source,
        score,
        node.text AS highlight
    ORDER BY score DESC
    LIMIT $limit
    """
    
    # If no specific policies are requested, search across all policies
    if not dataset_ids or dataset_ids[0] == "all":
        cypher_query = """
        CALL db.index.fulltext.queryNodes('clause_text_idx', $query_text) 
        YIELD node, score
        WHERE score >= $threshold
        MATCH (p:Policy)-[:CONTAINS_CLAUSE]->(node)
        RETURN 
            node.text AS content,
            p.insurer + ' - ' + p.policyName AS source,
            score,
            node.text AS highlight
        ORDER BY score DESC
        LIMIT $limit
        """
        
    # Prepare parameters
    params = {
        "query_text": query,
        "threshold": similarity_threshold,
        "policy_ids": dataset_ids,
        "limit": top_k
    }
    
    try:
        # Execute query
        results = neo4j_conn.execute_query(cypher_query, params)
        
        if not results:
            print("DEBUG - No results found or query failed")
            return []
        
        # Process and format the chunks
        processed_chunks = []
        for record in results:
            processed_chunk = {
                "content": record["content"],
                "source": record["source"],
                "score": record["score"],
                "highlighted_content": record["highlight"]
            }
            processed_chunks.append(processed_chunk)
        
        print(f"DEBUG - Processed {len(processed_chunks)} chunks from Neo4j")
        return processed_chunks
        
    except Exception as e:
        print(f"DEBUG - Exception in Neo4j document_retrieval: {e}")
        return []

def get_datasets(name=None, max_retries=3, retry_delay=1):
    """
    Get all policies from Neo4j database or filter by name
    
    Returns a list of dictionaries with policy information
    """
    print(f"DEBUG - Getting datasets from Neo4j")
    
    # Connect to Neo4j
    neo4j_conn = Neo4jConnection()
    
    # Prepare Cypher query
    cypher_query = """
    MATCH (p:Policy)
    WHERE $name IS NULL OR p.policyName CONTAINS $name
    RETURN p.policyId AS id, p.policyName AS name, p.insurer AS insurer
    """
    
    params = {"name": name}
    
    try:
        # Execute query
        results = neo4j_conn.execute_query(cypher_query, params)
        
        if not results:
            print("DEBUG - No policies found or query failed")
            return []
        
        # Format results to match expected structure
        datasets = []
        for record in results:
            dataset = {
                "id": record["id"],
                "name": record["name"],
                "insurer": record["insurer"]
            }
            datasets.append(dataset)
        
        print(f"DEBUG - Found {len(datasets)} policies in Neo4j")
        if datasets:
            print(f"DEBUG - First policy sample: {datasets[0]}")
        
        return datasets
        
    except Exception as e:
        print(f"DEBUG - Exception in Neo4j get_datasets: {e}")
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
