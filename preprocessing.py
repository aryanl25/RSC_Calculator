# rsc_module/preprocessing.py

import re

def preprocess_query(query: str) -> str:
    """
    Preprocess the raw SQL query by removing comments and extra whitespace.
    
    Args:
        query (str): The raw SQL query.
        
    Returns:
        str: The cleaned SQL query.
    """
    # Remove single-line comments (e.g., -- comment)
    query = re.sub(r'(--[^\n]*)', '', query)
    
    # Remove multi-line comments (e.g., /* comment */)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query
