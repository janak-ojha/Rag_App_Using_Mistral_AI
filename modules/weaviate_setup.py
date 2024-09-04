import weaviate
import os

def create_weaviate_client():
    # Retrieve the environment variables
    url = os.getenv("WEAVIATE_CLUSTER")
    api_key = os.getenv("WEAVIATE_API_KEY")
    
    # Create the Weaviate client with WCS connection
    client = weaviate.connect_to_wcs(
        cluster_url=url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key)
    )
    
    return client
