from dotenv import load_dotenv
import os

load_dotenv()

WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_CLUSTER = os.getenv('WEAVIATE_CLUSTER')
HUGGING_FACE_API_TOKEN = os.getenv('HUGGING_FACE_API_TOKEN')
MODEL_ID = os.getenv('MODEL_ID')
