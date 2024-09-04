from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )
    return embeddings
