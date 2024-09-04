import streamlit as st
from dotenv import load_dotenv
import os
import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_CLUSTER = os.getenv('WEAVIATE_CLUSTER')
hugging_face_api_token = os.getenv('hugging_face_api_token')
model_id = os.getenv('model_id')

# Initialize the Weaviate client
client = weaviate.Client(
    url=WEAVIATE_CLUSTER,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

# Initialize the embeddings
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

# Initialize the model
model = HuggingFaceEndpoint(
    huggingfacehub_api_token=hugging_face_api_token,
    repo_id=model_id,
    model_kwargs={"max_length": 180},
    temperature=1
)

# Define the output parser
output_parser = StrOutputParser()

# Define the prompt
template = """ You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Streamlit UI
st.title('RAG System PDF Processor')

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing..."):
        # Load and process PDF
        loader = PyPDFLoader(uploaded_file, extract_images=True)
        pages = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        docs = text_splitter.split_documents(pages)

        # Create vector DB
        vector_db = Weaviate.from_documents(
            docs, embeddings, client=client, by_text=False
        )

        # Initialize the retriever
        retriever = vector_db.as_retriever()

        # Define the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | output_parser
        )

        # Text input for the user's question
        question = st.text_input("Enter your question:")

        # Button to get the answer
        if st.button("Get Answer"):
            if question:
                with st.spinner("Getting answer..."):
                    # Invoke the RAG chain with the user's question
                    result = rag_chain.invoke({"question": question})
                    st.write(result)
            else:
                st.warning("Please enter a question.")
