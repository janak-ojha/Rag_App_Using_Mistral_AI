from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

def process_pdf(uploaded_file):
    try:
        st.write("Starting PDF loading...")
        loader = PyPDFLoader(uploaded_file, extract_images=True)
        pages = loader.load()
        st.write(f"Loaded {len(pages)} pages.")

        st.write("Starting text splitting...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        docs = text_splitter.split_documents(pages)
        st.write(f"Processed {len(docs)} document chunks.")
        
        return docs
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return []
