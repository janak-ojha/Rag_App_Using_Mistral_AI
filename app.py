import streamlit as st
from modules.config import WEAVIATE_CLUSTER
from modules.weaviate_setup import create_weaviate_client
from modules.embeddings_setup import create_embeddings
from modules.pdf_processor import process_pdf
from modules.rag_chain import create_rag_chain

def main():
    st.title('RAG System PDF Processor')

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        st.write("PDF uploaded. Starting processing...")

        with st.spinner("Processing..."):
            try:
                # Initialize Weaviate client and embeddings
                client = create_weaviate_client()
                st.write("Weaviate client created successfully.")
                embeddings = create_embeddings()
                st.write("Embeddings created successfully.")

                # Process PDF
                docs = process_pdf(uploaded_file)
                st.write(f"PDF processed. Number of documents: {len(docs)}")

                # Create vector DB
                from langchain.vectorstores import Weaviate
                vector_db = Weaviate.from_documents(
                    docs, embeddings, client=client, by_text=False
                )
                st.write("Vector DB created successfully.")

                # Initialize RAG chain
                rag_chain = create_rag_chain(vector_db)
                st.write("RAG chain initialized successfully.")

                # Text input for user question
                question = st.text_input("Enter your question:")

                # Button to submit the question
                if st.button("Get Answer"):
                    if question:
                        with st.spinner("Getting answer..."):
                            # Invoke the RAG chain with the user's question
                            result = rag_chain.invoke(question)
                            st.write(result)
                    else:
                        st.warning("Please enter a question.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a PDF file to start.")

if __name__ == "__main__":
    main()
