from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .config import HUGGING_FACE_API_TOKEN, MODEL_ID

def create_rag_chain(vector_db):
    template = """ You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = HuggingFaceEndpoint(
        huggingfacehub_api_token=HUGGING_FACE_API_TOKEN,
        repo_id=MODEL_ID,
        model_kwargs={"max_length": 180},
        temperature=1
    )

    output_parser = StrOutputParser()

    rag_chain = (
        {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )
    
    return rag_chain
