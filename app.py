import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

st.title("PDF Question Answering with LangChain")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load data
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    
    # API Key input
    api_key = st.text_input("Enter your MistralAI API Key", type="password")
    
    if api_key:
        # Define the embedding model
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
        
        # Create the vector store
        vector = FAISS.from_documents(documents, embeddings)
        
        # Define a retriever interface
        retriever = vector.as_retriever()
        
        # Define LLM
        model = ChatMistralAI(mistral_api_key=api_key)
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        
        <context>
        {context}
        </context>
        
        Question: {input}""")
        
        # Create a retrieval chain to answer questions
        document_chain = create_stuff_documents_chain(model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # User prompt input
        user_prompt = st.text_input("Enter your question")
        
        if user_prompt:
            with st.spinner("Processing..."):
                response = retrieval_chain.invoke({"input": user_prompt})
                st.write(response["answer"])

else:
    st.write("Please upload a PDF file to get started.")
