import streamlit as st
import os\
import requests

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}


from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# API KEY
api_token = os.getenv("REPLICATE_API_TOKEN")

st.title("Chatbot Sekolah Ora et Labora 🤖")

# LOAD FILES
documents = []

for file in os.listdir("data"):
    path = os.path.join("data", file)
    
    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif file.endswith(".txt"):
        loader = TextLoader(path)
    elif file.endswith(".docx"):
        loader = Docx2txtLoader(path)
    else:
        continue
        
    documents.extend(loader.load())

# SPLIT TEXT
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# VECTOR DB
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# QA SYSTEM
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=db.as_retriever()
)

# CHAT UI
query = st.text_input("Tanya tentang sekolah:")

if query:
    answer = qa.run(query)
    st.write("🤖", answer)
