import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

st.title("Chatbot Sekolah Ora et Labora 🤖")

# LOAD DOCUMENTS
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

# EMBEDDING (LOKAL)
embeddings = OllamaEmbeddings(model="llama3")
db = FAISS.from_documents(texts, embeddings)

# MODEL LOKAL
llm = Ollama(model="llama3")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# UI
query = st.text_input("Tanya tentang sekolah:")

if query:
    result = qa.run(query)
    st.write("🤖", result)


