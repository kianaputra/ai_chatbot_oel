import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.title("Chatbot Sekolah Ora et Labora 🤖")

# CEK FOLDER DATA
if not os.path.exists("data"):
    st.error("Folder 'data' tidak ditemukan ❌")
    st.stop()

# LOAD DOCUMENTS
documents = []

for file in os.listdir("data"):
    path = os.path.join("data", file)

    try:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        documents.extend(loader.load())
    except Exception as e:
        st.warning(f"Gagal load file: {file}")

# CEK DATA KOSONG
if not documents:
    st.error("Tidak ada dokumen yang bisa dibaca ❌")
    st.stop()

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
    try:
        result = qa.invoke(query)
        st.write("🤖", result["result"])
    except Exception as e:
        st.error("Terjadi error saat memproses pertanyaan ❌")


