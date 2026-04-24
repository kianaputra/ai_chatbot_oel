import streamlit as st
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from pathlib import Path

st.set_page_config(page_title="Chatbot Sekolah", page_icon="🤖")
st.title("🤖 Chatbot Sekolah Ora et Labora")
st.markdown("Tanya apapun tentang sekolah!")

# CEK OLLAMA (akan error jika Ollama tidak berjalan)
try:
    # Test koneksi ke Ollama
    test_llm = Ollama(model="llama3.2:3b")
    test_llm.invoke("test")
except Exception as e:
    st.error("❌ Ollama tidak terdeteksi!")
    st.info("""
    **Cara mengatasi:**
    1. Pastikan Ollama sudah terinstall di komputer/laptop Anda
    2. Jalankan di terminal: `ollama serve`
    3. Download model: `ollama pull llama3.2:3b` dan `ollama pull nomic-embed-text`
    4. Restart aplikasi
    """)
    st.stop()

# CEK FOLDER DATA
DATA_PATH = Path("data")
if not DATA_PATH.exists():
    st.error("❌ Folder 'data' tidak ditemukan!")
    st.info("Buat folder 'data' dan masukkan file TXT/PDF/DOCX tentang sekolah")
    st.stop()

# LOAD DOKUMEN
@st.cache_resource
def load_documents():
    documents = []
    files = list(DATA_PATH.glob("*.txt")) + list(DATA_PATH.glob("*.pdf")) + list(DATA_PATH.glob("*.docx"))
    
    if not files:
        return None
    
    for file in files:
        try:
            if file.suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix == ".docx":
                loader = Docx2txtLoader(str(file))
            else:
                continue
            documents.extend(loader.load())
            st.success(f"✅ Loaded: {file.name}")
        except Exception as e:
            st.warning(f"⚠️ Gagal baca {file.name}: {e}")
    
    return documents

# PROSES DOKUMEN
@st.cache_resource
def create_vectorstore(documents):
    # Split teks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    st.info(f"📄 {len(texts)} potongan teks diproses")
    
    # Buat embeddings dengan Ollama LOKAL (gratis, cepat)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

# LOAD DAN PROSES
with st.spinner("📚 Membaca file dokumen sekolah..."):
    docs = load_documents()

if docs is None:
    st.error("❌ Tidak ada file di folder 'data'!")
    st.info("Masukkan file .txt atau .pdf tentang sekolah ke folder 'data'")
    st.stop()

with st.spinner("🧠 Memproses pengetahuan sekolah (proses awal hanya sekali)..."):
    vectorstore = create_vectorstore(docs)
    st.success("✅ Database pengetahuan siap!")

# SETUP LLM dengan Ollama
@st.cache_resource
def setup_llm():
    return Ollama(
        model="llama3.2:3b",
        temperature=0.3,
        num_predict=512  # Batasi panjang output agar cepat
    )

llm = setup_llm()

# BUAT QA CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# INTERFACE CHAT
st.markdown("---")
st.success("✅ Sistem siap! Silakan tanya tentang sekolah.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input pertanyaan
if prompt := st.chat_input("Tanya tentang sekolah..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                with st.expander("📖 Lihat sumber jawaban"):
                    for i, doc in enumerate(response["source_documents"][:2]):
                        source_name = Path(doc.metadata.get("source", "Unknown")).name
                        st.write(f"**Sumber:** {source_name}")
                        st.write(doc.page_content[:300] + "...")
            except Exception as e:
                st.error(f"Error: {str(e)}")
