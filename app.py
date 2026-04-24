import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from pathlib import Path

st.set_page_config(page_title="Chatbot Sekolah", page_icon="🤖")
st.title("🤖 Chatbot Sekolah Ora et Labora")
st.markdown("Tanya apapun tentang sekolah!")

# CEK API KEY
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key belum disetting!")
    with st.expander("🔧 Cara Setting API Key"):
        st.code("""
1. Dapatkan API Key dari https://aistudio.google.com/
2. Di Streamlit Cloud: Settings → Secrets
3. Tambahkan: GEMINI_API_KEY = "your-api-key-here"
        """)
    st.stop()

# SETUP API KEY
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

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
    
    # Cari semua file txt
    txt_files = list(DATA_PATH.glob("*.txt"))
    
    if not txt_files:
        st.error("❌ Tidak ada file .txt di folder data!")
        return None
    
    for file in txt_files:
        try:
            loader = TextLoader(str(file), encoding="utf-8")
            documents.extend(loader.load())
            st.success(f"✅ Loaded: {file.name}")
        except Exception as e:
            st.warning(f"⚠️ Gagal load {file.name}: {e}")
    
    return documents

# PROSES DOKUMEN JADI VECTOR DATABASE
@st.cache_resource
def create_vectorstore(documents):
    # Split teks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    st.info(f"📄 {len(texts)} potongan teks diproses")
    
    # Buat embeddings lokal (GRATIS, CEPAT)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Buat vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

# LOAD DOKUMEN
with st.spinner("📚 Membaca file dokumen sekolah..."):
    docs = load_documents()

if docs is None:
    st.stop()

# BUAT VECTOR STORE
with st.spinner("🧠 Memproses pengetahuan sekolah..."):
    try:
        vectorstore = create_vectorstore(docs)
        st.success("✅ Database pengetahuan siap!")
    except Exception as e:
        st.error(f"❌ Gagal memproses: {e}")
        st.stop()

# SETUP LLM GEMINI
@st.cache_resource
def setup_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )

try:
    llm = setup_llm()
except Exception as e:
    st.error(f"❌ Gagal setup LLM: {e}")
    st.stop()

# BUAT QA CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# INTERFACE CHAT
st.markdown("---")
st.success("✅ Siap bertanya! Silakan tulis pertanyaan tentang sekolah.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input pertanyaan
if prompt := st.chat_input("Tanya tentang sekolah..."):
    # Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Proses jawaban
    with st.chat_message("assistant"):
        with st.spinner("🤔 Mencari jawaban..."):
            try:
                # Panggil QA chain
                response = qa_chain.invoke(prompt)
                
                # Cek response
                if isinstance(response, dict):
                    answer = response.get('result', 'Maaf, tidak bisa menjawab.')
                else:
                    answer = str(response)
                
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
