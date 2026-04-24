# Ganti semua import di atas dengan ini
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pathlib import Path

# Ini cara paling aman untuk import RetrievalQA
try:
    from langchain.chains import RetrievalQA
except ImportError:
    try:
        from langchain.chains.retrieval_qa.base import RetrievalQA
    except ImportError:
        from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="Chatbot Sekolah", page_icon="🤖")
st.title("🤖 Chatbot Sekolah Ora et Labora")
st.markdown("Tanya apapun tentang sekolah!")

# 1. KONFIGURASI API KEY (Pastikan sudah di set di Streamlit Secrets)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key belum disetting!")
    with st.expander("🔧 Cara Setting API Key"):
        st.code("""
1. Dapatkan API Key dari https://aistudio.google.com/
2. Di Streamlit Cloud: Settings -> Secrets
3. Tambahkan: GEMINI_API_KEY = "your-api-key-here"
        """)
    st.stop()

# Konfigurasi API
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 2. PATH DATA (Gunakan relative path)
DATA_PATH = Path("data")
if not DATA_PATH.exists():
    st.error("❌ Folder 'data' tidak ditemukan!")
    st.info("Pastikan folder 'data' berisi file PDF/TXT/DOCX tentang sekolah")
    st.stop()

# 3. LOAD DOKUMEN dengan cache agar tidak reload ulang setiap saat
@st.cache_resource(show_spinner=False)
def load_and_process_documents():
    documents = []
    files = list(DATA_PATH.glob("*.pdf")) + list(DATA_PATH.glob("*.txt")) + list(DATA_PATH.glob("*.docx"))
    
    if not files:
        return None
    
    progress_text = st.empty()
    for i, file in enumerate(files):
        progress_text.text(f"📖 Membaca file: {file.name} ({i+1}/{len(files)})")
        try:
            if file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif file.suffix == ".docx":
                loader = Docx2txtLoader(str(file))
            else:
                continue
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"⚠️ Gagal baca {file.name}: {e}")
    progress_text.empty()
    return documents

# 4. PROSES DOKUMEN JADI VECTOR DATABASE
@st.cache_resource(show_spinner=False)
def get_vectorstore(docs):
    if not docs:
        return None
        
    with st.spinner("✂️ Memotong teks jadi potongan kecil..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Ukuran kecil agar cepat
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(docs)
    
    # 🔥 (KUNCI PERBAIKAN): PAKAI EMBEDDINGS LOKAL (HuggingFace)
    # Ini GRATIS, tidak perlu API Key, dan tidak timeout karena jalan di CPU
    with st.spinner(f"🧠 Memproses {len(texts)} potongan teks (Metode Cepat Lokal)..."):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Buat Vector Store (Proses cepat karena lokal)
        vectorstore = FAISS.from_documents(texts[:200], embeddings) # Proses bertahap
        
        if len(texts) > 200:
            vectorstore.add_documents(texts[200:])
    
    return vectorstore

# 5. LOAD DAN PROSES (eksekusi utama)
docs = load_and_process_documents()
if docs is None:
    st.error("❌ Tidak ada file yang didukung di folder 'data'!")
    st.stop()

vector_db = get_vectorstore(docs)
if vector_db is None:
    st.error("❌ Gagal memproses dokumen!")
    st.stop()

# 6. SETUP LLM (Gemini)
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        request_timeout=30,
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )

llm = load_llm()

# Buat Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}), # Ambil 2 dokumen saja
    return_source_documents=True
)

# 7. INTERFACE CHAT (Loop percakapan)
st.success("✅ Sistem siap! Silakan tanya tentang sekolah.")

# Inisialisasi history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input chat
if prompt := st.chat_input("Ketik pertanyaan Anda tentang sekolah..."):
    # Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Proses jawaban
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Opsional: Tombol untuk lihat sumber
                with st.expander("📖 Lihat referensi jawaban"):
                    for i, doc in enumerate(response["source_documents"]):
                        source_name = Path(doc.metadata.get("source", "Unknown")).name
                        st.write(f"**Sumber {i+1}:** {source_name}")
                        st.write(doc.page_content[:300] + "...")
            except Exception as e:
                st.error(f"Maaf, terjadi kesalahan: {str(e)}")
                st.info("Coba tanyakan pertanyaan lain atau restart aplikasi.")
