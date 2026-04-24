import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

st.title("🤖 Chatbot Sekolah Ora et Labora")
st.markdown("Tanya apapun tentang sekolah!")

# Cek API key
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key belum disetting!")
    with st.expander("🔧 Cara Setting API Key"):
        st.code("""
1. Dapatkan API Key dari https://aistudio.google.com/
2. Di Streamlit Cloud: Settings → Secrets
3. Tambahkan: GEMINI_API_KEY = "your-api-key-here"
        """)
    st.stop()

# Setup API key untuk kedua library
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Konfigurasi tambahan untuk Gemini
import google.generativeai as genai
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Cek folder data
if not os.path.exists("data"):
    st.error("❌ Folder 'data' tidak ditemukan!")
    st.info("Buat folder 'data' dan masukkan file PDF/TXT/DOCX tentang sekolah")
    st.stop()

# Load dokumen
documents = []
with st.spinner("📚 Memuat dokumen sekolah..."):
    for file in os.listdir("data"):
        path = os.path.join("data", file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue
            documents.extend(loader.load())
            st.success(f"✅ Loaded: {file}")
        except Exception as e:
            st.warning(f"⚠️ Gagal load {file}: {str(e)[:100]}")

if not documents:
    st.error("❌ Tidak ada dokumen yang bisa dibaca!")
    st.stop()

# Split teks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Diperkecil dari 1000 untuk menghindari timeout
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

st.info(f"📄 Dokumen diproses: {len(texts)} potongan teks")

# Buat embeddings dan vector store
with st.spinner("🧠 Memproses pengetahuan sekolah (ini bisa makan waktu 1-2 menit)..."):
    try:
        # Gunakan model embedding yang lebih ringan
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            request_timeout=120.0  # Tambah timeout jadi 120 detik
        )
        
        # Proses FAISS dalam batch yang lebih kecil
        batch_size = 100
        if len(texts) > batch_size:
            st.info(f"⚙️ Memproses {len(texts)} potongan teks dalam beberapa batch...")
            # Inisialisasi dengan batch pertama
            db = FAISS.from_documents(texts[:batch_size], embeddings)
            # Tambahkan sisa batch
            for i in range(batch_size, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                db.add_documents(batch)
                st.progress(min(100, int((i+batch_size)/len(texts)*100)))
        else:
            db = FAISS.from_documents(texts, embeddings)
        
        st.success("✅ Database vektor berhasil dibuat!")
    except Exception as e:
        st.error(f"❌ Error saat membuat embeddings: {str(e)}")
        st.info("💡 Tips: Coba restart app atau kurangi ukuran file di folder data")
        st.stop()

# Setup LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    request_timeout=120.0
)

# Buat QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Interface chat
st.markdown("---")
query = st.text_input("💬 Tanya tentang sekolah:", placeholder="Contoh: Apa saja ekstrakurikuler yang tersedia?")

if query:
    with st.spinner("🤔 Memikirkan jawaban..."):
        try:
            response = qa_chain.invoke({"query": query})
            st.write("### 🤖 Jawaban:")
            st.write(response["result"])
            
            # Tampilkan sumber
            if "source_documents" in response:
                with st.expander("📖 Lihat sumber referensi"):
                    for i, doc in enumerate(response["source_documents"][:3]):
                        st.write(f"**Sumber {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:300] + "...")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Coba tanyakan pertanyaan yang lebih sederhana dulu.")

