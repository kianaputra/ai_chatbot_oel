import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings

st.title("🤖 Chatbot Sekolah Ora et Labora")
st.markdown("Tanya apapun tentang sekolah!")

# Cek API key
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key belum disetting! Baca petunjuk di bawah.")
    with st.expander("🔧 Cara Setting API Key"):
        st.code("""
1. Dapatkan API Key dari https://aistudio.google.com/
2. Di Streamlit Cloud: Settings → Secrets
3. Tambahkan: GEMINI_API_KEY = "your-api-key-here"
        """)
    st.stop()

# Setup API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Buat embeddings dan vector store (pakai model gratis)
with st.spinner("🧠 Memproses pengetahuan sekolah..."):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Alternatif pakai HuggingFace (lebih lambat tapi gratis tanpa API):
    # embeddings = HFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

# Setup LLM (Google Gemini - gratis)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
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
            
            # Tampilkan sumber (opsional)
            with st.expander("📖 Lihat sumber referensi"):
                for i, doc in enumerate(response["source_documents"][:3]):
                    st.write(f"**Sumber {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content[:300] + "...")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


