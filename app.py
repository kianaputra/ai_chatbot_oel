import streamlit as st
import requests
from pathlib import Path
import json

st.set_page_config(page_title="Chatbot Sekolah", page_icon="🤖")
st.title("🤖 Chatbot Sekolah Ora et Labora")
st.markdown("Tanya apapun tentang sekolah!")

# CEK API KEY HUGGING FACE
if "HF_TOKEN" not in st.secrets:
    st.error("❌ API Token Hugging Face belum disetting!")
    with st.expander("🔧 Cara Setting API Key"):
        st.code("""
1. Daftar gratis di https://huggingface.co/join
2. Buat token di https://huggingface.co/settings/tokens
3. Di Streamlit Cloud: Settings → Secrets
4. Tambahkan: HF_TOKEN = "hf_xxxxxxxxxxxxx"
        """)
    st.stop()

# SETUP API KEY
HF_TOKEN = st.secrets["HF_TOKEN"]

# CEK FOLDER DATA
DATA_PATH = Path("data")
if not DATA_PATH.exists():
    st.error("❌ Folder 'data' tidak ditemukan!")
    st.info("Buat folder 'data' dan masukkan file TXT tentang sekolah")
    st.stop()

# BACA SEMUA FILE TEKS
@st.cache_data
def load_all_texts():
    all_text = ""
    files = list(DATA_PATH.glob("*.txt")) + list(DATA_PATH.glob("*.md"))
    
    if not files:
        return None
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                all_text += f"\n\n--- {file.name} ---\n{content}\n"
        except Exception as e:
            st.warning(f"⚠️ Gagal baca {file.name}: {e}")
    
    return all_text

# LOAD DOKUMEN
with st.spinner("📚 Membaca file dokumen sekolah..."):
    school_data = load_all_texts()

if school_data is None:
    st.error("❌ Tidak ada file .txt di folder data!")
    st.stop()

st.success(f"✅ Berhasil memuat {len(school_data)} karakter pengetahuan sekolah!")

# FUNGSI PAKAI HUGGING FACE (GRATIS, PASTI JALAN)
def ask_huggingface(question, context):
    # API URL untuk model gratis
    API_URL = "https://api-inference.huggingface.co/models/indobenchmark/indobert-base-p1"
    
    # Batasi konteks
    max_context = 2000
    if len(context) > max_context:
        context = context[:max_context]
    
    prompt = f"""Answer based on school information below:

School Info:
{context}

Question: {question}

Answer:"""
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'Maaf, tidak bisa menjawab.')
            return str(result)
        else:
            return f"Error API: {response.status_code} - Coba lagi nanti."
    except Exception as e:
        return f"Error: {str(e)}"

# INTERFACE CHAT
st.markdown("---")
st.info("💡 Contoh pertanyaan: 'Apa saja ekstrakurikuler?', 'Siapa nama guru?', 'Kapan jadwal sekolah?'")

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
            answer = ask_huggingface(prompt, school_data)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
