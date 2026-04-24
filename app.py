import streamlit as st
import os
import google.generativeai as genai
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

# SETUP GEMINI - PAKAI MODEL YANG PASTI TERSEDIA
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Coba beberapa model
try:
    # Model yang paling stabil dan gratis
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    try:
        model = genai.GenerativeModel('models/gemini-1.0-pro')
    except:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

# CEK FOLDER DATA
DATA_PATH = Path("data")
if not DATA_PATH.exists():
    st.error("❌ Folder 'data' tidak ditemukan!")
    st.info("Buat folder 'data' dan masukkan file TXT tentang sekolah")
    st.stop()

# BACA SEMUA FILE TEKS (TANPA PREVIEW)
@st.cache_data
def load_all_texts():
    all_text = ""
    files = list(DATA_PATH.glob("*.txt")) + list(DATA_PATH.glob("*.md"))
    
    if not files:
        st.error("❌ Tidak ada file .txt di folder data!")
        return None
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                all_text += f"\n\n[Dokumen: {file.name}]\n{content}\n"
        except Exception as e:
            st.warning(f"⚠️ Gagal baca {file.name}: {e}")
    
    return all_text

# LOAD DOKUMEN
with st.spinner("📚 Membaca file dokumen sekolah..."):
    school_data = load_all_texts()

if school_data is None:
    st.stop()

st.success(f"✅ Berhasil memuat pengetahuan sekolah! ({len(school_data)} karakter)")

# FUNGSI CHAT DENGAN KONTEKS
def ask_question(question, context):
    # Batasi context agar tidak terlalu panjang
    max_context = 5000  # Ambil 5000 karakter pertama saja
    if len(context) > max_context:
        context = context[:max_context] + "..."
    
    prompt = f"""Anda adalah asisten AI untuk Sekolah Ora et Labora.
Jawab pertanyaan berikut berdasarkan INFORMASI SEKOLAH yang diberikan.
Jika tidak tahu, katakan "Maaf, informasi tentang itu tidak tersedia di data sekolah."
Jawab dalam bahasa Indonesia yang ramah dan jelas.

INFORMASI SEKOLAH:
{context}

PERTANYAAN: {question}

JAWABAN:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# INTERFACE CHAT
st.markdown("---")
st.info("💡 Saran pertanyaan: 'Apa saja ekstrakurikuler yang tersedia?' atau 'Siapa guru coding?', 'Kapan jadwal pramuka?'")

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
        with st.spinner("🤔 Mencari jawaban dari data sekolah..."):
            answer = ask_question(prompt, school_data)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
