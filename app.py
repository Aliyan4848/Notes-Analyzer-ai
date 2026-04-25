import gradio as gr
import os
from groq import Groq

# ✅ Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# 🔑 API KEY (from HF Secrets)
# =========================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Add it in Hugging Face Secrets.")

client = Groq(api_key=api_key)

# =========================
# 🌍 GLOBAL DB
# =========================
vector_db = None

# =========================
# 📄 PROCESS FILES
# =========================
def process_files(files):
    global vector_db

    all_docs = []

    for file in files:
        loader = PyPDFLoader(file.name)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(split_docs, embeddings)

    return f"✅ {len(files)} file(s) processed!"

# =========================
# 🤖 CHAT FUNCTION (FIXED)
# =========================
def chat_with_notes(message, history):
    global vector_db

    if history is None:
        history = []

    if vector_db is None:
        history.append({
            "role": "assistant",
            "content": "⚠️ Please upload notes first."
        })
        return history

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(message)

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = "\n\n--- SOURCES ---\n"
    for i, doc in enumerate(docs):
        sources += f"\n[{i+1}] {doc.page_content[:200]}..."

    prompt = f"""
You are an AI tutor.

Rules:
- Answer ONLY from notes
- If not found, say: Not found in notes
- Keep answer simple

NOTES:
{context}

QUESTION:
{message}

ANSWER:
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    # ✅ New Gradio message format
 history.append([message, answer + sources])
return history
    

# =========================
# 🎨 UI (FIXED FOR GRADIO 6)
# =========================
with gr.Blocks() as app:
    gr.Markdown("## 📚 AI Notes Assistant (RAG + Groq)")
    gr.Markdown("Upload PDFs and chat with your notes")

    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Upload Notes")
        upload_btn = gr.Button("📤 Process Notes")

    status = gr.Textbox(label="Status")

    upload_btn.click(process_files, inputs=file_input, outputs=status)

    chatbot = gr.Chatbot(height=400)
    
    msg = gr.Textbox(placeholder="Ask your question...")
    send_btn = gr.Button("Send")

    send_btn.click(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot)

    clear_btn = gr.Button("🗑️ Clear Chat")
    clear_btn.click(lambda: [], None, chatbot)

# ✅ Theme moved to launch (Gradio 6 fix)
app.launch(theme=gr.themes.Soft())
