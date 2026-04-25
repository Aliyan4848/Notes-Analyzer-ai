import gradio as gr
import os
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# 🔑 ENV VARIABLES
# =========================
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

if not model_name:
    raise ValueError("GROQ_MODEL not found in environment variables")

client = Groq(api_key=api_key)

# =========================
# 🌍 VECTOR DB
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

    return f"✅ {len(files)} file(s) processed successfully!"

# =========================
# 🤖 CHAT FUNCTION (RAG + NORMAL MODE)
# =========================
def chat_with_notes(message, history):
    global vector_db

    # 🟢 If no documents → normal AI chat
    if vector_db is None:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=model_name,
            temperature=0.7,
        )

        answer = response.choices[0].message.content
        return history + [[message, answer]]

    # 🔵 RAG MODE
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(message)

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = "\n\n--- SOURCES ---\n"
    for i, doc in enumerate(docs):
        sources += f"\n[{i+1}] {doc.page_content[:200]}..."

    prompt = f"""
You are an AI tutor.

Rules:
- Use ONLY the given notes
- If answer not found, say "Not found in notes"
- Keep answer simple

NOTES:
{context}

QUESTION:
{message}

ANSWER:
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    return history + [[message, answer + sources]]

# =========================
# 🎨 UI (GRADIO SAFE)
# =========================
with gr.Blocks() as app:
    gr.Markdown("## 📚 AI Notes Assistant (RAG + Groq)")
    gr.Markdown("Upload notes OR chat normally")

    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Upload PDFs")
        upload_btn = gr.Button("Process Notes")

    status = gr.Textbox(label="Status")

    upload_btn.click(process_files, inputs=file_input, outputs=status)

    chatbot = gr.Chatbot(height=400)

    msg = gr.Textbox(placeholder="Ask something...")
    send = gr.Button("Send")

    send.click(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot)

    clear = gr.Button("Clear Chat")
    clear.click(lambda: [], None, chatbot)

app.launch()
