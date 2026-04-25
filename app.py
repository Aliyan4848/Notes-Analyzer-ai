import gradio as gr
import os
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# =========================
# 🔑 API KEY
# =========================
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# =========================
# 🌍 GLOBAL STORAGE
# =========================
vector_db = None

# =========================
# 📄 PROCESS MULTIPLE PDFs
# =========================
def process_files(files):
    global vector_db

    all_docs = []

    for file in files:
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        all_docs.extend(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(docs, embeddings)

    return f"✅ {len(files)} file(s) processed successfully!"

# =========================
# 🤖 RAG CHAT FUNCTION
# =========================
def chat_with_notes(message, history):
    global vector_db

    if vector_db is None:
        return history + [[message, "⚠️ Please upload notes first."]]

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(message)

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = "\n\n--- SOURCES ---\n"
    for i, doc in enumerate(docs):
        sources += f"\n[{i+1}] {doc.page_content[:200]}..."

    prompt = f"""
You are an AI tutor helping a student.

STRICT RULES:
- Answer ONLY using the notes
- If answer is missing, say: "Not found in notes"
- Keep answer simple and clear

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

    final_answer = answer + sources

    history.append([message, final_answer])
    return history

# =========================
# 🎨 MODERN UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 📚 AI Notes Assistant (Advanced RAG)")
    gr.Markdown("Upload notes and chat like ChatGPT!")

    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Upload Notes (PDF)")
        upload_btn = gr.Button("📤 Process Notes")

    status = gr.Textbox(label="Status")

    upload_btn.click(process_files, inputs=file_input, outputs=status)

    chatbot = gr.Chatbot(height=400)

    msg = gr.Textbox(placeholder="Ask anything from your notes...")
    send_btn = gr.Button("Send")

    send_btn.click(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot)

app.launch()
