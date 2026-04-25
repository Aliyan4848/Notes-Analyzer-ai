import gradio as gr
import os
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# 🔑 CONFIGURATION
# =========================
api_key = os.getenv("GROQ_API_KEY")
model_name = "llama3-8b-8192" 

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize client GLOBALLY
client = Groq(api_key=api_key)

# =========================
# 🌍 VECTOR DB STORAGE
# =========================
vector_db = None

# =========================
# 📄 PROCESS FILES
# =========================
def process_files(files):
    global vector_db
    if not files:
        return "⚠️ Please upload at least one PDF."

    all_docs = []
    try:
        for file in files:
            loader = PyPDFLoader(file.name)
            docs = loader.load()
            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        split_docs = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(split_docs, embeddings)

        return f"✅ {len(files)} file(s) indexed! Ask me anything."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# 🤖 CHAT FUNCTION
# =========================
def chat_with_notes(message, history):
    global vector_db
    
    try:
        # NORMAL CHAT (No docs)
        if vector_db is None:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=model_name,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            return history + [[message, answer]]

        # RAG CHAT
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in docs])

        sources = "\n\n--- SOURCES ---\n"
        for i, doc in enumerate(docs):
            sources += f"\n[{i+1}] {doc.page_content[:150]}..."

        prompt = f"Use these notes to answer: {context}\n\nQuestion: {message}"

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        return history + [[message, answer + sources]]

    except Exception as e:
        return history + [[message, f"❌ System Error: {str(e)}"]]

# =========================
# 🎨 UI (GRADIO 6.0 COMPATIBLE)
# =========================
# 1. Removed theme from here
with gr.Blocks() as app:
    gr.Markdown("# 📚 AI Notes Assistant (v6.0 Optimized)")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", label="Upload PDFs")
            upload_btn = gr.Button("Index Notes", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            # 2. Removed type="messages"
            chatbot = gr.Chatbot(label="Chat History", height=450)
            msg = gr.Textbox(placeholder="Type your question...")
            
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

    upload_btn.click(process_files, inputs=file_input, outputs=status)
    
    send.click(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)
    msg.submit(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)
    clear.click(lambda: [], None, chatbot)

# 3. Theme is passed HERE now
if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
