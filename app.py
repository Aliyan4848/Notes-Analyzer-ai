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
# Get API key from secrets/env
api_key = os.getenv("GROQ_API_KEY")

# Hardcoding the model name to prevent "Variable Not Found" errors
model_name = "llama3-8b-8192" 

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables/secrets")

# Initialize the Groq client GLOBALLY so all functions can see it
client = Groq(api_key=api_key)

# =========================
# 🌍 VECTOR DB STORAGE
# =========================
vector_db = None

# =========================
# 📄 PROCESS FILES FUNCTION
# =========================
def process_files(files):
    global vector_db

    if not files:
        return "⚠️ No files uploaded."

    all_docs = []
    try:
        for file in files:
            loader = PyPDFLoader(file.name)
            docs = loader.load()
            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(all_docs)

        # Using a lightweight embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_documents(split_docs, embeddings)

        return f"✅ {len(files)} file(s) processed! You can now ask questions about them."
    except Exception as e:
        return f"❌ Error processing files: {str(e)}"

# =========================
# 🤖 CHAT FUNCTION (RAG + NORMAL)
# =========================
def chat_with_notes(message, history):
    global vector_db
    
    try:
        # 🟢 MODE 1: Normal AI Chat (No documents uploaded)
        if vector_db is None:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=model_name,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return history

        # 🔵 MODE 2: RAG Mode (Search through notes)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(message)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Prepare source snippets for transparency
        sources = "\n\n--- SOURCES ---\n"
        for i, doc in enumerate(docs):
            sources += f"\n[{i+1}] {doc.page_content[:200]}..."

        prompt = f"""
You are an AI tutor. 
Rules:
- Use ONLY the provided notes to answer.
- If the answer isn't in the notes, say "I couldn't find that in your notes."
- Keep it simple and helpful.

NOTES:
{context}

QUESTION:
{message}
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        
        # Add to history and return
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer + sources})
        return history

    except Exception as e:
        # If something breaks, show the error in the chat
        error_msg = f"❌ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history

# =========================
# 🎨 UI LAYOUT (GRADIO)
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📚 AI Notes Assistant")
    gr.Markdown("Upload your PDF notes and chat with them using RAG + Groq.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", label="Upload PDF Notes")
            upload_btn = gr.Button("🚀 Process & Index Notes", variant="primary")
            status = gr.Textbox(label="System Status", interactive=False)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Study Chat", type="messages", height=500)
            msg = gr.Textbox(placeholder="Ask a question about your notes...", label="Your Question")
            
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")

    # --- Button Logic ---
    upload_btn.click(process_files, inputs=file_input, outputs=status)

    # Sending messages
    send.click(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)
    msg.submit(chat_with_notes, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)

    # Clearing history
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    app.launch()
