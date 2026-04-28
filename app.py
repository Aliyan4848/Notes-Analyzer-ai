import os
import gradio as gr
import PyPDF2
from docx import Document
from groq import Groq
from typing import List, Tuple, Optional

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

class DocumentProcessor:
    @staticmethod
    def process_file(file_path: str) -> Tuple[str, str]:
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        try:
            if ext == '.pdf':
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    text = "".join(page.extract_text() or "" for page in pdf.pages)
            elif ext == '.docx':
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return f"Unsupported format: {ext}", ext
        except Exception as e:
            return f"Error: {str(e)}", ext
        return text, ext

class RAGProcessor:
    def __init__(self):
        self.documents = {}

    def add_document(self, doc_name: str, content: str):
        words = content.split()
        # Create chunks of 500 words with overlap
        chunks = [" ".join(words[i:i + 500]) for i in range(0, len(words), 450)]
        self.documents[doc_name] = {'chunks': chunks}

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "⚠️ No documents found. Please upload your notes in the 'Upload' tab first."
                
        all_context = ""
        for doc in self.documents.values():
            all_context += "\n".join(doc['chunks'][:5]) 

        try:
            chat_completion = client.chat.completions.create(
             model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant. Answer based ONLY on the provided context."},
                    {"role": "user", "content": f"Context: {all_context[:6000]}\n\nQuestion: {question}"}
                ],
                max_tokens=1024
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --- UI Functions ---
def handle_upload(files, state_processor):
    if not files:
        return "No files selected.", state_processor
    
    # Initialize processor if state is None
    if state_processor is None:
        state_processor = RAGProcessor()
        
    dp = DocumentProcessor()
    for file in files:
        text, _ = dp.process_file(file.name)
        state_processor.add_document(os.path.basename(file.name), text)
        
    return f"✅ Successfully processed {len(files)} file(s).", state_processor

def handle_query(question, state_processor):
    if state_processor is None:
        return "⚠️ Please upload files first."
    return state_processor.answer_question(question)

# --- Gradio Interface ---
# FIX: Removed theme from Blocks constructor
with gr.Blocks() as demo:
    session_rag = gr.State() # Starts as None
    
    gr.Markdown("# 🎓 AI Study Assistant (RAG)")
    
    with gr.Tabs():
        with gr.TabItem("1. Upload Notes"):
            file_input = gr.File(label="Upload PDFs, Docs, or TXT", file_count="multiple")
            upload_btn = gr.Button("Process Documents", variant="primary")
            status_out = gr.Textbox(label="Status")
            
            upload_btn.click(
                handle_upload, 
                inputs=[file_input, session_rag], 
                outputs=[status_out, session_rag]
            )
            
        with gr.TabItem("2. Ask Questions"):
            ques_input = gr.Textbox(label="Enter your question")
            ans_output = gr.Markdown()
            ask_btn = gr.Button("Search Notes", variant="primary")
            
            ask_btn.click(
                handle_query, 
                inputs=[ques_input, session_rag], 
                outputs=ans_output
            )

if __name__ == "__main__":
    # FIX: Moved theme to launch() and disabled ssr_mode for Python 3.13 stability
    demo.queue().launch(
        theme=gr.themes.Soft(),
        ssr_mode=False
    )
