import os
import gradio as gr
import PyPDF2
from docx import Document
from groq import Groq
from datetime import datetime
from typing import List, Tuple, Optional

# Initialize Groq client
# Note: Ensure GROQ_API_KEY is set in your environment variables
client = Groq()

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
        chunks = [" ".join(words[i:i + 500]) for i in range(0, len(words), 450)]
        self.documents[doc_name] = {'chunks': chunks}

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "No documents uploaded. Please upload educational notes first."
        
        # Simple retrieval: Combine first chunks for context
        all_context = ""
        for doc in self.documents.values():
            all_context += "\n".join(doc['chunks'][:3])

        try:
            # Fixed Groq API call syntax
            chat_completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "Answer based on the provided notes context."},
                    {"role": "user", "content": f"Context: {all_context[:5000]}\n\nQuestion: {question}"}
                ],
                max_tokens=1024
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# --- Gradio UI Logic ---

def handle_upload(files, state_processor):
    if not files:
        return "No files selected.", state_processor
    
    dp = DocumentProcessor()
    for file in files:
        text, _ = dp.process_file(file.name)
        state_processor.add_document(os.path.basename(file.name), text)
    
    return f"Successfully uploaded {len(files)} file(s).", state_processor

def handle_query(question, state_processor):
    return state_processor.answer_question(question)

with gr.Blocks() as demo:
    # This keeps the RAGProcessor alive for the duration of the user's session
    session_rag = gr.State(RAGProcessor())
    
    gr.Markdown("# 🎓 Educational RAG AI")
    
    with gr.Tab("1. Upload Notes"):
        file_input = gr.File(label="Upload PDF, DOCX, or TXT", file_count="multiple")
        upload_btn = gr.Button("Process Documents")
        status_out = gr.Textbox(label="Status")
        
        upload_btn.click(
            handle_upload, 
            inputs=[file_input, session_rag], 
            outputs=[status_out, session_rag]
        )
        
    with gr.Tab("2. Q&A"):
        ques_input = gr.Textbox(label="Your Question")
        ans_output = gr.Textbox(label="AI Response")
        ask_btn = gr.Button("Ask")
        
        ask_btn.click(
            handle_query, 
            inputs=[ques_input, session_rag], 
            outputs=ans_output
        )

if __name__ == "__main__":
    demo.launch()
