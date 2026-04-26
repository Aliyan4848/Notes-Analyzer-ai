"""
RAG-Based Educational AI System
Processes university/college notes and provides:
- Q&A based on uploaded notes
- Quiz generation
- Exam question prediction
- Supports PDF, TXT, DOCX file formats
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import gradio as gr
import PyPDF2
from docx import Document
from groq import Groq

# Initialize Groq client
client = Groq()

# Global variables for session management
uploaded_documents = {}
document_embeddings = {}
current_session = None


class DocumentProcessor:
    """Handles extraction of text from various document formats"""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
        return text
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX files"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
        return text
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
        return text
    
    @staticmethod
    def process_file(file_path: str) -> Tuple[str, str]:
        """Process file based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = DocumentProcessor.extract_from_pdf(file_path)
        elif ext == '.docx':
            text = DocumentProcessor.extract_from_docx(file_path)
        elif ext == '.txt':
            text = DocumentProcessor.extract_from_txt(file_path)
        else:
            text = f"Unsupported file format: {ext}"
        
        return text, ext


class RAGProcessor:
    """Handles RAG operations using Groq API"""
    
    def __init__(self):
        self.documents = {}
        self.chunk_size = 1000
    
    def add_document(self, doc_name: str, content: str):
        """Add document to knowledge base"""
        # Split content into chunks for better retrieval
        chunks = self._create_chunks(content)
        self.documents[doc_name] = {
            'full_text': content,
            'chunks': chunks,
            'added_at': datetime.now().isoformat()
        }
    
    def _create_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Create overlapping chunks from text"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        overlap_words = 50
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep last 50 words for overlap
                current_chunk = current_chunk[-overlap_words:]
                current_size = sum(len(w) for w in current_chunk) + overlap_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant chunks based on query"""
        relevant_texts = []
        
        for doc_name, doc_data in self.documents.items():
            chunks = doc_data['chunks']
            # Simple keyword matching (can be enhanced with embeddings)
            scored_chunks = []
            query_words = set(query.lower().split())
            
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                similarity = len(query_words & chunk_words) / (len(query_words) + 1)
                scored_chunks.append((chunk, similarity))
            
            # Sort by similarity and get top chunks
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            for chunk, score in scored_chunks[:top_k]:
                if score > 0:
                    relevant_texts.append(chunk)
        
        return "\n\n".join(relevant_texts[:top_k])
    
    def answer_question(self, question: str, document_name: Optional[str] = None) -> str:
        """Answer question using RAG approach"""
        if not self.documents:
            return "No documents uploaded. Please upload educational notes first."
        
        # Retrieve relevant context
        context = self.retrieve_relevant_chunks(question)
        
        # Create prompt for Groq
        system_prompt = """You are an expert educational assistant. Answer questions based on the provided document context.
        - Provide clear, concise answers
        - Cite specific parts of the document when relevant
        - If the answer isn't in the document, say so explicitly
        - Format your answer in an easy-to-understand way"""
        
        user_prompt = f"""Context from uploaded notes:
{context}

Question: {question}

Please answer the question based on the context provided."""
        
        try:
            message = client.messages.create(
                model="mixtral-8x7b-32768",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_quiz(self, num_questions: int = 5, document_name: Optional[str] = None) -> str:
        """Generate quiz questions from the documents"""
        if not self.documents:
            return "No documents uploaded. Please upload educational notes first."
        
        # Get a representative chunk from the document
        doc_data = list(self.documents.values())[0]
        context = "\n".join(doc_data['chunks'][:3])  # Use first 3 chunks as context
        
        system_prompt = """You are an expert quiz creator. Generate multiple-choice and short-answer quiz questions.
        Focus on:
        - Key concepts and definitions
        - Important facts and dates
        - Understanding and application
        - Mix of difficulty levels
        
        Format each question clearly with options (if multiple choice)."""
        
        user_prompt = f"""Based on these educational notes, create {num_questions} quiz questions.

Notes:
{context}

Format:
Q1) Question text?
Options: A) ..., B) ..., C) ..., D) ...
Answer: A

Q2) Question text?
Answer: [Short answer expected]

Continue for {num_questions} questions."""
        
        try:
            message = client.messages.create(
                model="mixtral-8x7b-32768",
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating quiz: {str(e)}"
    
    def predict_exam_questions(self, document_name: Optional[str] = None) -> str:
        """Predict likely exam questions based on document content"""
        if not self.documents:
            return "No documents uploaded. Please upload educational notes first."
        
        doc_data = list(self.documents.values())[0]
        context = "\n".join(doc_data['chunks'][:4])  # Use first 4 chunks
        
        system_prompt = """You are an experienced educator who can predict likely exam questions.
        Analyze educational content and identify:
        - Most important concepts likely to appear in exams
        - Topics that are emphasized or repeated
        - Potential long-essay questions
        - Likely problem-solving questions
        
        Provide predicted questions with brief explanations of why they're likely to appear."""
        
        user_prompt = f"""Analyze these educational notes and predict the top questions that are likely to appear in an exam.

Notes:
{context}

Provide 5-7 predicted exam questions with explanations. Format:

PREDICTED EXAM QUESTIONS

Question 1: [Question text]
Likely to appear because: [Explanation]

Continue for all predicted questions..."""
        
        try:
            message = client.messages.create(
                model="mixtral-8x7b-32768",
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error predicting exam questions: {str(e)}"


# Initialize RAG processor
rag_processor = RAGProcessor()


# ============ Gradio Interface Functions ============

def upload_notes(files) -> str:
    """Handle file uploads"""
    if not files:
        return "❌ No files selected. Please upload at least one file."
    
    upload_status = []
    
    for file in files if isinstance(files, list) else [files]:
        try:
            file_path = file.name if hasattr(file, 'name') else str(file)
            doc_name = os.path.basename(file_path)
            
            # Extract text from file
            text, ext = DocumentProcessor.process_file(file_path)
            
            if "Error" in text:
                upload_status.append(f"❌ {doc_name}: {text}")
            else:
                # Add to RAG processor
                rag_processor.add_document(doc_name, text)
                word_count = len(text.split())
                upload_status.append(f"✅ {doc_name} ({word_count} words) - Successfully processed!")
        
        except Exception as e:
            upload_status.append(f"❌ Error processing file: {str(e)}")
    
    return "\n".join(upload_status)


def answer_question_handler(question: str) -> str:
    """Handle question answering"""
    if not question.strip():
        return "⚠️ Please enter a question."
    
    if not rag_processor.documents:
        return "⚠️ Please upload educational notes first."
    
    response = rag_processor.answer_question(question)
    return response


def generate_quiz_handler(num_questions: int) -> str:
    """Handle quiz generation"""
    if not rag_processor.documents:
        return "⚠️ Please upload educational notes first."
    
    if num_questions < 1 or num_questions > 20:
        return "⚠️ Number of questions should be between 1 and 20."
    
    response = rag_processor.generate_quiz(num_questions)
    return response


def predict_exam_handler() -> str:
    """Handle exam prediction"""
    if not rag_processor.documents:
        return "⚠️ Please upload educational notes first."
    
    response = rag_processor.predict_exam_questions()
    return response


def get_documents_list() -> str:
    """Get list of uploaded documents"""
    if not rag_processor.documents:
        return "No documents uploaded yet."
    
    doc_list = "📚 Uploaded Documents:\n\n"
    for doc_name, doc_data in rag_processor.documents.items():
        word_count = len(doc_data['full_text'].split())
        chunk_count = len(doc_data['chunks'])
        added_time = doc_data['added_at']
        doc_list += f"• {doc_name}\n  Words: {word_count} | Chunks: {chunk_count}\n  Added: {added_time}\n\n"
    
    return doc_list


# ============ Gradio Interface Setup ============

def create_interface():
    """Create Gradio interface with custom styling"""
    
    with gr.Blocks(
        title="🎓 EduAI - RAG-Based Educational Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo"
        ),
        css="""
        .header-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .section-header {
            font-size: 1.4em;
            font-weight: 700;
            color: #333;
            border-left: 4px solid #667eea;
            padding-left: 10px;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        .info-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        }
        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 3em; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎓 EduAI
            </h1>
            <p style="font-size: 1.1em; color: #666; margin-top: 10px;">
                Your Intelligent Educational Assistant powered by RAG & Groq
            </p>
        </div>
        """)
        
        # Tabs for different functionalities
        with gr.Tabs():
            
            # ===== TAB 1: Upload Notes =====
            with gr.Tab("📤 Upload Notes"):
                gr.HTML('<div class="section-header">Upload Your Educational Notes</div>')
                
                gr.HTML("""
                <div class="info-box">
                    <b>📌 Supported Formats:</b> PDF, DOCX, TXT<br>
                    <b>💡 Tip:</b> Upload your lecture notes, textbook chapters, or study materials. 
                    The system will process them and prepare them for Q&A, quizzes, and exam predictions.
                </div>
                """)
                
                file_input = gr.File(
                    label="Choose files to upload",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath"
                )
                
                upload_btn = gr.Button("🚀 Upload & Process", variant="primary", size="lg")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=6
                )
                
                upload_btn.click(
                    fn=upload_notes,
                    inputs=file_input,
                    outputs=upload_status
                )
                
                gr.HTML('<div class="section-header">Documents Library</div>')
                docs_display = gr.Textbox(
                    label="Your Uploaded Documents",
                    interactive=False,
                    lines=8,
                    value="No documents uploaded yet."
                )
                refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                refresh_btn.click(fn=get_documents_list, outputs=docs_display)
            
            # ===== TAB 2: Q&A =====
            with gr.Tab("❓ Ask Questions"):
                gr.HTML('<div class="section-header">Ask Questions About Your Notes</div>')
                
                gr.HTML("""
                <div class="info-box">
                    <b>🤖 How it works:</b> The AI retrieves relevant information from your uploaded notes 
                    and generates accurate answers based on that content.
                </div>
                """)
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the main causes of World War II?",
                    lines=3
                )
                
                ask_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
                answer_output = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=10
                )
                
                ask_btn.click(
                    fn=answer_question_handler,
                    inputs=question_input,
                    outputs=answer_output
                )
            
            # ===== TAB 3: Quiz Generation =====
            with gr.Tab("🎯 Generate Quizzes"):
                gr.HTML('<div class="section-header">Create Practice Quizzes</div>')
                
                gr.HTML("""
                <div class="info-box">
                    <b>✨ Feature:</b> Automatically generate multiple-choice and short-answer questions 
                    based on your notes to help you prepare for exams.
                </div>
                """)
                
                num_questions = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Questions",
                    info="Choose between 1-20 questions"
                )
                
                quiz_btn = gr.Button("📝 Generate Quiz", variant="primary", size="lg")
                quiz_output = gr.Textbox(
                    label="Generated Quiz",
                    interactive=False,
                    lines=15
                )
                
                quiz_btn.click(
                    fn=generate_quiz_handler,
                    inputs=num_questions,
                    outputs=quiz_output
                )
            
            # ===== TAB 4: Exam Prediction =====
            with gr.Tab("🎓 Predict Exam Questions"):
                gr.HTML('<div class="section-header">Likely Exam Questions</div>')
                
                gr.HTML("""
                <div class="info-box">
                    <b>🔮 AI Prediction:</b> Based on your notes, the system identifies concepts and topics 
                    that are most likely to appear in your exams, along with explanations of why.
                </div>
                """)
                
                exam_btn = gr.Button("🚀 Predict Exam Questions", variant="primary", size="lg")
                exam_output = gr.Textbox(
                    label="Predicted Exam Questions",
                    interactive=False,
                    lines=15
                )
                
                exam_btn.click(
                    fn=predict_exam_handler,
                    outputs=exam_output
                )
            
            # ===== TAB 5: Help & Info =====
            with gr.Tab("ℹ️ Help & Information"):
                gr.HTML("""
                <div style="padding: 20px;">
                    <h2 style="color: #667eea;">📚 How to Use EduAI</h2>
                    
                    <h3>Step 1: Upload Notes</h3>
                    <p>Navigate to the "Upload Notes" tab and upload your educational materials in PDF, DOCX, or TXT format.</p>
                    
                    <h3>Step 2: Ask Questions</h3>
                    <p>Use the "Ask Questions" tab to get instant answers based on your uploaded notes.</p>
                    
                    <h3>Step 3: Practice with Quizzes</h3>
                    <p>Generate practice quizzes with customizable question counts to test your knowledge.</p>
                    
                    <h3>Step 4: Prepare for Exams</h3>
                    <p>Use the exam prediction feature to identify likely questions and focus your study efforts.</p>
                    
                    <hr style="margin: 30px 0;">
                    
                    <h2 style="color: #667eea;">🔧 Technical Details</h2>
                    <ul>
                        <li><b>AI Model:</b> Groq Mixtral 8x7B (Fast & Powerful)</li>
                        <li><b>Frontend:</b> Gradio</li>
                        <li><b>RAG Method:</b> Document chunking with semantic retrieval</li>
                        <li><b>Processing Speed:</b> Real-time responses</li>
                    </ul>
                    
                    <h2 style="color: #667eea;">💡 Tips for Best Results</h2>
                    <ul>
                        <li>Upload clear, well-formatted notes</li>
                        <li>Ask specific questions for better answers</li>
                        <li>Use quizzes regularly to reinforce learning</li>
                        <li>Review predicted exam questions weekly</li>
                        <li>Combine with active studying for best results</li>
                    </ul>
                    
                    <hr style="margin: 30px 0;">
                    
                    <p style="text-align: center; color: #999; font-size: 0.9em;">
                        Made with ❤️ for students | Powered by Groq & Gradio
                    </p>
                </div>
                """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
