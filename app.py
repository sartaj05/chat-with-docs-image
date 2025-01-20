import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
from fpdf import FPDF

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\91830\Desktop\sa\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GOOGLE_API_KEY is missing in environment variables. Please set it to proceed.")
    st.stop()

# ---------------- Helper Functions ---------------- #
def extract_text_from_pdfs(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDFs: {e}")
        return ""

def extract_text_from_images(image_docs):
    """Extracts text from uploaded image files using OCR."""
    text = ""
    try:
        for image_file in image_docs:
            image = Image.open(image_file)
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from images: {e}")
        return ""

def split_text_into_chunks(text):
    """Splits text into smaller, manageable chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []

def create_faiss_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created and saved locally! 🗂️")
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None

def load_faiss_vector_store():
    """Loads the FAISS vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS vector store: {e}")
        return None

def initialize_qa_chain():
    """Sets up the QA chain with a custom prompt for Google Generative AI."""
    prompt_template = """
    You are an AI that summarizes content. Given the provided context, write a concise summary in a way that's clear and easy to understand.
    Make sure to provide the main points, and avoid excessive details. Keep the summary short and focused.

    Context:\n {context}\n
    Summarize the content:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error initializing QA chain: {e}")
        return None

def summarize_documents(user_instruction="concise summary", topic=None):
    """Generates a summary for indexed documents filtered by topic."""
    vector_store = load_faiss_vector_store()
    if not vector_store:
        return ""
    # Search with the topic if provided
    query = user_instruction + (" " + topic if topic else "")
    docs = vector_store.similarity_search(query, k=3)
    
    prompt_template = f"""
    You are an AI that summarizes content. Based on the context provided, create a {user_instruction} summary.
    Context:\n {{context}}\n
    Summarize the content:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""

def answer_user_question(user_question, topic=None):
    """Handles user questions based on indexed content and topic filter."""
    vector_store = load_faiss_vector_store()
    if not vector_store:
        return ""
    # Search with the topic if provided
    query = user_question + (" " + topic if topic else "")
    docs = vector_store.similarity_search(query, k=3)
    chain = initialize_qa_chain()
    if not chain:
        return ""
    try:
        response = chain({"input_documents": docs}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return ""

import os

def generate_pdf_summary(summary):
    """Generates a downloadable PDF from the summary."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, txt=summary, align="L")

    # Change the path to a valid directory on your local machine
    pdf_file_path = os.path.join(os.getcwd(), "summary.pdf")  # Saves the PDF in the current working directory
    pdf.output(pdf_file_path)
    return pdf_file_path


def download_pdf(summary):
    """Provides the generated PDF for download."""
    pdf_file_path = generate_pdf_summary(summary)
    st.download_button(
        label="Download PDF Summary",
        data=open(pdf_file_path, "rb").read(),
        file_name="summary.pdf",
        mime="application/pdf"
    )


def main():
    st.set_page_config(page_title="Document and Image Assistant", page_icon="📄", layout="wide")
    st.header("📄 Chat with Documents and Images using Gemini AI 💬")

    # Sidebar for file uploads
    with st.sidebar:
        st.title("📂 Upload Your Files")
        st.write("Upload PDFs or images to extract and process text.")
        uploaded_files = st.file_uploader(
            "Select Files", accept_multiple_files=True, type=["pdf", "jpg", "jpeg", "png"]
        )
        process_button = st.button("📊 Process Files")
    
    # Placeholder for dynamic content
    placeholder = st.empty()

    # Processing files
    if uploaded_files:
        # Display file names in a multi-select dropdown
        file_names = [file.name for file in uploaded_files]
        selected_files = st.multiselect(
            "Select files to process", options=file_names, default=file_names
        )

        if process_button:
            # Filter selected files
            selected_uploaded_files = [file for file in uploaded_files if file.name in selected_files]

            with st.spinner("Processing your documents..."):
                pdf_files = [f for f in selected_uploaded_files if f.type == "application/pdf"]
                image_files = [f for f in selected_uploaded_files if f.type.startswith("image/")]

                pdf_text = extract_text_from_pdfs(pdf_files)
                image_text = extract_text_from_images(image_files)

                combined_text = pdf_text + "\n" + image_text
                if combined_text.strip():
                    text_chunks = split_text_into_chunks(combined_text)
                    create_faiss_vector_store(text_chunks)
                else:
                    st.error("No extractable text found in the selected files.")

    # Summarization
    with st.expander("📜 Summarize Content", expanded=True):
        user_topic = st.text_input("Enter the topic for summary (optional):")
        user_instruction = st.text_input("Enter summary instruction (e.g., 'short summary')")
        if user_instruction and st.button("✍️ Summarize"):
            with st.spinner("Generating summary..."):
                summary = summarize_documents(user_instruction, topic=user_topic)
                if summary:
                    st.success("Summary:")
                    st.write(summary)
                    download_pdf(summary)  # Provide download option if summary is generated

    # Question Answering
    with st.expander("💬 Ask Questions", expanded=True):
        user_topic_for_question = st.text_input("Enter the topic for your question (optional):")
        user_question = st.text_input("Type your question about the document:")
        if user_question and st.button("🔍 Get Answer"):
            with st.spinner("Fetching answer..."):
                answer = answer_user_question(user_question, topic=user_topic_for_question)
                if answer:
                    st.success("Answer:")
                    st.write(answer)

if __name__ == "__main__":
    main()
