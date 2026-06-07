import os
import platform
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_bytes
from fpdf import FPDF

import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is missing. Add it inside your .env file.")
    st.stop()

genai.configure(api_key=api_key)

if platform.system() == "Windows":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


FAISS_INDEX_PATH = "faiss_index"


def extract_text_from_pdfs(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        try:
            pdf.seek(0)
            reader = PdfReader(pdf)

            normal_text = ""
            for page in reader.pages:
                normal_text += page.extract_text() or ""

            if normal_text.strip():
                text += f"\n\n--- PDF: {pdf.name} ---\n"
                text += normal_text
            else:
                pdf.seek(0)
                images = convert_from_bytes(pdf.read())

                scanned_text = ""
                for image in images:
                    scanned_text += pytesseract.image_to_string(image) + "\n"

                text += f"\n\n--- Scanned PDF OCR: {pdf.name} ---\n"
                text += scanned_text

        except Exception as e:
            st.error(f"Error extracting PDF text from {pdf.name}: {e}")

    return text


def extract_text_from_images(image_docs):
    text = ""

    for image_file in image_docs:
        try:
            image = Image.open(image_file)
            image_text = pytesseract.image_to_string(image)

            text += f"\n\n--- Image OCR: {image_file.name} ---\n"
            text += image_text

        except Exception as e:
            st.error(f"Error extracting image text from {image_file.name}: {e}")

    return text


def extract_text_from_docx(docx_docs):
    text = ""

    for docx_file in docx_docs:
        try:
            document = Document(docx_file)

            docx_text = ""
            for para in document.paragraphs:
                if para.text.strip():
                    docx_text += para.text + "\n"

            for table in document.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    docx_text += " | ".join(row_text) + "\n"

            text += f"\n\n--- DOCX: {docx_file.name} ---\n"
            text += docx_text

        except Exception as e:
            st.error(f"Error extracting DOCX text from {docx_file.name}: {e}")

    return text


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=800
    )
    return splitter.split_text(text)


def create_faiss_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

        return vector_store

    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None


def load_faiss_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )

        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:
        st.error(f"Error loading FAISS vector store: {e}")
        return None


def initialize_qa_chain():
    prompt_template = """
You are a helpful document assistant.

Answer the user's question using only the context below.
If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def summarize_documents(user_instruction, topic=None, summary_length="short"):
    vector_store = load_faiss_vector_store()

    if not vector_store:
        return ""

    query = user_instruction
    if topic:
        query += " " + topic

    docs = vector_store.similarity_search(query, k=5)

    length_mapping = {
        "short": "Write a short summary with only the most important points.",
        "medium": "Write a medium-length summary with key points and useful details.",
        "long": "Write a detailed summary with important explanations and structure."
    }

    prompt_template = """
You are a document summarization assistant.

Instruction:
{instruction}

Summary Length:
{summary_length_instruction}

Context:
{context}

Write the summary:
"""

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "instruction", "summary_length_instruction"]
        )

        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        response = chain(
            {
                "input_documents": docs,
                "instruction": user_instruction,
                "summary_length_instruction": length_mapping.get(
                    summary_length,
                    length_mapping["short"]
                )
            },
            return_only_outputs=True
        )

        return response["output_text"]

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""


def answer_user_question(user_question, topic=None):
    vector_store = load_faiss_vector_store()

    if not vector_store:
        return ""

    query = user_question
    if topic:
        query += " " + topic

    docs = vector_store.similarity_search(query, k=5)

    try:
        chain = initialize_qa_chain()

        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )

        return response["output_text"]

    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return ""


def generate_pdf_summary(summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    safe_summary = summary.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 10, txt=safe_summary, align="L")

    return pdf.output(dest="S").encode("latin1")


def download_pdf(summary):
    pdf_data = generate_pdf_summary(summary)

    st.download_button(
        label="Download PDF Summary",
        data=pdf_data,
        file_name="summary.pdf",
        mime="application/pdf"
    )


def reset_index():
    if os.path.exists(FAISS_INDEX_PATH):
        for file_name in os.listdir(FAISS_INDEX_PATH):
            os.remove(os.path.join(FAISS_INDEX_PATH, file_name))
        os.rmdir(FAISS_INDEX_PATH)


def main():
    st.set_page_config(
        page_title="Document Summary Assistant",
        page_icon="📄",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .main {
            color: #222222;
        }
        .stButton button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }
        .stTextInput input {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📄 Chat with PDF, Scanned PDF, Images and DOCX using Gemini")
    st.caption("Upload files, extract text, summarize, and ask questions.")

    with st.sidebar:
        st.header("📂 Upload Files")

        uploaded_files = st.file_uploader(
            "Select PDF, scanned PDF, image, or DOCX files",
            accept_multiple_files=True,
            type=["pdf", "jpg", "jpeg", "png", "docx"]
        )

        process_button = st.button("📊 Process Files")

        if st.button("🧹 Reset Saved Index"):
            reset_index()
            st.success("Saved FAISS index removed.")

    if uploaded_files:
        file_names = [file.name for file in uploaded_files]

        selected_files = st.multiselect(
            "Select files to process",
            options=file_names,
            default=file_names
        )

        if process_button:
            selected_uploaded_files = [
                file for file in uploaded_files
                if file.name in selected_files
            ]

            with st.spinner("Extracting text and creating vector index..."):
                pdf_files = [
                    f for f in selected_uploaded_files
                    if f.type == "application/pdf" or f.name.lower().endswith(".pdf")
                ]

                image_files = [
                    f for f in selected_uploaded_files
                    if f.type.startswith("image/")
                ]

                docx_files = [
                    f for f in selected_uploaded_files
                    if f.name.lower().endswith(".docx")
                ]

                pdf_text = extract_text_from_pdfs(pdf_files)
                image_text = extract_text_from_images(image_files)
                docx_text = extract_text_from_docx(docx_files)

                combined_text = pdf_text + "\n" + image_text + "\n" + docx_text

                if combined_text.strip():
                    text_chunks = split_text_into_chunks(combined_text)

                    vector_store = create_faiss_vector_store(text_chunks)

                    if vector_store:
                        st.success(
                            "Files processed successfully. You can now summarize or ask questions."
                        )
                    else:
                        st.error(
                            "Text was extracted, but FAISS creation failed. Check your Google API key."
                        )
                else:
                    st.error("No extractable text found.")

    st.divider()

    with st.expander("📜 Summarize Content", expanded=True):
        user_topic = st.text_input("Enter topic for summary optional")
        user_instruction = st.text_input(
            "Enter summary instruction",
            value="Create a clear summary"
        )
        summary_length = st.selectbox(
            "Select summary length",
            ["short", "medium", "long"]
        )

        if st.button("✍️ Summarize"):
            if not os.path.exists(FAISS_INDEX_PATH):
                st.warning("Please process files first.")
            elif not user_instruction.strip():
                st.warning("Please enter summary instruction.")
            else:
                with st.spinner("Generating summary..."):
                    summary = summarize_documents(
                        user_instruction=user_instruction,
                        topic=user_topic,
                        summary_length=summary_length
                    )

                    if summary:
                        st.success("Summary")
                        st.write(summary)
                        download_pdf(summary)

    with st.expander("💬 Ask Questions", expanded=True):
        user_topic_for_question = st.text_input(
            "Enter topic for question optional"
        )
        user_question = st.text_input(
            "Type your question about the uploaded document"
        )

        if st.button("🔍 Get Answer"):
            if not os.path.exists(FAISS_INDEX_PATH):
                st.warning("Please process files first.")
            elif not user_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding answer..."):
                    answer = answer_user_question(
                        user_question=user_question,
                        topic=user_topic_for_question
                    )

                    if answer:
                        st.success("Answer")
                        st.write(answer)


if __name__ == "__main__":
    main()
