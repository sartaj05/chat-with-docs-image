import os
import platform

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


def apply_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(191, 219, 254, 0.58), transparent 32%),
                radial-gradient(circle at top right, rgba(221, 214, 254, 0.48), transparent 30%),
                linear-gradient(135deg, #f6f9ff 0%, #eef6ff 45%, #f8f7ff 100%);
            color: #172033;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fbfdff 0%, #eef6ff 100%);
            border-right: 1px solid #cfe0f6;
        }

        section[data-testid="stSidebar"] * {
            color: #172033 !important;
        }

        .main-title {
            font-size: 1.78rem;
            font-weight: 850;
            color: #163b88;
            margin-bottom: 0.28rem;
            letter-spacing: -0.02em;
        }

        .sub-title {
            font-size: 0.96rem;
            color: #52637a;
            line-height: 1.55;
        }

        .hero-card {
            padding: 1.25rem 1.7rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #fffefe 0%, #f8fbff 100%);
            border: 1px solid #d7e7fb;
            box-shadow: 0 12px 34px rgba(37, 99, 235, 0.09);
            margin-bottom: 1.6rem;
        }

        .metric-card {
            padding: 0.75rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #ffffff 0%, #f1f7ff 100%);
            border: 1px solid #dbeafe;
            text-align: center;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        }

        .metric-number {
            font-size: 1.35rem;
            font-weight: 850;
            color: #2563eb;
        }

        .metric-label {
            color: #52637a;
            font-size: 0.8rem;
            font-weight: 650;
        }

        .status-box {
            padding: 0.95rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #ecfdf5 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            color: #0f3f5f !important;
            font-weight: 750;
            margin-top: 1rem;
            line-height: 1.55;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }

        .file-card {
            padding: 0.75rem 0.9rem;
            border-radius: 13px;
            background: linear-gradient(135deg, #ffffff 0%, #f6faff 100%);
            border: 1px solid #dbeafe;
            margin-bottom: 0.5rem;
            box-shadow: 0 5px 14px rgba(15, 23, 42, 0.035);
        }

        .small-note {
            color: #64748b !important;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        div[data-testid="stExpander"] {
            background: linear-gradient(180deg, #fffefe 0%, #f8fbff 100%);
            border: 1px solid #d8e6fb;
            border-radius: 18px;
            box-shadow: 0 14px 32px rgba(30, 64, 175, 0.08);
            overflow: hidden;
        }

        div[data-testid="stExpander"] summary {
            background: linear-gradient(90deg, #f3f8ff 0%, #f8f7ff 100%);
            color: #163b88 !important;
            font-weight: 800;
            border-bottom: 1px solid #dbeafe;
            padding: 0.78rem 1rem !important;
        }

        div[data-testid="stExpander"] summary p {
            color: #163b88 !important;
            font-weight: 800 !important;
        }

        label {
            color: #334155 !important;
            font-weight: 650 !important;
        }

        .stTextInput input,
        .stTextArea textarea {
            background: #fbfdff !important;
            color: #172033 !important;
            border: 1px solid #bdd3ef !important;
            border-radius: 14px !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03) !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border: 1px solid #60a5fa !important;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.18) !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #7b8aa0 !important;
        }

        div[data-baseweb="select"] > div {
            background: #fbfdff !important;
            color: #172033 !important;
            border: 1px solid #bdd3ef !important;
            border-radius: 14px !important;
        }

        div[data-baseweb="select"] span {
            color: #172033 !important;
        }

        .stButton button {
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.65rem 1rem !important;
            font-weight: 750 !important;
            box-shadow: 0 10px 22px rgba(14, 165, 233, 0.22);
            transition: all 0.18s ease-in-out;
        }

        .stButton button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 100%) !important;
            color: white !important;
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(14, 165, 233, 0.28);
        }

        .stDownloadButton button {
            background: linear-gradient(135deg, #059669 0%, #34d399 100%) !important;
            color: white !important;
            border-radius: 14px !important;
            font-weight: 750 !important;
            border: none !important;
        }

        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #fbfdff 0%, #f2f8ff 100%);
            border: 1px dashed #93c5fd;
            border-radius: 18px;
            padding: 0.8rem;
        }

        [data-testid="stFileUploader"] section {
            background: #ffffff !important;
            border: 1px solid #dbeafe !important;
            border-radius: 15px !important;
        }

        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 750 !important;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
        }

        .stAlert {
            border-radius: 14px !important;
        }

        hr {
            border-color: #dbeafe;
            margin-top: 1rem;
            margin-bottom: 1.4rem;
        }

        @media (max-width: 768px) {
            .main-title {
                font-size: 1.35rem;
            }

            .hero-card {
                padding: 1rem;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
        label="⬇️ Download PDF Summary",
        data=pdf_data,
        file_name="summary.pdf",
        mime="application/pdf"
    )


def reset_index():
    if os.path.exists(FAISS_INDEX_PATH):
        for file_name in os.listdir(FAISS_INDEX_PATH):
            os.remove(os.path.join(FAISS_INDEX_PATH, file_name))
        os.rmdir(FAISS_INDEX_PATH)


def get_file_type_counts(uploaded_files):
    counts = {
        "PDF": 0,
        "Image": 0,
        "DOCX": 0
    }

    for file in uploaded_files:
        name = file.name.lower()
        if name.endswith(".pdf"):
            counts["PDF"] += 1
        elif name.endswith((".jpg", ".jpeg", ".png")):
            counts["Image"] += 1
        elif name.endswith(".docx"):
            counts["DOCX"] += 1

    return counts


def main():
    st.set_page_config(
        page_title="Document Summary Assistant",
        page_icon="📄",
        layout="wide"
    )

    apply_custom_css()

    st.markdown(
        """
        <div class="hero-card">
            <div class="main-title">📄 Document Summary Assistant</div>
            <div class="sub-title">
                Chat with PDF, scanned PDF, image, and DOCX files using Gemini AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("## 📂 Upload Center")
        st.markdown(
            '<p class="small-note">Supported: PDF, scanned PDF, JPG, PNG, DOCX.</p>',
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader(
            "Upload your files",
            accept_multiple_files=True,
            type=["pdf", "jpg", "jpeg", "png", "docx"]
        )

        if uploaded_files:
            counts = get_file_type_counts(uploaded_files)

            st.markdown("### 📊 Upload Summary")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-number">{counts["PDF"]}</div><div class="metric-label">PDF</div></div>',
                    unsafe_allow_html=True
                )
            with col_b:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-number">{counts["Image"]}</div><div class="metric-label">Image</div></div>',
                    unsafe_allow_html=True
                )
            with col_c:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-number">{counts["DOCX"]}</div><div class="metric-label">DOCX</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### 📁 Selected Files")
            for file in uploaded_files:
                size_kb = round(file.size / 1024, 2)
                st.markdown(
                    f"""
                    <div class="file-card">
                        <b>{file.name}</b><br>
                        <span class="small-note">{size_kb} KB</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        process_button = st.button("🚀 Process Files", use_container_width=True)

        if st.button("🧹 Reset Saved Index", use_container_width=True):
            reset_index()
            st.success("Saved FAISS index removed.")

        if os.path.exists(FAISS_INDEX_PATH):
            st.markdown(
                '<div class="status-box">✅ Vector index is ready. You can summarize or ask questions.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-box">ℹ️ No active vector index. Upload and process files first.</div>',
                unsafe_allow_html=True
            )

    if uploaded_files:
        file_names = [file.name for file in uploaded_files]

        st.markdown("### 🧾 File Selection")
        selected_files = st.multiselect(
            "Choose which files to process",
            options=file_names,
            default=file_names
        )

        if process_button:
            selected_uploaded_files = [
                file for file in uploaded_files
                if file.name in selected_files
            ]

            if not selected_uploaded_files:
                st.warning("Please select at least one file.")
                return

            progress_bar = st.progress(0)
            status_area = st.empty()

            with st.spinner("Extracting text and creating vector index..."):
                status_area.info("Step 1/4: Sorting uploaded files...")
                progress_bar.progress(15)

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

                status_area.info("Step 2/4: Extracting text from files...")
                progress_bar.progress(40)

                pdf_text = extract_text_from_pdfs(pdf_files)
                image_text = extract_text_from_images(image_files)
                docx_text = extract_text_from_docx(docx_files)

                combined_text = pdf_text + "\n" + image_text + "\n" + docx_text

                if combined_text.strip():
                    status_area.info("Step 3/4: Splitting text into chunks...")
                    progress_bar.progress(65)

                    text_chunks = split_text_into_chunks(combined_text)

                    status_area.info("Step 4/4: Creating FAISS vector index...")
                    progress_bar.progress(85)

                    vector_store = create_faiss_vector_store(text_chunks)

                    if vector_store:
                        progress_bar.progress(100)
                        status_area.success(
                            f"Files processed successfully. Created {len(text_chunks)} text chunks."
                        )
                    else:
                        status_area.error(
                            "Text was extracted, but FAISS creation failed. Check your Google API key."
                        )
                else:
                    status_area.error("No extractable text found.")

    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.expander("📜 Summarize Content", expanded=True):
            user_topic = st.text_input(
                "Topic optional",
                placeholder="Example: security, introduction, conclusion"
            )
            user_instruction = st.text_input(
                "Summary instruction",
                value="Create a clear summary"
            )
            summary_length = st.selectbox(
                "Summary length",
                ["short", "medium", "long"]
            )

            if st.button("✍️ Generate Summary", use_container_width=True):
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
                            st.success("Summary generated")
                            st.write(summary)
                            download_pdf(summary)

    with col2:
        with st.expander("💬 Ask Questions", expanded=True):
            user_topic_for_question = st.text_input(
                "Question topic optional",
                placeholder="Example: eligibility, cost, features"
            )
            user_question = st.text_area(
                "Your question",
                placeholder="Ask anything from the uploaded document...",
                height=120
            )

            if st.button("🔍 Get Answer", use_container_width=True):
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
                            st.success("Answer found")
                            st.write(answer)


if __name__ == "__main__":
    main()