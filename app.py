import os
import re
import shutil
import platform
from typing import List, Dict, Tuple

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
from langchain_core.documents import Document as LCDocument
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
    tesseract_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


COLLECTIONS_DIR = "collections"
DEFAULT_COLLECTION = "default"


def apply_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(147, 197, 253, 0.20), transparent 34%),
                radial-gradient(circle at top right, rgba(196, 181, 253, 0.18), transparent 32%),
                linear-gradient(135deg, #eaf1f8 0%, #e7eef7 45%, #edf0f6 100%);
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
            background: linear-gradient(180deg, #f4f8fb 0%, #e7eef6 100%);
            border-right: 1px solid #c8d7ea;
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
            background: linear-gradient(135deg, #f9fbfd 0%, #f1f6fb 100%);
            border: 1px solid #cbdcf0;
            box-shadow: 0 10px 28px rgba(30, 64, 175, 0.07);
            margin-bottom: 1.6rem;
        }

        .collection-card {
            padding: 0.95rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #eef6ff 0%, #edf4fb 100%);
            border: 1px solid #bfd4ef;
            margin-bottom: 1rem;
            font-weight: 750;
            color: #0f3f5f !important;
        }

        .metric-card {
            padding: 0.75rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #f8fbfd 0%, #eef5fb 100%);
            border: 1px solid #cbdcf0;
            text-align: center;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.035);
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
            background: linear-gradient(135deg, #e4f4ee 0%, #e7f0fa 100%);
            border: 1px solid #b8d4e8;
            color: #0f3f5f !important;
            font-weight: 750;
            margin-top: 1rem;
            line-height: 1.55;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035);
        }

        .file-card {
            padding: 0.75rem 0.9rem;
            border-radius: 13px;
            background: linear-gradient(135deg, #f8fbfd 0%, #eef5fb 100%);
            border: 1px solid #cbdcf0;
            margin-bottom: 0.5rem;
            box-shadow: 0 5px 14px rgba(15, 23, 42, 0.03);
        }

        .source-card {
            padding: 0.75rem 0.9rem;
            border-radius: 13px;
            background: #eef6ff;
            border: 1px solid #bfd4ef;
            margin-bottom: 0.55rem;
            font-size: 0.92rem;
        }

        .chat-user-card {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
            border: 1px solid #bfdbfe;
            margin-bottom: 0.7rem;
        }

        .chat-ai-card {
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #f8fafc 0%, #edf5fb 100%);
            border: 1px solid #cbdcf0;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }

        .chat-label {
            font-size: 0.82rem;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 0.35rem;
        }

        .small-note {
            color: #64748b !important;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        div[data-testid="stExpander"] {
            background: linear-gradient(180deg, #f9fbfd 0%, #f2f6fa 100%);
            border: 1px solid #cbdcf0;
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(30, 64, 175, 0.07);
            overflow: hidden;
        }

        div[data-testid="stExpander"] summary {
            background: linear-gradient(90deg, #eaf2fb 0%, #eef1fa 100%);
            color: #163b88 !important;
            font-weight: 800;
            border-bottom: 1px solid #d2dfef;
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
            background: #f8fafc !important;
            color: #172033 !important;
            border: 1px solid #aebfd4 !important;
            border-radius: 14px !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03) !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border: 1px solid #5b9beb !important;
            box-shadow: 0 0 0 3px rgba(91, 155, 235, 0.16) !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #7b8aa0 !important;
        }

        div[data-baseweb="select"] > div {
            background: #f8fafc !important;
            color: #172033 !important;
            border: 1px solid #aebfd4 !important;
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
            box-shadow: 0 8px 18px rgba(14, 165, 233, 0.18);
            transition: all 0.18s ease-in-out;
        }

        .stButton button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 100%) !important;
            color: white !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(14, 165, 233, 0.24);
        }

        .stDownloadButton button {
            background: linear-gradient(135deg, #059669 0%, #34d399 100%) !important;
            color: white !important;
            border-radius: 14px !important;
            font-weight: 750 !important;
            border: none !important;
        }

        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #f6f9fc 0%, #edf4fb 100%);
            border: 1px dashed #8cb9e8;
            border-radius: 18px;
            padding: 0.8rem;
        }

        [data-testid="stFileUploader"] section {
            background: #f8fafc !important;
            border: 1px solid #cbdcf0 !important;
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
            border-color: #cfdced;
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
    
def initialize_session_state():

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if "active_collection" not in st.session_state:
        st.session_state.active_collection = DEFAULT_COLLECTION


def sanitize_collection_name(name: str) -> str:

    name = name.strip().lower()

    name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        name
    )

    return name


def ensure_collections_dir():

    os.makedirs(
        COLLECTIONS_DIR,
        exist_ok=True
    )


def get_collection_path(
    collection_name: str
) -> str:

    ensure_collections_dir()

    return os.path.join(
        COLLECTIONS_DIR,
        collection_name
    )


def collection_exists(
    collection_name: str
) -> bool:

    path = get_collection_path(
        collection_name
    )

    return (
        os.path.exists(path)
        and
        os.path.exists(
            os.path.join(
                path,
                "index.faiss"
            )
        )
    )


def list_collections():

    ensure_collections_dir()

    collections = []

    for item in os.listdir(
        COLLECTIONS_DIR
    ):

        full_path = os.path.join(
            COLLECTIONS_DIR,
            item
        )

        if os.path.isdir(
            full_path
        ):
            collections.append(
                item
            )

    collections.sort()

    if DEFAULT_COLLECTION not in collections:
        collections.insert(
            0,
            DEFAULT_COLLECTION
        )

    return collections


def create_collection(
    collection_name: str
):

    collection_name = sanitize_collection_name(
        collection_name
    )

    if not collection_name:
        return None

    path = get_collection_path(
        collection_name
    )

    os.makedirs(
        path,
        exist_ok=True
    )

    return collection_name


def delete_collection(
    collection_name: str
):

    if collection_name == DEFAULT_COLLECTION:
        return False

    path = get_collection_path(
        collection_name
    )

    if os.path.exists(path):
        shutil.rmtree(path)

    if (
        collection_name
        in
        st.session_state.chat_histories
    ):
        del st.session_state.chat_histories[
            collection_name
        ]

    return True


def get_chat_history(
    collection_name: str
):

    if (
        collection_name
        not in
        st.session_state.chat_histories
    ):
        st.session_state.chat_histories[
            collection_name
        ] = []

    return st.session_state.chat_histories[
        collection_name
    ]


def save_chat_message(
    collection_name: str,
    question: str,
    answer: str,
    source_docs
):

    history = get_chat_history(
        collection_name
    )

    history.append(
        {
            "question": question,
            "answer": answer,
            "sources": source_docs
        }
    )


def clear_chat_history(
    collection_name: str
):

    st.session_state.chat_histories[
        collection_name
    ] = []


def build_chat_history_text(
    collection_name: str
):

    history = get_chat_history(
        collection_name
    )

    lines = []

    for index, item in enumerate(
        history,
        start=1
    ):

        lines.append(
            f"Q{index}: {item['question']}"
        )

        lines.append(
            f"A{index}: {item['answer']}"
        )

        lines.append("Sources:")

        seen = set()

        for doc in item["sources"]:

            label = build_source_label(
                doc.metadata
            )

            if label not in seen:
                seen.add(label)
                lines.append(
                    f"- {label}"
                )

        lines.append("")

    return "\n".join(lines)


def get_collection_index_path(
    collection_name: str
):

    return get_collection_path(
        collection_name
    )


def collection_has_index(
    collection_name: str
):

    path = get_collection_index_path(
        collection_name
    )

    return (
        os.path.exists(
            os.path.join(
                path,
                "index.faiss"
            )
        )
        and
        os.path.exists(
            os.path.join(
                path,
                "index.pkl"
            )
        )
    )


def reset_collection_index(
    collection_name: str
):

    path = get_collection_index_path(
        collection_name
    )

    faiss_file = os.path.join(
        path,
        "index.faiss"
    )

    pkl_file = os.path.join(
        path,
        "index.pkl"
    )

    if os.path.exists(
        faiss_file
    ):
        os.remove(
            faiss_file
        )

    if os.path.exists(
        pkl_file
    ):
        os.remove(
            pkl_file
        )


def build_source_label(
    metadata: Dict
) -> str:

    file_name = metadata.get(
        "file_name",
        "Unknown file"
    )

    file_type = metadata.get(
        "file_type",
        "Unknown"
    )

    page = metadata.get(
        "page"
    )

    chunk = metadata.get(
        "chunk"
    )

    if page:
        return (
            f"{file_name} | "
            f"{file_type} | "
            f"Page {page}"
        )

    return (
        f"{file_name} | "
        f"{file_type} | "
        f"Chunk {chunk}"
    )


def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
def split_text_with_metadata(
    text: str,
    base_metadata: Dict
) -> List[LCDocument]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )

    chunks = splitter.split_text(
        text
    )

    documents = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):
        metadata = base_metadata.copy()
        metadata["chunk"] = index

        documents.append(
            LCDocument(
                page_content=chunk,
                metadata=metadata
            )
        )

    return documents


def extract_documents_from_pdfs(
    pdf_docs
) -> List[LCDocument]:

    documents = []

    for pdf in pdf_docs:

        try:
            pdf.seek(0)

            reader = PdfReader(
                pdf
            )

            has_normal_text = False

            for page_index, page in enumerate(
                reader.pages,
                start=1
            ):

                page_text = (
                    page.extract_text()
                    or ""
                )

                if page_text.strip():

                    has_normal_text = True

                    page_docs = split_text_with_metadata(
                        page_text,
                        {
                            "file_name": pdf.name,
                            "file_type": "PDF",
                            "page": page_index
                        }
                    )

                    documents.extend(
                        page_docs
                    )

            if not has_normal_text:

                pdf.seek(0)

                images = convert_from_bytes(
                    pdf.read()
                )

                for page_index, image in enumerate(
                    images,
                    start=1
                ):

                    scanned_text = pytesseract.image_to_string(
                        image
                    )

                    if scanned_text.strip():

                        page_docs = split_text_with_metadata(
                            scanned_text,
                            {
                                "file_name": pdf.name,
                                "file_type": "Scanned PDF OCR",
                                "page": page_index
                            }
                        )

                        documents.extend(
                            page_docs
                        )

        except Exception as e:
            st.error(
                f"Error extracting PDF text from {pdf.name}: {e}"
            )

    return documents


def extract_documents_from_images(
    image_docs
) -> List[LCDocument]:

    documents = []

    for image_file in image_docs:

        try:
            image = Image.open(
                image_file
            )

            image_text = pytesseract.image_to_string(
                image
            )

            if image_text.strip():

                image_docs_generated = split_text_with_metadata(
                    image_text,
                    {
                        "file_name": image_file.name,
                        "file_type": "Image OCR"
                    }
                )

                documents.extend(
                    image_docs_generated
                )

        except Exception as e:
            st.error(
                f"Error extracting image text from {image_file.name}: {e}"
            )

    return documents


def extract_documents_from_docx(
    docx_docs
) -> List[LCDocument]:

    documents = []

    for docx_file in docx_docs:

        try:
            document = Document(
                docx_file
            )

            docx_text = ""

            for para in document.paragraphs:
                if para.text.strip():
                    docx_text += (
                        para.text
                        + "\n"
                    )

            for table in document.tables:

                for row in table.rows:

                    row_values = []

                    for cell in row.cells:
                        row_values.append(
                            cell.text.strip()
                        )

                    docx_text += (
                        " | ".join(
                            row_values
                        )
                        + "\n"
                    )

            if docx_text.strip():

                docx_docs_generated = split_text_with_metadata(
                    docx_text,
                    {
                        "file_name": docx_file.name,
                        "file_type": "DOCX"
                    }
                )

                documents.extend(
                    docx_docs_generated
                )

        except Exception as e:
            st.error(
                f"Error extracting DOCX text from {docx_file.name}: {e}"
            )

    return documents


def create_faiss_vector_store(
    collection_name: str,
    documents: List[LCDocument]
):

    try:

        embeddings = get_embeddings()

        vector_store = FAISS.from_documents(
            documents,
            embedding=embeddings
        )

        collection_path = get_collection_index_path(
            collection_name
        )

        os.makedirs(
            collection_path,
            exist_ok=True
        )

        vector_store.save_local(
            collection_path
        )

        return vector_store

    except Exception as e:
        st.error(
            f"Error creating FAISS vector store: {e}"
        )

        return None


def load_faiss_vector_store(
    collection_name: str
):

    try:

        embeddings = get_embeddings()

        collection_path = get_collection_index_path(
            collection_name
        )

        return FAISS.load_local(
            collection_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:
        st.error(
            f"Error loading FAISS vector store: {e}"
        )

        return None


def search_documents(
    collection_name: str,
    query: str,
    k: int = 6
) -> List[LCDocument]:

    vector_store = load_faiss_vector_store(
        collection_name
    )

    if not vector_store:
        return []

    return vector_store.similarity_search(
        query,
        k=k
    )


def show_sources(
    docs: List[LCDocument],
    title: str = "Sources"
):

    if not docs:
        return

    st.markdown(
        f"### 🔎 {title}"
    )

    seen = set()
    unique_docs = []

    for doc in docs:

        label = build_source_label(
            doc.metadata
        )

        if label not in seen:
            seen.add(
                label
            )
            unique_docs.append(
                doc
            )

    for i, doc in enumerate(
        unique_docs,
        start=1
    ):

        label = build_source_label(
            doc.metadata
        )

        preview = doc.page_content[
            :900
        ].strip()

        st.markdown(
            f"""
            <div class="source-card">
                <b>{i}. {label}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            f"View source preview {i}",
            expanded=False
        ):
            st.write(
                preview
            )
            
def initialize_qa_chain():

    prompt_template = """
You are a helpful document assistant.

Answer the user's question using only the provided context.

Rules:
- Be clear and direct.
- Do not invent information.
- If the answer is not available in the context, say:
  "I could not find this information in the uploaded document."
- The app will show source citations separately below your answer.

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
        input_variables=[
            "context",
            "question"
        ]
    )

    return load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )


def initialize_summary_chain():

    prompt_template = """
You are a document summarization assistant.

Use only the provided context.

Instruction:
{instruction}

Summary Length:
{summary_length_instruction}

Rules:
- Summarize clearly.
- Do not invent information.
- Keep the result useful and structured.
- The app will show source citations separately below the summary.

Context:
{context}

Write the summary:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "instruction",
            "summary_length_instruction"
        ]
    )

    return load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )


def summarize_documents(
    collection_name: str,
    user_instruction,
    topic=None,
    summary_length="short"
) -> Tuple[str, List[LCDocument]]:

    query = user_instruction

    if topic:
        query += " " + topic

    docs = search_documents(
        collection_name,
        query,
        k=6
    )

    if not docs:
        return "", []

    length_mapping = {
        "short":
            "Write a short summary with only the most important points.",
        "medium":
            "Write a medium-length summary with key points and useful details.",
        "long":
            "Write a detailed summary with important explanations and structure."
    }

    try:

        chain = initialize_summary_chain()

        response = chain(
            {
                "input_documents": docs,
                "instruction": user_instruction,
                "summary_length_instruction":
                    length_mapping.get(
                        summary_length,
                        length_mapping["short"]
                    )
            },
            return_only_outputs=True
        )

        return response["output_text"], docs

    except Exception as e:
        st.error(
            f"Error generating summary: {e}"
        )

        return "", docs


def answer_user_question(
    collection_name: str,
    user_question,
    topic=None
) -> Tuple[str, List[LCDocument]]:

    query = user_question

    if topic:
        query += " " + topic

    docs = search_documents(
        collection_name,
        query,
        k=6
    )

    if not docs:
        return "", []

    try:

        chain = initialize_qa_chain()

        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )

        return response["output_text"], docs

    except Exception as e:
        st.error(
            f"Error generating answer: {e}"
        )

        return "", docs


def generate_pdf_summary(
    summary
):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font(
        "Arial",
        size=12
    )

    safe_summary = summary.encode(
        "latin-1",
        "replace"
    ).decode(
        "latin-1"
    )

    pdf.multi_cell(
        0,
        10,
        txt=safe_summary,
        align="L"
    )

    return pdf.output(
        dest="S"
    ).encode(
        "latin1"
    )


def download_pdf(
    summary
):

    pdf_data = generate_pdf_summary(
        summary
    )

    st.download_button(
        label="⬇️ Download PDF Summary",
        data=pdf_data,
        file_name="summary.pdf",
        mime="application/pdf"
    )


def download_chat_history(
    collection_name: str
):

    history = get_chat_history(
        collection_name
    )

    if not history:
        return

    chat_text = build_chat_history_text(
        collection_name
    )

    st.download_button(
        label="⬇️ Download Chat History",
        data=chat_text,
        file_name=f"{collection_name}_chat_history.txt",
        mime="text/plain"
    )


def render_chat_history(
    collection_name: str
):

    history = get_chat_history(
        collection_name
    )

    if not history:
        st.info(
            "No chat yet. Ask a question after processing your selected collection."
        )
        return

    for index, item in enumerate(
        history,
        start=1
    ):

        st.markdown(
            f"""
            <div class="chat-user-card">
                <div class="chat-label">You · Question {index}</div>
                {item["question"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="chat-ai-card">
                <div class="chat-label">Assistant · Answer {index}</div>
                {item["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        show_sources(
            item["sources"],
            title=f"Sources for Answer {index}"
        )
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

    initialize_session_state()
    apply_custom_css()
    ensure_collections_dir()

    st.markdown(
        """
        <div class="hero-card">
            <div class="main-title">📄 Document Summary Assistant</div>
            <div class="sub-title">
                Organize documents into collections, summarize them, ask questions,
                and verify answers with source citations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:

        st.markdown("## 🗂️ Document Library")

        new_collection_name = st.text_input(
            "Create collection",
            placeholder="Example: project_report"
        )

        if st.button(
            "➕ Create Collection",
            use_container_width=True
        ):
            created_name = create_collection(
                new_collection_name
            )

            if created_name:
                st.session_state.active_collection = created_name
                st.success(
                    f"Collection created: {created_name}"
                )
            else:
                st.warning(
                    "Please enter a valid collection name."
                )

        collections = list_collections()

        selected_collection = st.selectbox(
            "Active collection",
            options=collections,
            index=collections.index(
                st.session_state.active_collection
            )
            if st.session_state.active_collection in collections
            else 0
        )

        st.session_state.active_collection = selected_collection

        active_collection = st.session_state.active_collection

        st.markdown(
            f"""
            <div class="collection-card">
                📁 Active Collection:<br>
                <b>{active_collection}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("## 📂 Upload Center")

        st.markdown(
            '<p class="small-note">Supported: PDF, scanned PDF, JPG, PNG, DOCX.</p>',
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader(
            "Upload your files",
            accept_multiple_files=True,
            type=[
                "pdf",
                "jpg",
                "jpeg",
                "png",
                "docx"
            ]
        )

        if uploaded_files:

            counts = get_file_type_counts(
                uploaded_files
            )

            st.markdown("### 📊 Upload Summary")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{counts["PDF"]}</div>
                        <div class="metric-label">PDF</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_b:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{counts["Image"]}</div>
                        <div class="metric-label">Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_c:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{counts["DOCX"]}</div>
                        <div class="metric-label">DOCX</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### 📁 Selected Files")

            for file in uploaded_files:

                size_kb = round(
                    file.size / 1024,
                    2
                )

                st.markdown(
                    f"""
                    <div class="file-card">
                        <b>{file.name}</b><br>
                        <span class="small-note">{size_kb} KB</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        process_button = st.button(
            "🚀 Process Files Into Collection",
            use_container_width=True
        )

        if st.button(
            "🧹 Reset Selected Collection Index",
            use_container_width=True
        ):
            reset_collection_index(
                active_collection
            )
            clear_chat_history(
                active_collection
            )
            st.success(
                f"Index reset for: {active_collection}"
            )

        if active_collection != DEFAULT_COLLECTION:

            if st.button(
                "🗑️ Delete Selected Collection",
                use_container_width=True
            ):
                deleted = delete_collection(
                    active_collection
                )

                if deleted:
                    st.session_state.active_collection = DEFAULT_COLLECTION
                    st.success(
                        "Collection deleted."
                    )
                else:
                    st.warning(
                        "Default collection cannot be deleted."
                    )

        if st.button(
            "💬 Clear Collection Chat",
            use_container_width=True
        ):
            clear_chat_history(
                active_collection
            )
            st.success(
                "Chat history cleared for this collection."
            )

        download_chat_history(
            active_collection
        )

        if collection_has_index(
            active_collection
        ):
            st.markdown(
                """
                <div class="status-box">
                    ✅ Collection index is ready. You can summarize or ask questions.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="status-box">
                    ℹ️ No index for this collection. Upload and process files first.
                </div>
                """,
                unsafe_allow_html=True
            )

    if uploaded_files:

        file_names = [
            file.name
            for file in uploaded_files
        ]

        st.markdown("### 🧾 File Selection")

        selected_files = st.multiselect(
            "Choose which files to process into active collection",
            options=file_names,
            default=file_names
        )

        if process_button:

            selected_uploaded_files = [
                file
                for file in uploaded_files
                if file.name in selected_files
            ]

            if not selected_uploaded_files:
                st.warning(
                    "Please select at least one file."
                )
                return

            progress_bar = st.progress(0)
            status_area = st.empty()

            with st.spinner(
                f"Processing files into collection: {active_collection}"
            ):

                status_area.info(
                    "Step 1/4: Sorting uploaded files..."
                )
                progress_bar.progress(15)

                pdf_files = [
                    file
                    for file in selected_uploaded_files
                    if file.type == "application/pdf"
                    or file.name.lower().endswith(".pdf")
                ]

                image_files = [
                    file
                    for file in selected_uploaded_files
                    if file.type.startswith("image/")
                ]

                docx_files = [
                    file
                    for file in selected_uploaded_files
                    if file.name.lower().endswith(".docx")
                ]

                status_area.info(
                    "Step 2/4: Extracting text with metadata..."
                )
                progress_bar.progress(40)

                documents = []

                documents.extend(
                    extract_documents_from_pdfs(
                        pdf_files
                    )
                )

                documents.extend(
                    extract_documents_from_images(
                        image_files
                    )
                )

                documents.extend(
                    extract_documents_from_docx(
                        docx_files
                    )
                )

                if documents:

                    status_area.info(
                        "Step 3/4: Preparing source metadata..."
                    )
                    progress_bar.progress(65)

                    status_area.info(
                        "Step 4/4: Creating collection index..."
                    )
                    progress_bar.progress(85)

                    vector_store = create_faiss_vector_store(
                        active_collection,
                        documents
                    )

                    if vector_store:

                        progress_bar.progress(100)

                        status_area.success(
                            f"Collection '{active_collection}' updated with {len(documents)} indexed chunks."
                        )

                    else:

                        status_area.error(
                            "Text was extracted, but FAISS creation failed. Check your Google API key."
                        )

                else:

                    status_area.error(
                        "No extractable text found."
                    )

    st.divider()

    st.markdown(
        f"""
        <div class="collection-card">
            Working inside collection: <b>{st.session_state.active_collection}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.expander(
            "📜 Summarize Collection",
            expanded=True
        ):

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
                [
                    "short",
                    "medium",
                    "long"
                ]
            )

            if st.button(
                "✍️ Generate Summary",
                use_container_width=True
            ):

                if not collection_has_index(
                    active_collection
                ):
                    st.warning(
                        "Please process files into this collection first."
                    )

                elif not user_instruction.strip():

                    st.warning(
                        "Please enter summary instruction."
                    )

                else:

                    with st.spinner(
                        "Generating summary from active collection..."
                    ):

                        summary, source_docs = summarize_documents(
                            collection_name=active_collection,
                            user_instruction=user_instruction,
                            topic=user_topic,
                            summary_length=summary_length
                        )

                        if summary:

                            st.success(
                                "Summary generated"
                            )

                            st.write(
                                summary
                            )

                            download_pdf(
                                summary
                            )

                            show_sources(
                                source_docs,
                                title="Summary Sources"
                            )

    with col2:

        with st.expander(
            "💬 Ask Collection",
            expanded=True
        ):

            user_topic_for_question = st.text_input(
                "Question topic optional",
                placeholder="Example: eligibility, cost, features"
            )

            user_question = st.text_area(
                "Your question",
                placeholder="Ask anything from the active collection...",
                height=120
            )

            if st.button(
                "🔍 Ask and Save to Chat",
                use_container_width=True
            ):

                if not collection_has_index(
                    active_collection
                ):

                    st.warning(
                        "Please process files into this collection first."
                    )

                elif not user_question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Finding answer from active collection..."
                    ):

                        answer, source_docs = answer_user_question(
                            collection_name=active_collection,
                            user_question=user_question,
                            topic=user_topic_for_question
                        )

                        if answer:

                            save_chat_message(
                                active_collection,
                                user_question,
                                answer,
                                source_docs
                            )

                            st.success(
                                "Answer saved to this collection chat history"
                            )

    st.divider()

    st.markdown(
        f"## 💬 Chat History: `{active_collection}`"
    )

    render_chat_history(
        active_collection
    )


if __name__ == "__main__":
    main()
    