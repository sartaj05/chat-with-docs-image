{
    "project_name": "Document Summary Assistant",
    "description": "Document Summary Assistant is an application that takes any document (PDF/Image) and generates smart summaries. The app extracts text from uploaded PDFs and image files, then uses natural language processing (NLP) techniques to summarize the content. It offers a user-friendly interface for uploading documents, processing them, and receiving concise summaries based on user preferences. The app uses Streamlit for its frontend interface, making it simple to interact with the system through a web-based UI.",
    "features": [
        {
            "feature": "Document Upload",
            "description": "Users can upload PDF and image files (e.g., scanned documents). Supports both drag-and-drop and file picker interfaces for easy document uploads."
        },
        {
            "feature": "Text Extraction",
            "description": "Text is extracted from PDF documents using pdfminer.six while maintaining their original formatting. For image files, OCR technology (e.g., Tesseract) is used for text extraction."
        },
        {
            "feature": "Summary Generation",
            "description": "Automatically generates summaries of the document content using the sumy library. Users can choose the summary length (short, medium, long). Summaries highlight key points and main ideas, ensuring that the essential information is captured."
        },
        {
            "feature": "Streamlit Interface",
            "description": "The application uses Streamlit for a user-friendly, web-based interface that allows users to upload documents, select summary lengths, and view results in an interactive manner."
        }
    ],
    "installation": {
        "prerequisites": [
            "Python 3.6+",
            "Required Python packages (listed in requirements.txt)",
            "Tesseract OCR installed"
        ],
        "steps": [
            "Clone the repository: git clone https://github.com/sartaj05/chat-with-docs-image.git",
            "Create a virtual environment: python3 -m venv venv",
            "Activate the virtual environment (Windows): venv\\Scripts\\activate or (macOS/Linux): source venv/bin/activate",
            "Install required dependencies: pip install -r requirements.txt",
            "Install Tesseract OCR on your machine (instructions are provided in the README)",
            "Run the Streamlit app: streamlit run app.py"
        ]
    },
    "dependencies": [
        "flask",
        "pdfminer.six",
        "pytesseract",
        "sumy",
        "nltk",
        "streamlit"
    ],
    "usage": {
        "steps": [
            "Upload a PDF or image document using the Streamlit interface.",
            "The app processes the file and extracts the content (using pdfminer.six for PDFs or pytesseract for images).",
            "Choose the summary length (short, medium, or long) from the Streamlit options.",
            "View the generated summary, which highlights the key points of the document."
        ]
    },
    "license": "MIT License"
}
