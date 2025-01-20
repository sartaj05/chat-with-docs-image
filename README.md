{
    "project_name": "Document Summary Assistant",
    "description": "Document Summary Assistant is an application that takes any document (PDF/Image) and generates smart summaries. The app extracts text from uploaded PDFs and image files, then uses natural language processing (NLP) techniques to summarize the content. It offers a user-friendly interface for uploading documents, processing them, and receiving concise summaries based on user preferences. The app uses Streamlit for its frontend interface, making it simple to interact with the system through a web-based UI.",
    "features": [
        {
            "feature": "Document Upload",
            "description": "Users can upload PDF and image files (e.g., scanned documents). Supports both drag-and-drop and file picker interfaces for easy document uploads. Users can also choose whether to summarize an image or a PDF."
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
            "description": "The application uses Streamlit for a user-friendly, web-based interface that allows users to upload documents, select summary lengths, and view results in an interactive manner. Users can also select whether they want the summarization to be based on PDF or image documents."
        },
        {
            "feature": "Summary PDF Generation",
            "description": "Once the summary is generated, the app automatically creates a PDF of the summary. Users can download this PDF summary directly from the Streamlit interface."
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
            "Create a virtual environment: python3 -m venv venv (optional)",
            "Activate the virtual environment (Windows): venv\\Scripts\\activate or (macOS/Linux): source venv/bin/activate",
            "Install required dependencies: pip install -r requirements.txt",
            "Install Tesseract OCR on your machine (instructions are provided in the README)",
            "Run the Streamlit app: streamlit run app.py"
        ]
    },
    "dependencies": [
        "pdfminer.six",
        "pytesseract",
        "sumy",
        "nltk",
        "streamlit",
        "reportlab"
    ],
    "usage": {
        "steps": [
            "Upload a PDF or image document using the Streamlit interface.",
            "The app processes the file and extracts the content (using pdfminer.six for PDFs or pytesseract for images).",
            "Choose the summary length (short, medium, or long) from the Streamlit options.",
            "Select whether you want the summarization for a PDF or image file.",
            "View the generated summary, which highlights the key points of the document.",
            "Download the generated PDF summary directly from the Streamlit interface."
        ]
    },
    "license": "MIT License"
}
