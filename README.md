# Document Summary Assistant

Document Summary Assistant is an application that takes any document (PDF/Image) and generates smart summaries. The app extracts text from uploaded PDFs and image files, then uses natural language processing (NLP) techniques to summarize the content. It offers a user-friendly interface for uploading documents, processing them, and receiving concise summaries based on user preferences. The app uses Streamlit for its frontend interface, making it simple to interact with the system through a web-based UI.

## Features

### 1. Document Upload
- Users can upload PDF and image files (e.g., scanned documents).
- Supports both drag-and-drop and file picker interfaces for easy document uploads.
- Users can choose whether to summarize an image or a PDF.

### 2. Text Extraction
- Text is extracted from PDF documents using **pdfminer.six** while maintaining their original formatting.
- For image files, OCR technology (e.g., **Tesseract**) is used for text extraction. If you upload an image, ensure that Tesseract OCR is installed on your system, as it is required for processing image files.

### 3. Summary Generation
- Automatically generates summaries of the document content using the **sumy** library.
- Users can choose the summary length (short, medium, long).
- Summaries highlight key points and main ideas, ensuring that the essential information is captured.

### 4. Streamlit Interface
- The application uses **Streamlit** for a user-friendly, web-based interface.
- Users can upload documents, select summary lengths, and view results in an interactive manner.
- Users can also select whether they want the summarization to be based on PDF or image documents.

### 5. Summary PDF Generation
- Once the summary is generated, the app automatically creates a PDF of the summary.
- Users can download this PDF summary directly from the Streamlit interface.

### 6. Question-Answer Functionality (New Feature)
- Users can ask questions related to the content of the uploaded PDF document.
- The app processes the query and retrieves the most relevant answer based on the extracted text.
- Leverages natural language processing (NLP) techniques to provide accurate and context-aware responses.
- An intuitive input field in the Streamlit interface allows users to type their questions and view the answers instantly.

## Installation

### Prerequisites
- Python 3.6+
- Required Python packages (listed in `requirements.txt`)
- Tesseract OCR installed

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/sartaj05/chat-with-docs-image.git
   ```
2. Create a virtual environment (optional):
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Install Tesseract OCR on your machine (instructions are provided in the README).
6. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dependencies
- **pdfminer.six**
- **pytesseract**
- **sumy**
- **nltk**
- **streamlit**
- **reportlab**
- **google-generativeai**
- **python-dotenv**
- **langchain**
- **PyPDF2**
- **faiss-cpu**
- **langchain_google_genai**
- **Pillow**
- **langchain==0.3.14**
- **openai**
- **langchain-community==0.3.14**
- **fpdf**
- **pypdf**

## Usage

### Steps
1. Upload a PDF or image document using the Streamlit interface.
2. The app processes the file and extracts the content (using **pdfminer.six** for PDFs or **pytesseract** for images).
3. Choose the summary length (short, medium, or long) from the Streamlit options.
4. Select whether you want the summarization for a PDF or image file.
5. View the generated summary, which highlights the key points of the document.
6. Ask questions related to the uploaded PDF content in the provided input field.
7. The app will generate and display an answer based on the document content.
8. Download the generated PDF summary directly from the Streamlit interface.
9. Utilize the **question-answer** functionality to extract specific details or clarifications about the uploaded document for enhanced understanding.

## License
This project is licensed under the MIT License.
