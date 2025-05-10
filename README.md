
# TextQuest - AI-Powered Textbook Question Answering System

![TextQuest Logo](logo.png)

*A smart assistant for answering textbook questions with precision*

---

## 📖 Overview

TextQuest is an AI-powered application that helps students and educators generate accurate answers to textbook questions. By uploading a textbook PDF and a question paper, the system:

- Extracts and structures questions from the paper  
- Analyzes the textbook content using advanced NLP  
- Generates precise answers based on the textbook material  
- Formats answers according to question type (MCQ, short answer, essay, etc.)  
- Exports results as a well-formatted PDF

---

## ✨ Key Features

- **Textbook Analysis**: Processes PDF textbooks into searchable knowledge bases  
- **Smart Question Parsing**: Automatically detects question types and structures  
- **Context-Aware Answers**: Generates answers strictly from textbook content  
- **Adaptive Formatting**: Formats responses based on question type and marks  
- **PDF Export**: Creates printable answer sheets with proper formatting  
- **Interactive Chat**: Ask additional questions about the textbook content

---

## 🛠️ Technology Stack

- **Backend**: Python  
- **NLP Framework**: LangChain  
- **Embeddings**: HuggingFace (`sentence-transformers/all-mpnet-base-v2`)  
- **Vector Database**: FAISS  
- **LLM**: Gemma3 (via Ollama)  
- **Web Interface**: Streamlit  
- **PDF Processing**: PyMuPDF (`fitz`)  
- **PDF Generation**: FPDF2

---

## 🚀 Installation

### Prerequisites

- Python 3.9+  
- Ollama running with Gemma3 model (`gemma3:4b-it-qat`)  
- Required fonts (DejaVu Sans) in `fonts/` directory

### Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/textquest.git
cd textquest
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download required fonts:

- Place `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` in the `fonts/` directory

Start Ollama (if not running):

```bash
ollama serve
```

---

## 🏃‍♂️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to:  
[http://localhost:8501](http://localhost:8501)

---

## 🖥️ Usage Guide

### Upload Textbook:

- In the sidebar, upload your textbook PDF  
- Wait for processing to complete (first time may take several minutes)

### Upload Question Paper:

- After textbook processing, upload your question paper PDF  
- The system will parse and structure the questions

### View Generated Answers:

- Answers will appear automatically in the main panel  
- Click **"View References"** to see textbook sources

### Ask Additional Questions:

- Use the chat input at the bottom to ask custom questions

### Export Answers:

- Click **"Generate PDF"** in the sidebar to download all answers

---

## 📂 Project Structure

```
textquest/
├── app.py                 # Main Streamlit application
├── retriever.py           # Textbook processing and QA system
├── parser.py              # Question paper parsing logic
├── answer_generator.py    # Answer generation functions
├── pdf_exporter.py        # PDF export functionality
├── utils.py               # Utility functions
├── fonts/                 # Font files directory
├── faiss_indexes/         # Stores processed textbook indexes
└── README.md              # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository  
2. Create a new branch:  
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:  
   ```bash
   git commit -am "Add some feature"
   ```
4. Push to the branch:  
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or support, please contact:  
**[Your Name]** – [your.email@example.com]  
Project Link: [https://github.com/yourusername/textquest](https://github.com/yourusername/textquest)

---

**TextQuest** – *Making textbook learning smarter and more efficient* 📚✨
