import streamlit as st
import tempfile
from typing import Optional

from modules.retriever import (
    compute_file_hash, load_or_build_faiss,
    extract_text_data, init_qa_chain
)
from modules.parser import parse_question_structure
from modules.utils import render_answer
from modules.pdf_exporter import PDFExporter

# --- Streamlit Setup ---
st.set_page_config(
    page_title="TextQuest",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("TextQuest")

# --- Session State Init ---
DEFAULT_SESSION_STATE = {
    "chat_history": [],
    "processed_question_hashes": set(),
    "structured_questions": {},
    "structured_answers": {},
    "vectordb": None,
    "qa_chain": None,
    "textbook_hash": None,
    "answers_ready": False
}

for key, value in DEFAULT_SESSION_STATE.items():
    st.session_state.setdefault(key, value)

# --- Sidebar: File Uploads ---
with st.sidebar:
    st.header("Upload Documents")
    
    # Upload textbook
    textbook = st.file_uploader(
        "Textbook PDF",
        type="pdf",
        help="Upload the textbook PDF for reference"
    )
    
    # Upload question paper - enabled only after textbook is processed
    if st.session_state.vectordb:
        questions = st.file_uploader(
            "Question Paper PDF",
            type="pdf",
            help="Upload the question paper PDF"
        )
    else:
        st.file_uploader(
            "Question Paper PDF",
            type="pdf",
            disabled=True,
            help="Upload the textbook first to enable question paper upload"
        )
        questions = None


def format_question(question_data: dict) -> str:
    """Format question with metadata."""
    base = f"[Q{question_data['number']}] {question_data['question']}"
    if question_data.get("is_mcq", False):
        return f"{base} [MCQ]"  # Removed marks display
    return base  # Removed marks display

def process_textbook(uploaded_file) -> Optional["FAISS"]:
    """Process uploaded textbook and initialize QA system."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    file_hash = compute_file_hash(open(tmp_file_path, "rb"))
    if file_hash != st.session_state.textbook_hash:
        st.session_state.vectordb = load_or_build_faiss(open(tmp_file_path, "rb"), file_hash)
        st.session_state.qa_chain = init_qa_chain(st.session_state.vectordb)
        st.session_state.textbook_hash = file_hash
        st.success("Textbook processed successfully!")
        st.rerun()  # ✅ Updated method
    return st.session_state.vectordb

# --- Textbook Processing ---
if textbook and not st.session_state.vectordb:
    with st.spinner("Processing textbook..."):
        process_textbook(textbook)

# --- Question Paper Processing ---
if questions:
    q_hash = compute_file_hash(questions)
    if q_hash not in st.session_state.processed_question_hashes:
        with st.spinner("Parsing questions..."):
            docs, _ = extract_text_data(questions)
            st.session_state.structured_questions = parse_question_structure(docs)
            st.session_state.structured_answers = {}
            st.session_state.processed_question_hashes.add(q_hash)

        if st.session_state.qa_chain:
            for part, sections in st.session_state.structured_questions.items():
                st.markdown(f"## {part}")
                for section, questions_list in sections.items():
                    st.markdown(f"### {section}")
                    st.session_state.structured_answers.setdefault(part, {}).setdefault(section, [])
                    for q in questions_list:
                        with st.container():
                            st.markdown(f"**Q{q['number']}. {q['question']}**")

                            # To this:
                            if q.get("is_mcq", False):
                                st.caption("MCQ")  # Removed marks display

                            with st.spinner(f"Generating answer for Q{q['number']}..."):
                                try:
                                    formatted_question = format_question(q)
                                    result = st.session_state.qa_chain({"query": formatted_question})
                                    render_answer(result["result"])

                                    if result["source_documents"]:
                                        with st.expander("View References"):
                                            for doc in result["source_documents"]:
                                                page = doc.metadata.get("page", "N/A")
                                                st.caption(f"Page {page}")
                                                st.text(doc.page_content[:500] + "...")

                                    st.session_state.structured_answers[part][section].append({
                                        "number": q["number"],
                                        "question": q["question"],
                                        "answer": result["result"],
                                        "sources": result["source_documents"],
                                        "marks": q["marks"]
                                    })

                                except Exception as e:
                                    st.error(f"Error answering question: {str(e)}")

            st.session_state.answers_ready = True

# --- Manual Question Input ---
if st.session_state.qa_chain:
    user_question = st.chat_input("Ask a question about the textbook...")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": user_question})
                    render_answer(result["result"])

                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": result["result"],
                        "sources": result["source_documents"]
                    })
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

# --- Display Chat History ---
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        render_answer(entry["answer"])
        if entry["sources"]:
            with st.expander("References"):
                for doc in entry["sources"]:
                    page = doc.metadata.get("page", "N/A")
                    st.caption(f"Page {page}")
                    st.text(doc.page_content[:500] + "...")

# --- Sidebar PDF Download ---
with st.sidebar:
    st.markdown("### Download Answers as PDF")
    if st.session_state.answers_ready and st.session_state.structured_answers:
        if st.button("Generate PDF"):
            pdf = PDFExporter()
            pdf.add_title("Generated Answers - TextQuest")

            for part, sections in st.session_state.structured_answers.items():
                pdf.set_font("DejaVu", "", 14)
                pdf.cell(0, 10, part, ln=True)
                pdf.ln(3)

                for section, q_list in sections.items():
                    pdf.set_font("DejaVu", "", 12)
                    pdf.cell(0, 10, section, ln=True)
                    pdf.ln(2)

                    for q in q_list:
                        pdf.add_question_answer(q)

            pdf_data = pdf.export()

            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="textquest_answers.pdf",
                mime="application/pdf"
            )
    else:
        st.button("Generate PDF", disabled=True)