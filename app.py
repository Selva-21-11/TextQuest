import streamlit as st
import tempfile

from modules.retriever import (
    compute_file_hash, load_or_build_faiss,
    extract_text_data, init_qa_chain
)
from modules.parser import parse_question_structure
from modules.answer_generator import generate_answers
from modules.utils import render_answer

# --- Streamlit Setup ---
st.set_page_config(page_title="TextQuest", layout="wide")
st.title("TextQuest")

# --- Session State Init ---
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("processed_question_hashes", set())
st.session_state.setdefault("structured_questions", {})
st.session_state.setdefault("structured_answers", {})
st.session_state.setdefault("vectordb", None)
st.session_state.setdefault("qa_chain", None)

# --- Sidebar: File Uploads ---
textbook = st.sidebar.file_uploader("Upload Textbook PDF", type="pdf")
questions = st.sidebar.file_uploader("Upload Question Paper (PDF)", type="pdf")

# --- Textbook VectorDB Load/Build ---
if textbook and st.session_state.vectordb is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(textbook.read())
        tmp_file_path = tmp_file.name

    file_hash = compute_file_hash(open(tmp_file_path, "rb"))
    st.session_state.vectordb = load_or_build_faiss(open(tmp_file_path, "rb"), file_hash)

# --- QA Chain Initialization ---
if st.session_state.vectordb and st.session_state.qa_chain is None:
    st.session_state.qa_chain = init_qa_chain(st.session_state.vectordb)

# --- Question Paper Parsing ---
# --- Question Paper Upload + Immediate Parsing and Answering ---
if questions:
    q_hash = compute_file_hash(questions)
    if q_hash not in st.session_state.processed_question_hashes:
        docs, _ = extract_text_data(questions)
        structured = parse_question_structure(docs)
        st.session_state.structured_questions = structured
        st.session_state.structured_answers = {}

        # Live answering on upload
        if st.session_state.qa_chain:
            for part, sections in structured.items():
                st.session_state.structured_answers[part] = {}
                st.markdown(f"## {part}")
                for section, questions_list in sections.items():
                    st.session_state.structured_answers[part][section] = []
                    st.markdown(f"### {section}")
                    for q in questions_list:
                        with st.spinner(f"Answering Q{q['number']}..."):
                            try:
                                result = st.session_state.qa_chain({"query": q["question"]})
                                st.markdown(f"**Q{q['number']}. {q['question']}**")
                                render_answer(result["result"])
                                if result["source_documents"]:
                                    st.markdown("**References:**")
                                    for doc in result["source_documents"]:
                                        page = doc.metadata.get("page", "?")
                                        with st.expander(f"📄 Page {page} Reference"):
                                            st.write(doc.page_content.strip())

                                st.session_state.structured_answers[part][section].append({
                                    "number": q["number"],
                                    "question": q["question"],
                                    "answer": result["result"],
                                    "sources": result["source_documents"],
                                    "marks": q["marks"]
                                })

                            except Exception as e:
                                st.error(f"Error answering Q{q['number']}: {e}")

        st.session_state.processed_question_hashes.add(q_hash)


# --- Manual Question Input ---
if st.session_state.qa_chain and st.session_state.vectordb:
    user_question = st.chat_input("💬 Ask a custom question...")
    if user_question:
        result = st.session_state.qa_chain({"query": user_question})
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": result["result"],
            "sources": result["source_documents"]
        })

# --- Display Manual Chat History ---
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        render_answer(entry["answer"])
        if entry["sources"]:
            st.markdown("**References:**")
            for doc in entry["sources"]:
                page = doc.metadata.get("page", "?")
                with st.expander(f"Page {page} Reference"):
                    st.write(doc.page_content.strip())
