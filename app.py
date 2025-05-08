import os
import re
import fitz  # PyMuPDF
import pickle
import hashlib
import tempfile
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# --- Config ---
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Streamlit Setup ---
st.set_page_config(page_title="Textbook Q&A Chat", layout="wide")
st.title("📘 Textbook Q&A")

# --- Utility Functions ---

def compute_file_hash(file_data) -> str:
    """Compute the MD5 hash for a file."""
    file_data.seek(0)
    return hashlib.md5(file_data.read()).hexdigest()

def get_faiss_paths(file_hash):
    """Generate paths for FAISS index and embedding store based on file hash."""
    return (
        os.path.join(FAISS_DIR, f"{file_hash}.faiss"),
        os.path.join(FAISS_DIR, f"{file_hash}.pkl")
    )

def extract_text_data(file_data):
    """
    Extract text content from PDF and convert it to documents.
    This function uses PyMuPDF (fitz) to read and process the PDF.
    """
    file_data.seek(0)
    doc = fitz.open(stream=file_data.read(), filetype="pdf")
    documents, tables = [], []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            documents.append(Document(page_content=f"[Page {i}]\n{text}", metadata={"page": i}))
            tables.append((i, page.get_text("dict")))  # Render tables from here

    return documents, tables

def extract_questions_from_text(text):
    """Extract questions from the provided text."""
    questions, current = [], ""
    for line in text.splitlines():
        line = line.strip()
        if re.match(r'^(Q\d*[\).]?|[0-9]+[\).])\s', line):
            if current:
                questions.append(current.strip())
            current = line
        else:
            current += f" {line}"
    if current:
        questions.append(current.strip())
    return [q for q in questions if len(q.split()) > 4]

def render_answer(answer):
    """Render the answer with proper formatting (LaTeX, code blocks, etc.)."""
    code_block = False
    code_buffer = []
    
    for line in answer.splitlines():
        # Handle code blocks
        if line.strip().startswith("```"):
            if code_block:
                st.code("\n".join(code_buffer))
                code_block = False
                code_buffer = []
            else:
                code_block = True
        elif code_block:
            code_buffer.append(line)
        # Handle LaTeX formulas
        elif line.startswith("$$") and line.endswith("$$"):
            st.latex(line.strip("$$"))
        elif re.search(r"\\frac|\\sum|\\int|\\pi|\\theta|\\epsilon", line):
            st.latex(line)
        else:
            st.markdown(line)

def load_or_build_faiss(file, file_hash):
    """
    Load an existing FAISS index or build a new one from the provided file.
    Returns the FAISS vector store.
    """
    index_path, store_path = get_faiss_paths(file_hash)

    if os.path.exists(index_path) and os.path.exists(store_path):
        with open(store_path, "rb") as f:
            embeddings = pickle.load(f)
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs, tables = extract_text_data(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(index_path)
        with open(store_path, "wb") as f:
            pickle.dump(embeddings, f)

        st.session_state.table_data = tables

    return vectordb

def get_prompt_template():
    """Return the prompt template for the Q&A chain."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=""" 
        You are an intelligent, highly knowledgeable tutor AI answering questions strictly based on a textbook. Follow these detailed rules:
        
        ## Rules:
        1. Base answers only on the provided context.
        2. Start with a brief overview before specifics.
        3. Use Markdown:
            - `##` headings, bullet points, LaTeX (`$$`), and code blocks.
        4. Clarify formulas or code with steps/examples.
        5. Reference pages/figures/tables if applicable.
        6. For summaries/lists, be structured and hierarchical.
        7. Be concise yet complete.

        ---
        ## Context:
        {context}
        
        ---
        ## Question:
        {question}
        
        ---
        ## Answer:
        """
    )

def init_qa_chain(vectordb):
    """
    Initialize the Q&A chain with a retriever and language model.
    """
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15})
    llm = Ollama(model="deepseek-llm:7b")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()}
    )

# --- Sidebar: File Uploads ---
textbook = st.sidebar.file_uploader("📚 Upload Textbook PDF", type="pdf")
questions = st.sidebar.file_uploader("❓ Upload Question Paper (Optional)", type="pdf")

# --- Session State ---
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("processed_question_hashes", set())

# --- Vector DB Load/Build ---
if textbook and "vectordb" not in st.session_state:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(textbook.read())
    tmp_file_path = tmp_file.name
    tmp_file.close()

    file_hash = compute_file_hash(open(tmp_file_path, "rb"))
    st.session_state.vectordb = load_or_build_faiss(open(tmp_file_path, "rb"), file_hash)

# --- QA Logic ---
if "vectordb" in st.session_state:
    qa_chain = init_qa_chain(st.session_state.vectordb)

    # --- Uploaded Question File Handling ---
    if questions:
        q_hash = compute_file_hash(questions)
        if q_hash not in st.session_state.processed_question_hashes:
            docs, _ = extract_text_data(questions)
            q_text = "\n".join(d.page_content for d in docs)
            extracted_questions = extract_questions_from_text(q_text)

            if extracted_questions:
                st.subheader("🧠 Answers to Uploaded Questions:")
                for q in extracted_questions:
                    with st.spinner(f"Answering: {q[:60]}..."):
                        result = qa_chain({"query": q})
                        st.session_state.chat_history.append({
                            "question": q,
                            "answer": result["result"],
                            "sources": result["source_documents"],
                            "rendered": False
                        })

                        # Show answer immediately
                        with st.chat_message("user"):
                            st.markdown(q)
                        with st.chat_message("assistant"):
                            render_answer(result["result"])
                            if result["source_documents"]:
                                st.markdown("**References:**")
                                for doc in result["source_documents"]:
                                    page = doc.metadata.get("page", "?")
                                    with st.expander(f"📄 Page {page} Reference"):
                                        st.write(doc.page_content.strip())
                        st.session_state.chat_history[-1]["rendered"] = True
            else:
                st.warning("⚠️ No valid questions could be extracted.")
            st.session_state.processed_question_hashes.add(q_hash)

    # --- Manual Chat Input ---
    user_question = st.chat_input("💬 Ask a question from the textbook...")
    if user_question:
        result = qa_chain({"query": user_question})
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": result["result"],
            "sources": result["source_documents"],
            "rendered": False
        })

    # --- Display Previous Chat History ---
    for entry in st.session_state.chat_history:
        if not entry.get("rendered", False):
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                render_answer(entry["answer"])
                if entry["sources"]:
                    st.markdown("**References:**")
                    for doc in entry["sources"]:
                        page = doc.metadata.get("page", "?")
                        with st.expander(f"📄 Page {page} Reference"):
                            st.write(doc.page_content.strip())
            entry["rendered"] = True
