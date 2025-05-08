import os
import pickle
from typing import Tuple
import tempfile
import hashlib

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import fitz  # PyMuPDF

# Constants
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)

def compute_file_hash(file_data) -> str:
    file_data.seek(0)
    return hashlib.md5(file_data.read()).hexdigest()

def get_faiss_paths(file_hash: str) -> Tuple[str, str]:
    return (
        os.path.join(FAISS_DIR, f"{file_hash}.faiss"),
        os.path.join(FAISS_DIR, f"{file_hash}.pkl")
    )

def extract_text_data(file_data):
    file_data.seek(0)
    doc = fitz.open(stream=file_data.read(), filetype="pdf")
    documents, tables = [], []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            documents.append(Document(page_content=f"[Page {i}]\n{text}", metadata={"page": i}))
            tables.append((i, page.get_text("dict")))

    return documents, tables

def load_or_build_faiss(file, file_hash: str):
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

    return vectordb

def get_prompt_template():
    return PromptTemplate(
        input_variables=["context", "question", "marks"],
        template="""
You are an intelligent, highly knowledgeable tutor AI. Answer the following questions based strictly on the provided textbook context. Follow these detailed rules:

## Rules:
1. **For MCQs**, always choose one of the options (A, B, C, D) based on the context. Do not generate your own options.
2. **For non-MCQs**, answer based on the context. Provide detailed answers if the question is worth more marks.
3. If the question has **high marks** (greater than 3), your answer should be **more detailed** with relevant explanations, examples, or steps.
4. **If the question has low marks** (1-2 marks), provide **brief answers** with only the key information.
5. If the provided query is **not a valid question** (e.g., instruction or formula), respond with: "This is not a valid question."
6. Always format your answers using **Markdown**:
   - Use `##` for headings, bullet points for lists, LaTeX (`$$`) for equations, and code blocks where necessary.
7. **Be concise yet complete** and make sure your answer strictly follows the rules provided.

---

## Context:
{context}

---

## Question:
{question}

---



---

## Answer:
"""
    )


def init_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15})
    llm = Ollama(model="deepseek-llm:7b")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()}
    )
