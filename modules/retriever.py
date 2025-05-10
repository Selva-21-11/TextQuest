import os
import pickle
import hashlib
import tempfile

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import fitz

# Constants
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)

def compute_file_hash(file_data) -> str:
    file_data.seek(0)
    return hashlib.md5(file_data.read()).hexdigest()

def get_faiss_paths(file_hash: str) -> tuple:
    return (
        os.path.join(FAISS_DIR, f"{file_hash}.faiss"),
        os.path.join(FAISS_DIR, f"{file_hash}.pkl")
    )

def extract_text_data(file_data):
    file_data.seek(0)
    doc = fitz.open(stream=file_data.read(), filetype="pdf")
    documents = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            documents.append(Document(page_content=f"[Page {i}]\n{text}", metadata={"page": i}))

    return documents, []

def load_or_build_faiss(file, file_hash: str):
    index_path, store_path = get_faiss_paths(file_hash)

    if os.path.exists(index_path) and os.path.exists(store_path):
        with open(store_path, "rb") as f:
            embeddings = pickle.load(f)
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs, _ = extract_text_data(file)
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
        input_variables=["context", "question"],
        template="""
You are an expert at answering ALL types of exam questions with 100% textbook accuracy.  
**Automatically detect question type and respond accordingly**:

### 1. **UNIVERSAL RULES**:
- Use ONLY the provided text - NEVER add external knowledge
- Match the EXACT format/depth the question demands
- For ambiguous questions, select the MOST PRECISE textbook answer

### 2. **AUTO-FORMATTING GUIDE**:

#### A) Multiple Choice Questions:
- Output ONLY exact option text  
- Example:  
  The bird called plaintively: a. happily b. sadly c. rigorously d. vainly  
  b. sadly  

#### B) Short Answers (1-5 sentences):
- Include ALL key details from text  
- Adjust length based on marks (1-2 marks: 1-2 sentences | 3-5 marks: 3-5 sentences)  
- Example:  
  What does INSV stand for? (2 marks)  
  INSV stands for Indian Naval Sailing Vessel used for training cadets in maritime skills.  

#### C) Long Answers (100-300 words):
- **Structure by marks**:  
  - **5 marks**: Intro + 3 body paras (causes/effects/solutions) + conclusion  
  - **8 marks**: Intro + 5 body paras (detailed analysis) + conclusion  
- Example for 5 marks:  
  Describe the seagull's struggles  
  [Intro] The young seagull's fear of flying... [Body 1] Cause: Height anxiety... [Body 2] Effect: Hunger... [Conclusion] Overcoming fear...  

#### D) Formal Letters:
- **Auto-format with**:  
  1. Sender/Receiver addresses  
  2. Date + Bold Subject  
  3. 5-paragraph body (problem → causes → effects → solutions → conclusion)  
  4. "Yours faithfully" + Name  
- Example:  
  Write a letter about TV's bad influence  
  [Full formatted letter with addresses/date/5 paras]  

#### E) Poetry/Passage Analysis:
- **Poetry**: Literal meaning + devices (rhyme/metaphor) + theme  
- **Passages**: Direct quotes + contextual explanation  
- Example:  
  Analyze "A Bird came down the Walk"  
  Literally describes... Metaphor "Velvet Head"... Theme: Nature's beauty...  

#### F) Grammar/Editing:
- **Voice Change**: Exact mechanical transformation  
- **Punctuation**: Correct errors without explanations  
- Example:  
  Change voice: "The portrait was painted by my grandmother"  
  My grandmother painted the portrait.  

#### G) Note-making/Summaries:
- Extract KEYWORDS in bullet points  
- Example:  
  Summarize the blood composition passage  
  - Red cells: Carry oxygen via hemoglobin  
  - White cells: Fight infections  
  - Platelets: Clot blood  

### 3. **SMART DETECTION FEATURES**:
1. **Auto-length adjustment**:  
   - 1-2 mark questions → 1-2 sentences  
   - 5+ mark questions → Paragraphs with structure  
2. **Context-aware responses**:  
   - If question references a poem → Analyze poetic devices  
   - If question mentions "letter" → Apply formal format  
3. **Error prevention**:  
   - Rejects non-textbook answers  
   - Flags unanswerable questions  

### 4. **PROHIBITED**:  
- "According to the text..."  
- Personal opinions  
- Marks/point references  
- Explanations unless explicitly asked  

---  
**Textbook Content**:  
{context}  

**Question**:  
{question}  

**Answer** (Auto-formatted based on question type):  
"""  
    )

def init_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15})
    llm = Ollama(model="gemma3:4b-it-qat")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()}
    )