from typing import Dict
from langchain.chains import RetrievalQA
from tqdm import tqdm
from modules.parser import extract_options

def generate_answers(structured_questions: Dict, qa_chain: RetrievalQA) -> Dict:
    structured_answers = {}

    for part, sections in structured_questions.items():
        structured_answers[part] = {}
        for section, questions in sections.items():
            structured_answers[part][section] = []

            for q in questions:
                try:
                    query = q["question"]
                    # Modify the QA chain call to:
                    result = qa_chain({
                        "query": query,
                        "marks": q["marks"],
                        "is_mcq": q.get("is_mcq", False),
                        "options": extract_options(q["question"]) if q.get("is_mcq", False) else ""
                    })
                    structured_answers[part][section].append({
                        "number": q["number"],
                        "question": q["question"],
                        "answer": result["result"],
                        "sources": result["source_documents"],
                        "marks": q["marks"]
                    })
                except Exception as e:
                    structured_answers[part][section].append({
                        "number": q["number"],
                        "question": q["question"],
                        "answer": f"❌ Error generating answer: {e}",
                        "sources": [],
                        "marks": q["marks"]
                    })

    return structured_answers