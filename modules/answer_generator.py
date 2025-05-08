from typing import Dict
from langchain.chains import RetrievalQA
from tqdm import tqdm

def generate_answers(structured_questions: Dict, qa_chain: RetrievalQA) -> Dict:
    structured_answers = {}

    for part, sections in structured_questions.items():
        structured_answers[part] = {}
        for section, questions in sections.items():
            structured_answers[part][section] = []

            for q in questions:
                try:
                    query = q["question"]
                    result = qa_chain({"query": query})
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
