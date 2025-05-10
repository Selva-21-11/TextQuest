import re
from langchain.docstore.document import Document

import re

def parse_question_structure(documents: list) -> dict:
    structured_questions = {}
    current_part = None
    current_section = None
    section_marks_info = {}
    part_marks = {}  # Track marks for each part

    for doc in documents:
        lines = [normalize_text(line.strip()) for line in doc.page_content.splitlines()]
        lines = [line for line in lines if line]  # Remove empty lines

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect PART headers
            part_match = re.match(r'^PART\s*[-–—]?\s*([\w\d]+)', line, re.IGNORECASE)
            if part_match:
                current_part = f"Part {part_match.group(1).upper()}"
                structured_questions[current_part] = {}
                part_marks[current_part] = set()  # Initialize marks set for this part
                current_section = None
                i += 1
                continue

            # Detect SECTION headers (combine lines if needed)
            section_line = line
            marks_info = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.search(r'\d+\s*[x×*]\s*\d+', next_line):
                    marks_info = next_line.strip()
                    i += 1  # Skip the mark line

            section_match = re.match(r'^SECTION\s*[-–—]?\s*([\w\d]+)\s*(.*)', section_line, re.IGNORECASE)
            if section_match:
                section_num = section_match.group(1).strip()
                current_section = f"Section {section_num}"
                if current_part:
                    structured_questions[current_part][current_section] = []
                    # Store marks info for this section
                    section_marks_info[current_section] = marks_info
                i += 1
                continue

            # Set fallback section
            if current_part and not current_section:
                current_section = "General"
                if current_section not in structured_questions[current_part]:
                    structured_questions[current_part][current_section] = []
                    section_marks_info[current_section] = ""

            # Skip known instruction lines
            if is_instruction_line(line):
                i += 1
                continue

            # Question detection
            q_match = re.match(r'^(\(?\d+[a-zA-Z]?\)?)[\.\)]?\s*(.+)', line)
            if q_match:
                number = q_match.group(1).strip("().")
                question = q_match.group(2).strip()
                
                # Detect MCQ options
                options = []
                is_mcq = False
                option_pattern = r'^\(?([a-dA-D])\)?\s+.+'
                
                while i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if re.match(option_pattern, next_line):
                        options.append(next_line.strip())
                        is_mcq = True
                        i += 1
                    else:
                        break
                        
                # Format MCQ options clearly
                if is_mcq:
                    question += "\nOptions:\n" + "\n".join(options)
                    
                # Use the stored marks info for this section when estimating marks
                marks = estimate_marks(section_marks_info.get(current_section, ""))
                
                # Track marks for this part
                if current_part:
                    part_marks[current_part].add(marks)
                
                question_data = {
                    "number": number,
                    "question": question,
                    "marks": marks,
                    "is_mcq": is_mcq
                }
                structured_questions[current_part][current_section].append(question_data)
                i += 1
                continue

            # Append continuation lines for multi-line questions
            if structured_questions.get(current_part, {}).get(current_section):
                structured_questions[current_part][current_section][-1]["question"] += f" {line}"

            i += 1

    # Categorize parts based on marks
    categorized_questions = {}
    for part, sections in structured_questions.items():
        marks = part_marks.get(part, set())
        
        # Determine part type based on marks
        if not marks:
            part_name = part
        elif len(marks) == 1:
            mark = marks.pop()
            if mark == 1:
                part_name = f"{part} - Descriptive Answers"
            elif mark == 2:
                part_name = f"{part} - Short Answer"
            elif mark == 5:
                part_name = f"{part} - Descriptive Answers"
            elif mark >= 8:
                part_name = f"{part} - Descriptive Answers"
            else:
                part_name = f"{part} - Descriptive Answers"
        else:
            min_mark = min(marks)
            max_mark = max(marks)
            part_name = f"{part} - Questions ({min_mark}-{max_mark} Marks)"
        
        categorized_questions[part_name] = sections

    return categorized_questions

def extract_options(question_text: str) -> str:
    options = []
    for line in question_text.splitlines():
        opt_match = re.match(r'^\(?([a-dA-D])\)?\s+(.+)', line)
        if opt_match:
            options.append(f"{opt_match.group(1).upper()}. {opt_match.group(2)}")
    return "\n".join(options) if options else ""


def normalize_text(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_instruction_line(line: str) -> bool:
    instruction_patterns = [
        r'^answer (any|the)', r'^choose', r'^fill in', r'^rewrite', r'^rearrange',
        r'^punctuate', r'^report', r'^combine', r'^quote', r'^match the',
        r'^complete the following', r'^read the', r'^write (a|an|the)?',
        r'^identify', r'^make notes', r'^paraphrase', r'^prepare',
        r'^each question carries', r'^attempt any', r'^section [a-z]+ carries'
    ]
    return any(re.search(p, line, re.IGNORECASE) for p in instruction_patterns)


def estimate_marks(section_text: str) -> int:
    # Look for "x" or "×" for multiplying marks (e.g., "2 x 5")
    match = re.search(r'(\d+)\s*[x×*]\s*(\d+)', section_text)
    if match:
        return int(match.group(2))

    # Look for explicit marks (e.g., "1 mark", "3 marks")
    for m in [1, 2, 3, 4, 5, 6, 8, 10]:
        if f"{m} mark" in section_text.lower() or f"{m} marks" in section_text.lower():
            return m

    # Fallback to default mark
    return 1