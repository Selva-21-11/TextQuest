import re
import streamlit as st

def sanitize_latex(latex: str) -> str:
    unsupported = [
        r"\\begin{equation}", r"\\end{equation}", r"\\ref", r"\\cite"
    ]
    for cmd in unsupported:
        latex = latex.replace(cmd, "")
    latex = re.sub(r'[^\x00-\x7F]+', '', latex)
    return latex

def render_answer(answer: str):
    code_block = False
    code_buffer = []

    for line in answer.splitlines():
        if line.strip().startswith("```"):
            if code_block:
                st.code("\n".join(code_buffer))
                code_block = False
                code_buffer = []
            else:
                code_block = True
        elif code_block:
            code_buffer.append(line)
        elif line.startswith("$$") and line.endswith("$$"):
            sanitized = sanitize_latex(line.strip("$$"))
            try:
                st.latex(f"$$ {sanitized} $$")
            except Exception as e:
                st.error(f"KaTeX error: {e}")
        elif re.search(r"\\frac|\\sum|\\int|\\pi|\\theta|\\epsilon", line):
            sanitized = sanitize_latex(line)
            try:
                st.latex(sanitized)
            except Exception as e:
                st.error(f"KaTeX error: {e}")
        else:
            st.markdown(line)
