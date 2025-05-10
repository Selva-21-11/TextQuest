from fpdf import FPDF
import io
import os

class PDFExporter(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

        # Register the regular and bold versions of the font
        font_path = os.path.join("fonts", "DejaVuSans.ttf")
        bold_font_path = os.path.join("fonts", "DejaVuSans-Bold.ttf")  # Ensure you have the bold font
        self.add_font("DejaVu", "", font_path, uni=True)  # Regular font
        self.add_font("DejaVu", "B", bold_font_path, uni=True)  # Bold font
        self.set_font("DejaVu", size=12)

        # Set default margins
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.set_top_margin(15)

    def add_title(self, title):
        self.set_font("DejaVu", "B", 16)  # Use bold for the title
        self.cell(0, 10, title, ln=True, align="C")
        self.ln(5)

    def safe_multicell(self, text, line_height=8):
        """
        Safely handles multi-line text with proper wrapping and alignment.
        """
        width = self.w - self.l_margin - self.r_margin  # Use the full width minus the margins
        self.set_font("DejaVu", "", 12)  # Default text font and size
        self.multi_cell(width, line_height, text)
        
    def add_question_answer(self, q_data):
        """
        Adds a question and answer pair to the PDF, formatted correctly.
        """
        # Question
        self.set_font("DejaVu", "B", 12)  # Bold for question
        self.multi_cell(0, 10, f"Q{q_data['number']}: {q_data['question']}")
        self.ln(3)
        
        # Answer
        answer = q_data['answer'].replace("\n", " ").strip()
        self.set_font("DejaVu", "", 12)  # Regular font for the answer
        self.safe_multicell(f"Answer: {answer}")
        self.ln(5)

    def add_section_title(self, section_title):
        """
        Adds a section title to the PDF.
        """
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, section_title, ln=True, align="L")
        self.ln(3)
        
    def add_subtitle(self, subtitle):
        """
        Adds a subtitle to the PDF.
        """
        self.set_font("DejaVu", "I", 12)  # Italic for subtitles
        self.cell(0, 10, subtitle, ln=True, align="L")
        self.ln(2)

    def export(self):
        """
        Export the PDF as a byte array for storage or download.
        """
        output = io.BytesIO()
        self.output(output)
        return output.getvalue()