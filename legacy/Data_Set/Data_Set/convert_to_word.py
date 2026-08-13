import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MD_FILE = r"C:\Users\U SRIYA\.gemini\antigravity\brain\0c7f144d-eaf7-4bd8-ac3d-23f51558fc2c\preprocessing_documentation.md"
OUT_FILE = r"C:\Users\U SRIYA\Documents\Data_Set\Preprocessing_Documentation.docx"

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_table_from_md(doc, lines, start):
    """Parse markdown table lines and add as Word table."""
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith('|'):
        row = [c.strip() for c in lines[i].strip().split('|')[1:-1]]
        rows.append(row)
        i += 1

    # Remove separator row (---|---)
    rows = [r for r in rows if not all(re.match(r'^[-:]+$', c) for c in r if c)]

    if not rows:
        return i

    col_count = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=col_count)
    table.style = 'Table Grid'

    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            # Handle bold in cells
            p = cell.paragraphs[0]
            p.clear()
            parts = re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', cell_text)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith('`') and part.endswith('`'):
                    run = p.add_run(part[1:-1])
                    run.font.name = 'Courier New'
                    run.font.size = Pt(9)
                else:
                    p.add_run(part)
            # Header row styling
            if r_idx == 0:
                set_cell_bg(cell, '1F4E79')
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
    return i


def convert_md_to_docx(md_path, out_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Headings
        if stripped.startswith('#### '):
            p = doc.add_heading(stripped[5:], level=4)
            i += 1
        elif stripped.startswith('### '):
            p = doc.add_heading(stripped[4:], level=3)
            i += 1
        elif stripped.startswith('## '):
            p = doc.add_heading(stripped[3:], level=2)
            i += 1
        elif stripped.startswith('# '):
            p = doc.add_heading(stripped[2:], level=1)
            i += 1

        # Horizontal rule
        elif stripped == '---':
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '6')
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), '4472C4')
            pBdr.append(bottom)
            pPr.append(pBdr)
            i += 1

        # Table
        elif stripped.startswith('|'):
            i = add_table_from_md(doc, lines, i)
            doc.add_paragraph()  # spacing after table

        # Code block
        elif stripped.startswith('```'):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i].rstrip('\n'))
                i += 1
            i += 1  # skip closing ```
            p = doc.add_paragraph()
            p.style = 'No Spacing'
            run = p.add_run('\n'.join(code_lines))
            run.font.name = 'Courier New'
            run.font.size = Pt(9)
            # Light grey background via shading
            pPr = p._p.get_or_add_pPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), 'F2F2F2')
            pPr.append(shd)
            doc.add_paragraph()

        # Bullet points
        elif stripped.startswith('- '):
            text = stripped[2:]
            p = doc.add_paragraph(style='List Bullet')
            parts = re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', text)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith('`') and part.endswith('`'):
                    run = p.add_run(part[1:-1])
                    run.font.name = 'Courier New'
                    run.font.size = Pt(9)
                else:
                    p.add_run(part)
            i += 1

        # Normal paragraph
        else:
            p = doc.add_paragraph()
            parts = re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', stripped)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith('`') and part.endswith('`'):
                    run = p.add_run(part[1:-1])
                    run.font.name = 'Courier New'
                    run.font.size = Pt(9)
                else:
                    p.add_run(part)
            i += 1

    doc.save(out_path)
    print(f"[OK] Saved: {out_path}")


convert_md_to_docx(MD_FILE, OUT_FILE)
