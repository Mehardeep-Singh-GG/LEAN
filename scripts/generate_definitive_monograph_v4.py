#!/usr/bin/env python3
"""
scripts/generate_definitive_monograph_v4.py

Project LEAN Definitive Archival Monograph (Version 4.0 — No Page Limit Edition)
Compiles the comprehensive, exhaustive research treatise documenting EVERY single
milestone (M0–M11), mathematical derivation, algorithm, statistical proof, factorial
ablation, systems accounting, hardware benchmark, failure analysis, and appendix.

Outputs:
  - paper/MANUSCRIPT_EXTENDED.md
  - paper/MANUSCRIPT.md
  - paper/manuscript_extended.html
  - paper/manuscript.html
  - paper/project_lean_monograph_v3.pdf
  - paper/project_lean_monograph_v2.pdf
  - paper/project_lean_manuscript.pdf
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add scripts directory to path to import sections
sys.path.insert(0, str(Path(__file__).resolve().parent))

import treatise_sections as s1_3
import treatise_sections_part2 as s4_5
import treatise_sections_part3 as s6_8
import treatise_sections_part4 as s9_12

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

print("Starting compilation of Project LEAN Archival Monograph v4.0...")

# 1. Assemble Full Markdown Document
md_parts = [
    "# The Priced-Quanta Monograph: Decoupling Algorithmic Reasoning from Associative Memory for Sub-4-Bit Neural Compression\n\n"
    "**Project LEAN Research Initiative — Definitive Archival Treatise v4.0 (No Page Limit Edition)**  \n"
    "*Mehardeep Singh, Project LEAN Core Team*  \n"
    "`lean@antigravity.ai` | [https://github.com/Mehardeep-Singh-GG/LEAN](https://github.com/Mehardeep-Singh-GG/LEAN) | October 2026\n\n"
    "---\n\n",
    s1_3.get_section_abstract(),
    "\n\n---\n\n",
    s1_3.get_section_1(),
    "\n\n---\n\n",
    s1_3.get_section_2(),
    "\n\n---\n\n",
    s1_3.get_section_3(),
    "\n\n---\n\n",
    s4_5.get_section_4(),
    "\n\n---\n\n",
    s4_5.get_section_5(),
    "\n\n---\n\n",
    s6_8.get_section_6(),
    "\n\n---\n\n",
    s6_8.get_section_7(),
    "\n\n---\n\n",
    s6_8.get_section_8(),
    "\n\n---\n\n",
    s9_12.get_section_9(),
    "\n\n---\n\n",
    s9_12.get_section_10(),
    "\n\n---\n\n",
    s9_12.get_section_11(),
    "\n\n---\n\n",
    s9_12.get_section_12(),
]

full_markdown = "".join(md_parts)

# Write to both MANUSCRIPT_EXTENDED.md and MANUSCRIPT.md
for target_md in [PAPER_DIR / "MANUSCRIPT_EXTENDED.md", PAPER_DIR / "MANUSCRIPT.md"]:
    with open(target_md, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    print(f"Written {target_md} ({len(full_markdown)} characters).")

# 2. Build HTML with Embedded MathJax and Inline SVGs
# Convert Markdown tables and headers to clean HTML
import re

def markdown_to_html_simple(md_text):
    # Process headers
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    table_lines = []
    in_code = False
    code_lines = []
    code_lang = ""

    def flush_table(tbl):
        if not tbl:
            return ""
        out = ['<div class="table-wrap"><table class="academic-table">']
        header_done = False
        for idx, row in enumerate(tbl):
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if not cols:
                continue
            if "---" in cols[0]:
                header_done = True
                continue
            if not header_done:
                out.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>")
        out.append("</tbody></table></div>")
        return "\n".join(out)

    def flush_code(lines, lang):
        code_content = "\n".join(lines).replace("<", "&lt;").replace(">", "&gt;")
        return f'<pre><code class="language-{lang}">{code_content}</code></pre>'

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code block
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_lines = []
            else:
                in_code = False
                html_lines.append(flush_code(code_lines, code_lang))
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Tables
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_lines.append(line.strip())
            i += 1
            continue
        elif in_table:
            in_table = False
            html_lines.append(flush_table(table_lines))
            table_lines = []

        # Headers
        if line.startswith("# "):
            html_lines.append(f'<h1 class="main-title">{line[2:].strip()}</h1>')
        elif line.startswith("## "):
            sec_title = line[3:].strip()
            page_break = ' class="section-heading page-break"' if any(sec_title.startswith(p) for p in ["4.", "6.", "7.", "8.", "10.", "12."]) else ' class="section-heading"'
            html_lines.append(f'<h2{page_break}>{sec_title}</h2>')
        elif line.startswith("### "):
            html_lines.append(f'<h3 class="subsection-heading">{line[4:].strip()}</h3>')
        elif line.startswith("#### "):
            html_lines.append(f'<h4 class="subsubsection-heading">{line[5:].strip()}</h4>')
        elif line.strip() == "---":
            html_lines.append('<hr class="divider">')
        elif line.strip() == "":
            html_lines.append("")
        elif line.startswith("- ") or line.startswith("* "):
            html_lines.append(f'<li class="bullet">{line[2:].strip()}</li>')
        elif re.match(r'^\d+\.\s', line):
            m = re.match(r'^\d+\.\s(.*)', line)
            html_lines.append(f'<li class="ordered">{m.group(1)}</li>')
        else:
            # Inline formatting
            processed = line
            processed = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed)
            processed = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed)
            processed = re.sub(r'`(.*?)`', r'<code>\1</code>', processed)
            html_lines.append(f'<p>{processed}</p>')

        i += 1

    if in_table:
        html_lines.append(flush_table(table_lines))

    return "\n".join(html_lines)

# Assemble HTML sections
from treatise_figures import get_fig1_svg, get_fig2_svg, get_fig3_svg, get_fig4_svg, get_fig5_svg

html_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project LEAN Archival Monograph v4.0 - Definitive Treatise</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<style>
@page {
    size: letter;
    margin: 18mm 15mm 18mm 15mm;
    @bottom-center {
        content: "Page " counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 8.5pt;
        color: #475569;
    }
    @top-right {
        content: "Project LEAN Definitive Archival Treatise v4.0";
        font-family: 'Times New Roman', Times, serif;
        font-size: 7.5pt;
        color: #64748b;
    }
}
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.2pt;
    line-height: 1.40;
    color: #0f172a;
    background-color: #fff;
    margin: 0;
    padding: 0;
}
.header-box {
    text-align: center;
    border-bottom: 2.5px solid #0f172a;
    padding-bottom: 12pt;
    margin-bottom: 14pt;
}
.series-tag {
    font-size: 8pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #475569;
    margin-bottom: 4pt;
}
h1.main-title {
    font-size: 15.5pt;
    font-weight: bold;
    line-height: 1.25;
    color: #0b1a30;
    margin: 4pt 0 6pt 0;
}
.subtitle {
    font-size: 9.8pt;
    font-style: italic;
    color: #334155;
    margin-bottom: 8pt;
}
.authors-block {
    font-size: 8.6pt;
    line-height: 1.35;
    color: #1e293b;
}
.abstract-box {
    background-color: #f8fafc;
    border-left: 3.5px solid #0284c7;
    border-right: 1px solid #e2e8f0;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    padding: 10pt 14pt;
    margin-bottom: 14pt;
    border-radius: 4px;
}
.abstract-box h3 {
    margin-top: 0;
    margin-bottom: 4pt;
    font-size: 10pt;
    color: #0369a1;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
h2.section-heading {
    font-size: 11.5pt;
    font-weight: bold;
    color: #0f172a;
    border-bottom: 1.2px solid #cbd5e1;
    padding-bottom: 2pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
}
h3.subsection-heading {
    font-size: 10pt;
    font-weight: bold;
    color: #1e293b;
    margin-top: 10pt;
    margin-bottom: 4pt;
}
h4.subsubsection-heading {
    font-size: 9.2pt;
    font-weight: bold;
    color: #334155;
    margin-top: 8pt;
    margin-bottom: 3pt;
}
p {
    margin-top: 0;
    margin-bottom: 5pt;
    text-align: justify;
}
li.bullet, li.ordered {
    margin-bottom: 2.5pt;
    margin-left: 14pt;
}
.table-wrap {
    margin: 8pt 0;
    width: 100%;
    overflow-x: auto;
}
table.academic-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.2pt;
    margin-bottom: 6pt;
}
table.academic-table th {
    background-color: #f1f5f9;
    border-top: 1.5px solid #334155;
    border-bottom: 1.5px solid #334155;
    padding: 3.5pt 5pt;
    text-align: left;
    font-weight: bold;
    color: #0f172a;
}
table.academic-table td {
    border-bottom: 1px solid #e2e8f0;
    padding: 3pt 5pt;
    color: #1e293b;
}
table.academic-table tr:last-child td {
    border-bottom: 1.5px solid #334155;
}
pre {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6pt 8pt;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 7.6pt;
    line-height: 1.28;
    overflow-x: auto;
    margin: 6pt 0;
}
code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.2pt;
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 2px;
}
pre code {
    background-color: transparent;
    padding: 0;
}
hr.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 10pt 0;
}
.figure-container {
    margin: 10pt 0;
    text-align: center;
    page-break-inside: avoid;
}
.page-break {
    page-break-before: always;
}
</style>
</head>
<body>
"""

html_body = []

# Title & Metadata
html_body.append("""
<div class="header-box">
    <div class="series-tag">Project LEAN Archival Series &bull; Definitive Archival Treatise v4.0 (No Page Limit Edition)</div>
    <h1 class="main-title">The Priced-Quanta Monograph: Decoupling Algorithmic Reasoning from Associative Memory for Sub-4-Bit Neural Compression</h1>
    <div class="subtitle">A Comprehensive Theoretical, Mechanistic, and Systems Treatise from Synthetic Universe Calibration to 7.6-Billion Parameter Consumer Hardware Deployment</div>
    <div class="authors-block">
        <strong>Mehardeep Singh</strong> &bull; Lead Author &amp; Principal Architect<br>
        <span style="color: #475569;">Advanced AI Systems Laboratory &bull; Project LEAN Research Initiative</span><br>
        <code>lean@antigravity.ai</code> &bull; Official Repository: <a href="https://github.com/Mehardeep-Singh-GG/LEAN">https://github.com/Mehardeep-Singh-GG/LEAN</a> &bull; October 2026
    </div>
</div>
""")

# Abstract
html_body.append('<div class="abstract-box">' + markdown_to_html_simple(s1_3.get_section_abstract()) + '</div>')

# Section 1 with Figure 1
html_body.append(markdown_to_html_simple(s1_3.get_section_1()))
html_body.append(get_fig1_svg())

# Section 2 with Figure 2
html_body.append(markdown_to_html_simple(s1_3.get_section_2()))
html_body.append(get_fig2_svg())

# Section 3
html_body.append(markdown_to_html_simple(s1_3.get_section_3()))

# Section 4 with Figure 5
html_body.append(markdown_to_html_simple(s4_5.get_section_4()))
html_body.append(get_fig5_svg())

# Section 5
html_body.append(markdown_to_html_simple(s4_5.get_section_5()))

# Section 6 with Figure 4
html_body.append(markdown_to_html_simple(s6_8.get_section_6()))
html_body.append(get_fig4_svg())

# Section 7
html_body.append(markdown_to_html_simple(s6_8.get_section_7()))

# Section 8 with Figure 3
html_body.append(markdown_to_html_simple(s6_8.get_section_8()))
html_body.append(get_fig3_svg())

# Section 9, 10, 11, 12
html_body.append(markdown_to_html_simple(s9_12.get_section_9()))
html_body.append(markdown_to_html_simple(s9_12.get_section_10()))
html_body.append(markdown_to_html_simple(s9_12.get_section_11()))
html_body.append(markdown_to_html_simple(s9_12.get_section_12()))

html_body.append("</body></html>")

full_html = html_head + "".join(html_body)

# Write to both manuscript_extended.html and manuscript.html
for target_html in [PAPER_DIR / "manuscript_extended.html", PAPER_DIR / "manuscript.html"]:
    with open(target_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Written {target_html} ({len(full_html)} characters).")

# 3. Compile PDF using Google Chrome Headless
chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
output_pdf_v3 = PAPER_DIR / "project_lean_monograph_v3.pdf"

print("Compiling PDF via Headless Chrome...")
chrome_cmd = [
    chrome_bin,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=12000",
    f"--print-to-pdf={output_pdf_v3}",
    str(PAPER_DIR / "manuscript_extended.html"),
]

res = subprocess.run(chrome_cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("Chrome stderr:", res.stderr)
    sys.exit(res.returncode)

# Mirror to v2 and project_lean_manuscript.pdf for universal access
shutil.copyfile(output_pdf_v3, PAPER_DIR / "project_lean_monograph_v2.pdf")
shutil.copyfile(output_pdf_v3, PAPER_DIR / "project_lean_manuscript.pdf")

# Verify page counts using pypdf
import pypdf
reader = pypdf.PdfReader(output_pdf_v3)
page_count = len(reader.pages)
file_size = os.path.getsize(output_pdf_v3)

print("\n" + "="*70)
print(f"✅ SUCCESS: Project LEAN Archival Monograph v4.0 Compiled!")
print(f"Total Pages: {page_count} pages")
print(f"File Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
print(f"Artifacts Updated:")
print(f"  - {PAPER_DIR / 'project_lean_monograph_v3.pdf'} ({page_count} pages)")
print(f"  - {PAPER_DIR / 'project_lean_monograph_v2.pdf'} ({page_count} pages)")
print(f"  - {PAPER_DIR / 'project_lean_manuscript.pdf'} ({page_count} pages)")
print(f"  - {PAPER_DIR / 'MANUSCRIPT_EXTENDED.md'}")
print(f"  - {PAPER_DIR / 'MANUSCRIPT.md'}")
print(f"  - {PAPER_DIR / 'manuscript_extended.html'}")
print(f"  - {PAPER_DIR / 'manuscript.html'}")
print("="*70 + "\n")
