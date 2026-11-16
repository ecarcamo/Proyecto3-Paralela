from pathlib import Path

import markdown
from weasyprint import HTML

REPORT_DIR = Path(__file__).resolve().parent

STYLE = """
@page { size: Letter; margin: 2cm 1.8cm; @bottom-center { content: counter(page); color: #52514e; font-size: 9pt; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; line-height: 1.45; color: #0b0b0b; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
h2 { font-size: 13pt; margin-top: 16pt; border-bottom: 1px solid #e4e3df; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin-top: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 8.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #e4e3df; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f2ef; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; background: #f3f2ef; padding: 0 2pt; }
pre { background: #f3f2ef; padding: 6pt; font-size: 8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
p.caption, em { color: #52514e; }
blockquote { border-left: 3px solid #2a78d6; margin: 6pt 0; padding: 2pt 8pt; color: #52514e; }
"""


def main():
    source = (REPORT_DIR / "informe.md").read_text(encoding="utf-8")
    body = markdown.markdown(source, extensions=["tables", "fenced_code"])
    html = f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'><style>{STYLE}</style></head><body>{body}</body></html>"
    HTML(string=html, base_url=str(REPORT_DIR)).write_pdf(REPORT_DIR / "informe.pdf")
    print(REPORT_DIR / "informe.pdf")


if __name__ == "__main__":
    main()
