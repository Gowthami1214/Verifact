import sys, json, csv, os
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

doc = Document(r'c:\Users\Gowthami\TruthSync\VERIFACT_Research_Paper_15refs.docx')
print(f"Total paragraphs: {len(doc.paragraphs)}")
print()
for i, p in enumerate(doc.paragraphs):
    if p.text.strip():
        print(f"[{i}] STYLE={p.style.name!r:28s} | {p.text}")
