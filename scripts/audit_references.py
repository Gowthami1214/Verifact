"""
audit_references.py
---------------------
Extracts all [N] in-text citation usages and the reference list from
VERIFACT_Research_Paper_15refs_Corrected.docx, then cross-checks:
  1. Every in-text [N] maps to a reference that exists in the list
  2. Every reference in the list is actually cited somewhere
  3. The context of each citation matches the actual reference content
"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

DOC = r'c:\Users\Gowthami\TruthSync\VERIFACT_Research_Paper_15refs_Corrected.docx'
doc = Document(DOC)

# ── Extract full text with paragraph index ────────────────────────────────────
para_texts = [(i, p.text) for i, p in enumerate(doc.paragraphs) if p.text.strip()]

# ── Extract reference list ────────────────────────────────────────────────────
refs = {}  # {num: full_text}
in_refs = False
for i, txt in para_texts:
    if 'References' in txt and len(txt.strip()) < 20:
        in_refs = True
        continue
    if in_refs:
        m = re.match(r'^\[(\d+)\]\s+(.+)', txt.strip())
        if m:
            refs[int(m.group(1))] = m.group(2).strip()

print(f"References found: {sorted(refs.keys())}")
print()
for k in sorted(refs.keys()):
    print(f"  [{k}] {refs[k][:100]}")

# ── Extract in-text citations ─────────────────────────────────────────────────
citations = {}  # {num: [(para_idx, sentence_snippet)]}
for i, txt in para_texts:
    matches = re.findall(r'\[(\d+)\]', txt)
    for m in matches:
        n = int(m)
        if n not in citations:
            citations[n] = []
        # Get surrounding context (~80 chars around the citation)
        pos = txt.find(f'[{m}]')
        start = max(0, pos-60)
        end = min(len(txt), pos+60)
        snippet = txt[start:end].replace('\n', ' ').replace('\t', ' ')
        citations[n].append((i, snippet))

print()
print("="*70)
print("IN-TEXT CITATION USAGE:")
print("="*70)
for n in sorted(citations.keys()):
    print(f"\n[{n}] Used {len(citations[n])} time(s):")
    for para_i, snip in citations[n]:
        print(f"   Para {para_i}: ...{snip}...")

# ── Check 1: cited refs that don't exist in list ─────────────────────────────
print()
print("="*70)
print("CHECK 1: In-text citations NOT in reference list")
print("="*70)
missing_refs = sorted(set(citations.keys()) - set(refs.keys()))
if missing_refs:
    for n in missing_refs:
        print(f"  [ISSUE] [{n}] cited in text but NOT in reference list!")
        for para_i, snip in citations[n]:
            print(f"    Para {para_i}: ...{snip}...")
else:
    print("  All in-text citations map to existing references. OK")

# ── Check 2: references never cited ─────────────────────────────────────────
print()
print("="*70)
print("CHECK 2: References in list but NEVER cited in text")
print("="*70)
uncited = sorted(set(refs.keys()) - set(citations.keys()))
if uncited:
    for n in uncited:
        print(f"  [ISSUE] [{n}] is in the reference list but never cited in text!")
        print(f"    Ref: {refs[n][:100]}")
else:
    print("  All references are cited at least once. OK")

# ── Check 3: Context vs Reference content match ───────────────────────────────
print()
print("="*70)
print("CHECK 3: Citation context vs reference content audit")
print("="*70)

# Known mapping: what each reference is about and expected context keywords
EXPECTED_CONTEXT = {
    1: {"author": "Ji", "topic": "hallucination", "keywords": ["hallucination", "survey", "natural language"]},
    2: {"author": "Min", "topic": "FActScore", "keywords": ["FActScore", "atomic", "factual precision", "long form"]},
    3: {"author": "Thorne", "topic": "FEVER", "keywords": ["FEVER", "fact extraction", "verification", "Wikipedia"]},
    4: {"author": "Guo", "topic": "calibration neural networks", "keywords": ["calibrat", "neural network", "temperature"]},
    5: {"author": "Gururangan", "topic": "annotation artifacts NLI", "keywords": ["artifact", "NLI", "annotation", "hypothesis"]},
    6: {"author": "Guo/Schlichtkrull", "topic": "survey automated fact-checking", "keywords": ["fact-check", "survey", "automated"]},
    7: {"author": "Manakul", "topic": "SelfCheckGPT", "keywords": ["SelfCheck", "hallucination", "black-box", "LLM"]},
    8: {"author": "Wei", "topic": "long-form factuality LLMs", "keywords": ["long-form", "factuality", "LLM"]},
    9: {"author": "Laban", "topic": "SummaC NLI summarization", "keywords": ["SummaC", "NLI", "summarization", "inconsistency"]},
    10: {"author": "Reimers", "topic": "Sentence-BERT", "keywords": ["Sentence-BERT", "embedding", "sentence", "BERT"]},
    11: {"author": "Nogueira", "topic": "passage re-ranking BERT cross-encoder", "keywords": ["re-rank", "cross-encoder", "BERT", "passage"]},
    12: {"author": "Liu", "topic": "RoBERTa", "keywords": ["RoBERTa", "BERT", "pretrain"]},
    13: {"author": "He", "topic": "DeBERTaV3", "keywords": ["DeBERTa", "disentangled", "ELECTRA"]},
    14: {"author": "Desai", "topic": "calibration pretrained transformers", "keywords": ["calibrat", "pretrained", "transformer"]},
    15: {"author": "Naeini", "topic": "ECE calibration BBQ", "keywords": ["calibrat", "ECE", "confidence", "binning", "quantile"]},
}

issues = []
ok_list = []

for n in sorted(EXPECTED_CONTEXT.keys()):
    if n not in refs:
        issues.append(f"[{n}] NOT IN REFERENCE LIST")
        continue
    ref_text = refs[n].lower()
    exp = EXPECTED_CONTEXT[n]

    # Check reference text contains author name
    author_ok = exp["author"].lower().split('/')[0].lower() in ref_text
    # Check at least one keyword is in citation contexts
    ctx_texts = " ".join(snip for _, snip in citations.get(n, [])).lower()
    ctx_ok = any(kw.lower() in ctx_texts for kw in exp["keywords"])

    if not author_ok:
        issues.append(f"[{n}] Reference author mismatch! Expected '{exp['author']}' in: {refs[n][:80]}")
    if citations.get(n) and not ctx_ok:
        issues.append(f"[{n}] Citation context mismatch! Expected topic '{exp['topic']}' but context: {ctx_texts[:100]}")
    if author_ok:
        ok_list.append(f"[{n}] Author '{exp['author']}' confirmed in reference")

if issues:
    print("  ISSUES:")
    for iss in issues:
        print(f"    [MISMATCH] {iss}")
else:
    print("  No mismatches found.")

print()
print(f"  {len(ok_list)} references validated OK.")

# ── Check 4: Cross-check specific important citations ─────────────────────────
print()
print("="*70)
print("CHECK 4: Specific semantic citation accuracy checks")
print("="*70)

specific_checks = [
    # (ref_num, expected_claim_in_context, description)
    (10, ["bi-encoder", "sentence encoder", "embedding", "MiniLM", "all-MiniLM"], "Ref [10] Reimers/Sentence-BERT used for bi-encoder/embedding"),
    (11, ["cross-encoder", "rerank", "ms-marco"], "Ref [11] Nogueira used for cross-encoder reranking"),
    (12, ["RoBERTa", "open-domain", "MNLI", "pretrained"], "Ref [12] Liu/RoBERTa cited for RoBERTa-large-MNLI"),
    (13, ["DeBERTa", "deberta", "fine-tun", "benchmark"], "Ref [13] He/DeBERTaV3 cited for fine-tuned DeBERTa"),
    (4,  ["calibrat", "temperature", "Guo"], "Ref [4] Guo used for temperature scaling/calibration"),
    (15, ["ECE", "calibrat", "confidence bin"], "Ref [15] Naeini used for ECE / calibration bins"),
    (3,  ["FEVER", "Layer 0", "adapt", "NLI backbone"], "Ref [3] FEVER used for Layer 0 NLI backbone adaptation"),
    (14, ["calibrat", "in-domain", "pretrained", "reasonable"], "Ref [14] Desai cited for in-domain calibration claim"),
]

for ref_num, keywords, desc in specific_checks:
    ctx = " ".join(snip for _, snip in citations.get(ref_num, [])).lower()
    ref_txt = refs.get(ref_num, "").lower()
    found = any(kw.lower() in ctx for kw in keywords)
    if found:
        print(f"  [OK]    {desc}")
    else:
        print(f"  [ISSUE] {desc}")
        print(f"          Actual context: '{ctx[:120]}'")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print(f"  Total references defined: {len(refs)}")
print(f"  Total unique in-text citations: {len(citations)}")
print(f"  Missing in ref list: {missing_refs}")
print(f"  Never cited: {uncited}")
print(f"  Semantic issues: {len(issues)}")
