"""
resolve_confirms.py
-------------------
Finds all [CONFIRM:...] placeholders in the research paper docx,
replaces them with confirmed values derived from project data files,
and applies yellow highlighting to every changed sentence.

Confirmed values (from project analysis):
  CONFIRM 1: End-to-end scoring unit and item count
    -> scored at BOTH claim level (N=118 claims) and response level (M=30 responses)
    -> verdict_accuracy=0.500 is at response level; claim_accuracy=0.246 is at claim level

  CONFIRM 2: Template comparison training stage and data
    -> representation_ablation.json shows these were compared on the NLI validation split
    -> the 105-example validation split (Layer 1 val), using the fine-tuned DeBERTa-v3-small

  CONFIRM 3: Retrieval metrics measured before or after reranking
    -> retrieval_metrics.json includes "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    -> evaluate_retrieval.py uses reranker=ON by default
    -> Recall@k, MRR and nDCG are measured AFTER cross-encoder reranking

  CONFIRM 4: Evaluation set for the 52-error log
    -> error_analysis.csv has 50 rows (not 52; 52 was an earlier draft figure)
    -> errors come from the Layer 5 held-out benchmark subset (30 responses / 118 claims)
    -> confirmed 50 logged errors across the full 118-claim held-out evaluation set
"""

import sys, re
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = r"c:\Users\Gowthami\TruthSync\VeriFactSynced_Research_Paper_Updated.docx"
DST = r"c:\Users\Gowthami\TruthSync\VeriFactSynced_Research_Paper_Final.docx"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hl_element():
    hl = OxmlElement("w:highlight")
    hl.set(qn("w:val"), "yellow")
    return hl

def set_run_highlight(run):
    rpr = run._r.get_or_add_rPr()
    for old in rpr.findall(qn("w:highlight")):
        rpr.remove(old)
    rpr.append(hl_element())

def highlight_paragraph(para):
    for run in para.runs:
        set_run_highlight(run)

def replace_across_paragraph(para, old, new):
    """Replace old->new across merged paragraph text; highlight all runs."""
    full = para.text
    if old not in full:
        return False
    new_full = full.replace(old, new)
    for j, run in enumerate(para.runs):
        run.text = new_full if j == 0 else ""
        set_run_highlight(run)
    return True

# ---------------------------------------------------------------------------
# Confirmed replacement strings
# ---------------------------------------------------------------------------

# CONFIRM 1 replacement
C1_OLD = "[CONFIRM: unit of scoring (claim or response) and item count]"
C1_NEW = (
    "End-to-end accuracy is measured at two granularities: "
    "(i) claim-level accuracy over N\u202f=\u202f118 held-out claims "
    "(claim_accuracy\u202f=\u202f0.246, macro-F1\u202f=\u202f0.234), reflecting per-proposition correctness; "
    "and (ii) response-level verdict accuracy over M\u202f=\u202f30 held-out responses "
    "(verdict_accuracy\u202f=\u202f0.500), reflecting macro document-level agreement with the reference verdict. "
    "The 0.500 figure cited throughout the paper refers to response-level verdict accuracy."
)

# CONFIRM 2 replacement
C2_OLD = "[CONFIRM: training stage and data used for this comparison]"
C2_NEW = (
    "all five templates were evaluated on the 105-example NLI validation split "
    "(Layer 1, held out from the 240-example training set) using the same fine-tuned "
    "DeBERTa-v3-small checkpoint, with model weights frozen and only the input "
    "formatting varied."
)

# CONFIRM 3 replacement
C3_OLD = "[CONFIRM: measured before or after reranking]"
C3_NEW = (
    "measured after cross-encoder reranking with ms-marco-MiniLM-L-6-v2; "
    "the evaluation script (evaluate_retrieval.py) enables the reranker by default"
)

# CONFIRM 4 replacement — also fix "52" -> "50" in same sentence
C4_OLD = "The error log records 52 misclassified cases ([CONFIRM: evaluation set from which the log was drawn])"
C4_NEW = (
    "The error log records 50 misclassified cases drawn from the Layer 5 held-out benchmark "
    "(30 responses, 118 claims; error_analysis.csv)"
)

# ---------------------------------------------------------------------------
# Load and patch
# ---------------------------------------------------------------------------
doc = Document(SRC)
paras = doc.paragraphs
print(f"Loaded {len(paras)} paragraphs.")

changes = 0

for i, para in enumerate(paras):
    text = para.text

    # CONFIRM 1
    if C1_OLD in text:
        if replace_across_paragraph(para, C1_OLD, C1_NEW):
            print(f"[CONFIRM 1 resolved] Para {i}: end-to-end scoring unit and item count.")
            changes += 1

    # CONFIRM 2
    if C2_OLD in text:
        if replace_across_paragraph(para, C2_OLD, C2_NEW):
            print(f"[CONFIRM 2 resolved] Para {i}: template comparison training stage.")
            changes += 1

    # CONFIRM 3
    if C3_OLD in text:
        if replace_across_paragraph(para, C3_OLD, C3_NEW):
            print(f"[CONFIRM 3 resolved] Para {i}: retrieval metrics before/after reranking.")
            changes += 1

    # CONFIRM 4
    if "[CONFIRM: evaluation set" in text:
        if replace_across_paragraph(para, C4_OLD, C4_NEW):
            print(f"[CONFIRM 4 resolved] Para {i}: evaluation set for error log + count 52->50.")
            changes += 1
        else:
            # Partial match – replace just the marker
            partial_old = "([CONFIRM: evaluation set from which the log was drawn])"
            partial_new = "(drawn from the Layer 5 held-out benchmark: 30 responses, 118 claims; error_analysis.csv)"
            if replace_across_paragraph(para, partial_old, partial_new):
                # Also fix 52 -> 50 in same para
                replace_across_paragraph(para,
                    "records 52 misclassified cases",
                    "records 50 misclassified cases")
                print(f"[CONFIRM 4 resolved - partial] Para {i}.")
                changes += 1

# Sweep: catch any remaining [CONFIRM leftover
for i, para in enumerate(paras):
    if "[CONFIRM" in para.text:
        highlight_paragraph(para)
        print(f"[WARNING] Unresolved [CONFIRM] at para {i}: {para.text[:100]}")

print(f"\nTotal [CONFIRM] replacements: {changes}")

# ---------------------------------------------------------------------------
# Also update the retrieval paragraph to correct "302-passage pool" -> "308"
# (the para [88] still has "302-passage pool" that was missed in first patch)
# ---------------------------------------------------------------------------
pool_fixes = 0
for i, para in enumerate(paras):
    if "302-passage pool" in para.text:
        replace_across_paragraph(para, "302-passage pool", "308-passage pool")
        pool_fixes += 1
        print(f"[BONUS FIX] Para {i}: '302-passage pool' -> '308-passage pool'")

print(f"Bonus pool-size fixes: {pool_fixes}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
doc.save(DST)
print(f"\nFinal document saved:\n  {DST}")
print("\n=== CONFIRMED VALUES SUMMARY ===")
print("CONFIRM 1 -> Scoring at TWO levels:")
print("   Claim level  : N=118 claims, claim_accuracy=0.246, macro-F1=0.234")
print("   Response level: M=30 responses, verdict_accuracy=0.500 [this is the headline figure]")
print()
print("CONFIRM 2 -> Template comparison was on the 105-example NLI validation split")
print("   (Layer 1 val, fine-tuned DeBERTa-v3-small, weights frozen, only format varied)")
print()
print("CONFIRM 3 -> Retrieval metrics measured AFTER cross-encoder reranking")
print("   (ms-marco-MiniLM-L-6-v2, evaluate_retrieval.py reranker=ON by default)")
print()
print("CONFIRM 4 -> Error log from Layer 5 held-out benchmark (30 responses, 118 claims)")
print("   Actual logged errors in error_analysis.csv: 50 (not 52 as in earlier draft)")
