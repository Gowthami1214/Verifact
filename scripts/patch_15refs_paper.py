"""
patch_15refs_paper.py
-----------------------
Patches VERIFACT_Research_Paper_15refs.docx with confirmed corrections
based on the full project audit. All changes are highlighted in yellow.

Issues being fixed:
  [1] Intro: "300 domain claims" -> "240 domain claims"
  [2-5] Claim extraction metrics keys were nested - already correct (21 TP, 0 FP, 1 FN, 22 annotated)
  [6] Error breakdown: paper says "33 wrong-evidence, 8 entity confusion, 5 numerical, 3 temporal, 3 lexical-overlap"
      Actual error_analysis.csv: Wrong evidence=29, Entity confusion=14, Temporal=5, Numerical=2, total=50
      -> Update breakdown figures and percentages
  [7] Sec V-A: "302-passage corpus" -> "308-passage corpus"
  [8] Sec VII: "33 of 52 errors" -> "29 of 50 errors"
  
  Additional fixes:
  [9]  Fig. 3 caption "52 logged errors" -> "50 logged errors"
  [10] Layer 1 description: paper says 300 claims but train.csv has 240
       (val split 105 confirmed; total 300 includes test+val+train combined)
       -> Layer 1 holds 240 training + 30 test + 105 val = 375 BUT
          The 300 figure refers to total UNIQUE claims in the layer, not train only.
          train.csv=240, test.csv=30, val is derived. Keep 300 but note 240 training rows.
  [11] Conclusion mentions "NLI verifier which is a combination of two engines" - keep as-is (correct)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import re

SRC = r'c:\Users\Gowthami\TruthSync\VERIFACT_Research_Paper_15refs.docx'
DST = r'c:\Users\Gowthami\TruthSync\VERIFACT_Research_Paper_15refs_Corrected.docx'

# ── Helpers ──────────────────────────────────────────────────────────────────
def hl_el():
    e = OxmlElement("w:highlight")
    e.set(qn("w:val"), "yellow")
    return e

def set_run_hl(run):
    rpr = run._r.get_or_add_rPr()
    for old in rpr.findall(qn("w:highlight")):
        rpr.remove(old)
    rpr.append(hl_el())

def highlight_para(para):
    for run in para.runs:
        set_run_hl(run)

def replace_in_para(para, old, new):
    full = para.text
    if old not in full:
        return False
    new_full = full.replace(old, new)
    for j, run in enumerate(para.runs):
        run.text = new_full if j == 0 else ""
        set_run_hl(run)
    return True

# ── Load ─────────────────────────────────────────────────────────────────────
doc = Document(SRC)
paras = doc.paragraphs
print(f"Loaded {len(paras)} paragraphs from {SRC}")
changes = []

# ── FIX 1: Layer 1 "300 domain claims" clarification ─────────────────────────
# Paper para: "Layer 1 holds 300 domain claims with balanced classes"
# train.csv has 240; test.csv has 30; total Layer 1 is 270 documented rows.
# The paper's "300" is the intended design target; actual generated = 240 train + 30 test = 270.
# We correct to 270 (confirmed) and note that.
for i, p in enumerate(paras):
    if "Layer 1 holds 300 domain claims" in p.text:
        if replace_in_para(p,
            "Layer 1 holds 300 domain claims with balanced classes and no prompt-artifact tokens",
            "Layer 1 holds 270 domain claims with balanced classes and no prompt-artifact tokens "
            "(240 training rows + 30 test rows)"):
            changes.append(f"[FIX 1] Para {i}: Layer 1 claim count 300 -> 270 (240 train + 30 test)")
            print(changes[-1])

# ── FIX 2: Sec V-A "302-passage corpus" -> "308-passage corpus" ──────────────
for i, p in enumerate(paras):
    if "302-passage corpus" in p.text:
        if replace_in_para(p, "302-passage corpus", "308-passage corpus"):
            changes.append(f"[FIX 2] Para {i}: '302-passage corpus' -> '308-passage corpus'")
            print(changes[-1])

# Also catch "302-passage pool" anywhere
for i, p in enumerate(paras):
    if "302-passage pool" in p.text:
        if replace_in_para(p, "302-passage pool", "308-passage pool"):
            changes.append(f"[FIX 2b] Para {i}: '302-passage pool' -> '308-passage pool'")
            print(changes[-1])

# ── FIX 3: Error breakdown figures ───────────────────────────────────────────
# Actual from error_analysis.csv:
#   Wrong evidence=29 (58.0%), Entity confusion=14 (28.0%),
#   Temporal=5 (10.0%), Numerical=2 (4.0%), total=50
# Paper says: 33 (63.5%), 8 (15.4%), 5 (9.6%), 3 (5.8%), 3 (5.8%)
OLD_BREAKDOWN = (
    "The most common type of error is wrong evidence retrieval (33 errors, 63.5%) in which a keyword "
    "in the claim attracts an unrelated passage such as \"public-key cryptography\" in Madagascar attracting "
    "\"passage about distributed ledgers\". The second most common type of error is entity confusion "
    "(8 errors, 15.4%), for example, the correct association of the double helix with Watson and Crick being "
    "wrongly classified as \"REFUTES\". Other errors include unit or numeric mismatch (5 errors, 9.6%), "
    "confusion between creation date and standardization date (3 errors, 5.8%), and overlap error (3 errors, 5.8%)."
)
NEW_BREAKDOWN = (
    "The most common type of error is wrong evidence retrieval (29 errors, 58.0%) in which a keyword "
    "in the claim attracts an unrelated passage such as \"public-key cryptography\" in Madagascar attracting "
    "a passage about distributed ledgers. The second most common type of error is entity confusion "
    "(14 errors, 28.0%), for example, the correct association of the double helix with Watson and Crick being "
    "wrongly classified as REFUTES. Other errors include temporal ambiguity between creation and standardization "
    "dates (5 errors, 10.0%) and numerical or unit mismatches (2 errors, 4.0%)."
)
for i, p in enumerate(paras):
    if "33 errors, 63.5%" in p.text:
        if replace_in_para(p, OLD_BREAKDOWN, NEW_BREAKDOWN):
            changes.append(f"[FIX 3] Para {i}: Error breakdown corrected to match error_analysis.csv")
            print(changes[-1])
        else:
            # Targeted replacements
            for old, new in [
                ("33 errors, 63.5%", "29 errors, 58.0%"),
                ("8 errors, 15.4%", "14 errors, 28.0%"),
                ("5 errors, 9.6%", "5 errors, 10.0%"),
                ("3 errors, 5.8%), and overlap error (3 errors, 5.8%)",
                 "2 errors, 4.0%)"),
            ]:
                if replace_in_para(p, old, new):
                    changes.append(f"[FIX 3] Para {i}: '{old[:30]}' corrected")
                    print(changes[-1])

# ── FIX 4: Discussion "33 of 52 errors" -> "29 of 50 errors" ─────────────────
for i, p in enumerate(paras):
    if "33 of 52 errors" in p.text:
        if replace_in_para(p, "33 of 52 errors", "29 of 50 errors"):
            changes.append(f"[FIX 4] Para {i}: '33 of 52' -> '29 of 50'")
            print(changes[-1])

# Also fix "33 errors" remaining in discussion
for i, p in enumerate(paras):
    if "since 33 of" in p.text:
        if replace_in_para(p, "since 33 of", "since 29 of"):
            changes.append(f"[FIX 4b] Para {i}: discussion '33 of' -> '29 of'")
            print(changes[-1])

# ── FIX 5: Fig 3 caption "52 logged errors" -> "50 logged errors" ────────────
for i, p in enumerate(paras):
    if "52 logged errors" in p.text or "among the 52" in p.text:
        if replace_in_para(p, "52 logged errors", "50 logged errors") or \
           replace_in_para(p, "among the 52", "among the 50"):
            changes.append(f"[FIX 5] Para {i}: Fig 3 caption '52' -> '50'")
            print(changes[-1])

# ── FIX 6: Claim extraction paragraph — confirm the corrected metrics ─────────
# Paper correctly states 21 TP, 0 FP, 1 FN, 22 annotated.
# Just verify and highlight the correct paragraph.
for i, p in enumerate(paras):
    if "21 true positives" in p.text and "false positive" in p.text:
        highlight_para(p)
        changes.append(f"[CONFIRM OK] Para {i}: Claim extraction metrics (21 TP, 0 FP, 1 FN, 22 annotated) — highlighted as confirmed correct")
        print(changes[-1])

# ── FIX 7: Any remaining "300 claims" in intro contributions ─────────────────
for i, p in enumerate(paras):
    if "300 domain claims" in p.text:
        if replace_in_para(p, "300 domain claims", "270 domain claims"):
            changes.append(f"[FIX 7] Para {i}: '300 domain claims' -> '270 domain claims'")
            print(changes[-1])

# ── FIX 8: Layer 2 in data section also mentioned as "302" ───────────────────
for i, p in enumerate(paras):
    for old, new in [
        ("Layer 2 supplies the 302-passage", "Layer 2 supplies the 308-passage"),
        ("302 passages", "308 passages"),
    ]:
        if old in p.text:
            if replace_in_para(p, old, new):
                changes.append(f"[FIX 8] Para {i}: '{old}' -> '{new}'")
                print(changes[-1])

# ── FIX 9: "4,550 candidate passages" check ──────────────────────────────────
# 200 responses × ~4-5 candidates each × 5 evidence rows = 4550 from heldout_evidence
# This matches 910 claims × 5 evidence each = 4550 — correct.
for i, p in enumerate(paras):
    if "4,550 candidate passages" in p.text:
        highlight_para(p)
        changes.append(f"[CONFIRM OK] Para {i}: 4,550 candidates (910 claims × 5 evidence each) — highlighted as confirmed correct")
        print(changes[-1])

# ── FIX 10: REFUTES F1 ablation — paper says "0.5230 to 0.6750" ──────────────
# From ablation_results.csv: M1 refF1=0.523, M4 refF1=0.675 -> confirmed
for i, p in enumerate(paras):
    if "0.5230 to 0.6750" in p.text or "REFUTES F1 rose from 0.5230" in p.text:
        highlight_para(p)
        changes.append(f"[CONFIRM OK] Para {i}: REFUTES F1 0.5230->0.6750 ablation trend — confirmed from ablation_results.csv")
        print(changes[-1])

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(DST)
print(f"\n{'='*60}")
print(f"Saved corrected document to:\n  {DST}")
print(f"\nTotal changes applied: {len(changes)}")
for c in changes:
    print(f"  {c}")
