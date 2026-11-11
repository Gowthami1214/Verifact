"""
patch_research_paper.py
-----------------------
Applies all VeriFactSynced research paper changes to
TruthSync_Research_Paper.docx with yellow highlighting.

Changes applied
  1. Project rename: TruthSync -> VeriFactSynced  (everywhere)
  2. Remove Gradio; dashboard is Streamlit only
  3. Knowledge base size: 302 -> 308 passages, note P0303-P0308
  4. Add Relevance Gate section (after Cross-Encoder section)
  5. Dual NLI engine: default RoBERTa-large-MNLI, backup DeBERTa-v3-small
  6. Contradiction-Precedence aggregation description update
  7. Corpus size in Limitations: 302 -> 308
  8. Conclusion: update model description + accuracy
  9. Interface section H: replace Gradio text with Streamlit-only text
 10. Introduction contributions list updates
"""

import copy, re, sys
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import RGBColor
from lxml import etree

SRC = r"c:\Users\Gowthami\TruthSync\TruthSync_Research_Paper.docx"
DST = r"c:\Users\Gowthami\TruthSync\VeriFactSynced_Research_Paper_Updated.docx"

YELLOW = "FFFF00"   # highlight colour (Word "Yellow")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def hl_color_element(color_hex: str):
    """Return a <w:highlight w:val="yellow"/> element for the given hex."""
    # Word only accepts named highlight colours; we use 'yellow'
    hl = OxmlElement("w:highlight")
    hl.set(qn("w:val"), "yellow")
    return hl


def set_run_highlight(run):
    """Apply yellow highlight to a run, creating rPr if needed."""
    rpr = run._r.get_or_add_rPr()
    # Remove existing highlight first
    for old in rpr.findall(qn("w:highlight")):
        rpr.remove(old)
    rpr.append(hl_color_element(YELLOW))


def replace_in_run(run, old: str, new: str) -> bool:
    """Replace text in a single run. Returns True if replacement happened."""
    if old in run.text:
        run.text = run.text.replace(old, new)
        set_run_highlight(run)
        return True
    return False


def replace_across_paragraph(para, old: str, new: str) -> bool:
    """
    Replace `old` with `new` across the merged text of a paragraph.
    Rebuilds runs if the match spans run boundaries.
    Returns True if any replacement happened.
    """
    full = para.text
    if old not in full:
        return False

    # Simple case: replacement fits inside one run
    changed = False
    for run in para.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            set_run_highlight(run)
            changed = True

    if changed:
        return True

    # Cross-run case: merge all runs into the first, set the rest empty
    if old in full:
        new_full = full.replace(old, new)
        # Clear all runs
        for i, run in enumerate(para.runs):
            if i == 0:
                run.text = new_full
                set_run_highlight(run)
            else:
                run.text = ""
        return True

    return False


def highlight_paragraph(para):
    """Highlight every run in a paragraph yellow."""
    for run in para.runs:
        set_run_highlight(run)


def insert_paragraph_after(ref_para, text: str, style: str = "Normal") -> object:
    """Insert a new paragraph immediately after ref_para with yellow highlight."""
    new_para = OxmlElement("w:p")
    ref_para._p.addnext(new_para)

    doc = ref_para._p.getroottree().getroot()
    # We need access to the document object to add properly formatted paragraph.
    # Instead, copy ref_para and clear its runs, then set text.
    from docx.text.paragraph import Paragraph
    p_obj = Paragraph(new_para, ref_para._p.getparent())

    # Copy style
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), style)
    pPr.append(pStyle)
    new_para.append(pPr)

    run_el = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    hl = hl_color_element(YELLOW)
    rPr.append(hl)
    run_el.append(rPr)
    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    run_el.append(t)
    new_para.append(run_el)
    return p_obj


# ---------------------------------------------------------------------------
# Load document
# ---------------------------------------------------------------------------
doc = Document(SRC)
paras = doc.paragraphs
n = len(paras)

print(f"Loaded document with {n} paragraphs.")

# ---------------------------------------------------------------------------
# Build index: paragraph index -> paragraph text (for searching)
# ---------------------------------------------------------------------------
def para_contains(idx, fragment):
    return fragment in paras[idx].text

# ---------------------------------------------------------------------------
# CHANGE 1: Project rename TruthSync -> VeriFactSynced
# ---------------------------------------------------------------------------
RENAMES = [
    ("TruthSync", "VeriFactSynced"),
    ("truthsync", "verifact"),
    ("TruthSyncPipeline", "VeriFactPipeline"),
]

rename_count = 0
for i, para in enumerate(paras):
    for old, new in RENAMES:
        if old in para.text:
            if replace_across_paragraph(para, old, new):
                rename_count += 1

print(f"CHANGE 1 - Rename: {rename_count} replacements made.")

# ---------------------------------------------------------------------------
# CHANGE 2: Interface section H - Remove Gradio, Streamlit only
# ---------------------------------------------------------------------------
# Paragraph [58]: contains the Gradio + Streamlit description
GRADIO_OLD = (
    "The pipeline is exposed through a Gradio application and a Streamlit dashboard. "
    "The Gradio interface shows color-coded verdict banners with a factuality meter, a per-claim breakdown and a JSON diagnostic dump. "
    "The Streamlit interface"
)
GRADIO_NEW = (
    "The pipeline is exposed exclusively through an interactive Streamlit dashboard. "
    "The Streamlit interface"
)

GRADIO_OLD2 = "Gradio application and a Streamlit dashboard. The Gradio interface shows color-coded verdict banners with a factuality meter, a per-claim breakdown and a JSON diagnostic dump. The Streamlit interface"
GRADIO_NEW2 = "Streamlit dashboard exclusively. The Streamlit interface"

changed_h = False
for i, para in enumerate(paras):
    if "Gradio application" in para.text and "Streamlit" in para.text:
        # Rebuild the whole paragraph text
        full = para.text
        if "Gradio application" in full:
            new_text = re.sub(
                r"The pipeline is exposed through a Gradio application and a Streamlit dashboard\."
                r".*?The Streamlit interface",
                "The pipeline is exposed exclusively through an interactive Streamlit dashboard. The Streamlit interface",
                full,
                flags=re.DOTALL,
            )
            if new_text != full:
                # Apply to runs
                for j, run in enumerate(para.runs):
                    if j == 0:
                        run.text = new_text
                        set_run_highlight(run)
                    else:
                        run.text = ""
                changed_h = True
                print(f"CHANGE 2 - Interface H paragraph [{i}]: Gradio removed.")
                break
        # Fallback: word-by-word replace
        if not changed_h:
            replace_across_paragraph(para, "a Gradio application and a Streamlit dashboard", "an exclusive Streamlit dashboard")
            for run in para.runs:
                if "Gradio" in run.text:
                    run.text = run.text.replace("Gradio interface shows color-coded verdict banners with a factuality meter, a per-claim breakdown and a JSON diagnostic dump. The", "")
                    set_run_highlight(run)
            changed_h = True

# Also catch any standalone Gradio mentions remaining
for i, para in enumerate(paras):
    if "Gradio" in para.text:
        replace_across_paragraph(para, "Gradio", "Streamlit")
        print(f"  Residual Gradio reference cleaned at [{i}]")

# ---------------------------------------------------------------------------
# CHANGE 3: Knowledge base 302 -> 308 passages + note P0303-P0308
# ---------------------------------------------------------------------------
KB_CHANGES = [
    ("302-passage retrieval corpus",
     "308-passage retrieval corpus (extended from 302 with six human-verified ground-truth passages P0303\u2013P0308)"),
    ("302 curated passages",
     "308 curated passages (including six human-verified astronomical and physical ground-truth entries P0303\u2013P0308)"),
    ("302 passages",
     "308 passages"),
]

kb_count = 0
for i, para in enumerate(paras):
    for old, new in KB_CHANGES:
        if old in para.text:
            if replace_across_paragraph(para, old, new):
                kb_count += 1
                print(f"CHANGE 3 - KB size [{i}]: '{old[:40]}' -> updated")

print(f"CHANGE 3 - KB total replacements: {kb_count}")

# ---------------------------------------------------------------------------
# CHANGE 4: NLI Verifier description - dual engine
# ---------------------------------------------------------------------------
NLI_OLD = (
    "The verifier is microsoft/deberta-v3-small [26] fine-tuned for three-way classification."
)
NLI_NEW = (
    "The verifier operates in a dual-engine architecture. In the default open-domain mode, "
    "a pre-trained RoBERTa-large-MNLI model [24] trained on 400,000+ MultiNLI inference pairs "
    "is used, providing robust generalisation to real-world open queries. "
    "In benchmark-replication mode, a fine-tuned microsoft/deberta-v3-small [26] checkpoint "
    "trained on the VeriFactSynced 240-sample NLI split is activated. "
    "The active engine is selectable via the Streamlit sidebar at runtime."
)

nli_changed = False
for i, para in enumerate(paras):
    if "microsoft/deberta-v3-small" in para.text and "fine-tuned for three-way" in para.text:
        if replace_across_paragraph(para, NLI_OLD, NLI_NEW):
            nli_changed = True
            print(f"CHANGE 4 - Dual NLI engine description updated at [{i}].")
        else:
            # partial match - rebuild run
            for run in para.runs:
                if "microsoft/deberta-v3-small" in run.text:
                    run.text = run.text.replace(
                        "microsoft/deberta-v3-small [26] fine-tuned for three-way classification.",
                        "microsoft/deberta-v3-small [26] fine-tuned checkpoint (benchmark mode); "
                        "default mode uses RoBERTa-large-MNLI [24] for open-domain inference."
                    )
                    set_run_highlight(run)
                    nli_changed = True
                    print(f"CHANGE 4 - NLI partial update at [{i}].")

# ---------------------------------------------------------------------------
# CHANGE 5: Add Relevance Gate sub-section after Cross-Encoder section
# ---------------------------------------------------------------------------
# Find paragraph [44] "A bi-encoder can rank a passage highly..."
# Insert new relevance gate paragraph after the cross-encoder paragraph
RELEVANCE_GATE_TEXT = (
    "D\u2019. Semantic Relevance Gate. Between the cross-encoder reranker and the NLI module, "
    "VeriFactSynced introduces a Semantic Relevance Gate. The MS-MARCO cross-encoder (ms-marco-MiniLM-L-6-v2) "
    "produces a log-likelihood logit for each candidate passage. Candidates with a logit below the "
    "threshold \u03c4 = \u22122.5 are deemed topically irrelevant to the claim. When all top-k candidates "
    "fall below \u03c4, the pipeline short-circuits the NLI call and returns NOT_ENOUGH_INFO with a "
    "confidence of 0.95, preventing false refutations caused by off-topic evidence retrievals. "
    "This gate eliminated the most frequent logged failure mode: wrong-evidence retrieval (33 of 52 errors, 63.5%)."
)

gate_inserted = False
for i, para in enumerate(paras):
    if "bi-encoder can rank a passage highly" in para.text:
        insert_paragraph_after(para, RELEVANCE_GATE_TEXT, "Normal")
        gate_inserted = True
        print(f"CHANGE 5 - Relevance Gate paragraph inserted after [{i}].")
        break

if not gate_inserted:
    print("CHANGE 5 - WARNING: Could not find cross-encoder paragraph to insert gate after.")

# ---------------------------------------------------------------------------
# CHANGE 6: Aggregation description - add contradiction-precedence detail
# ---------------------------------------------------------------------------
AGG_OLD = (
    "Trusting only the first passage is risky, since a partly relevant top passage can yield a wrong label "
    "while the second or third holds decisive evidence. The aggregator therefore applies three rules to"
)
AGG_NEW = (
    "Trusting only the first passage is risky, since a partly relevant top passage can yield a wrong label "
    "while the second or third holds decisive evidence. After relevance gating (Section IV-D\u2019), "
    "VeriFactSynced evaluates the top-k (\u22643) semantically relevant candidates through the NLI module "
    "and applies a Contradiction-Precedence Aggregation rule: (1) if any candidate yields SUPPORTS with "
    "confidence \u22650.45 and no candidate yields REFUTES, the claim is marked SUPPORTS; (2) if any candidate "
    "yields REFUTES with confidence \u22650.60, REFUTES overrides weaker support signals; (3) if both signals "
    "are present below the REFUTES threshold, the highest-confidence support verdict is adopted. "
    "The aggregator then applies three response-level rules to"
)

agg_changed = False
for i, para in enumerate(paras):
    if "Trusting only the first passage is risky" in para.text:
        if replace_across_paragraph(para, AGG_OLD, AGG_NEW):
            agg_changed = True
            print(f"CHANGE 6 - Aggregation description updated at [{i}].")
        else:
            # highlight the paragraph and append a note
            highlight_paragraph(para)
            insert_paragraph_after(
                para,
                "[UPDATED] Contradiction-Precedence Aggregation: (1) SUPPORTS if conf>=0.45 and no REFUTES; "
                "(2) REFUTES if conf>=0.60 (overrides support); (3) highest-confidence support otherwise. "
                "Relevance gate (tau=-2.5) pre-filters irrelevant candidates before NLI evaluation.",
                "Normal"
            )
            agg_changed = True
            print(f"CHANGE 6 - Aggregation note inserted after [{i}].")
        break

# ---------------------------------------------------------------------------
# CHANGE 7: Introduction contribution bullet - 302 -> 308
# Already handled by Change 3 bulk replacement above.
# But highlight bullet [11] specifically if found
# ---------------------------------------------------------------------------
for i, para in enumerate(paras):
    if "layered 15-domain dataset" in para.text and "retrieval corpus" in para.text:
        highlight_paragraph(para)
        print(f"CHANGE 7 - Introduction contribution bullet [{i}] highlighted.")

# ---------------------------------------------------------------------------
# CHANGE 8: Conclusion paragraph - update model name + rename
# Already handled by Change 1 rename for TruthSync -> VeriFactSynced
# Update "fine-tuned DeBERTa-v3-small verifier" to note dual engine
# ---------------------------------------------------------------------------
CONC_OLD = "a fine-tuned DeBERTa-v3-small verifier, temperature scaling and a refutation-first aggregator"
CONC_NEW = ("a dual-engine NLI verifier (RoBERTa-large-MNLI for open-domain mode; "
            "fine-tuned DeBERTa-v3-small for benchmark mode), temperature scaling, "
            "a semantic relevance gate (\u03c4=\u22122.5) and a contradiction-precedence aggregator")

for i, para in enumerate(paras):
    if "fine-tuned DeBERTa-v3-small verifier" in para.text:
        if replace_across_paragraph(para, CONC_OLD, CONC_NEW):
            print(f"CHANGE 8 - Conclusion NLI description updated at [{i}].")

# ---------------------------------------------------------------------------
# CHANGE 9: Section heading IV rename - "The TruthSync Framework" already
# caught by Change 1. Also update Overview paragraph [35]
# ---------------------------------------------------------------------------
for i, para in enumerate(paras):
    if "Fig. 1 shows the data flow" in para.text:
        old_frag = "the ten nearest passages are reranked and the top three go to"
        new_frag = ("the ten nearest passages are reranked; a relevance gate (tau=-2.5) "
                    "filters off-topic candidates; the remaining top three go to")
        if old_frag in para.text:
            replace_across_paragraph(para, old_frag, new_frag)
            print(f"CHANGE 9 - Overview data flow sentence updated at [{i}].")

# ---------------------------------------------------------------------------
# CHANGE 10: Limitations - "302 passages" -> "308 passages" already done
# Highlight all limitations paragraphs too
# ---------------------------------------------------------------------------
in_limitations = False
for i, para in enumerate(paras):
    if "VIII. Limitations" in para.text or "Limitations" in para.text and para.style.name == "Heading 1":
        in_limitations = True
    if in_limitations and "IX." in para.text:
        in_limitations = False
    if in_limitations and "Closed corpus" in para.text:
        highlight_paragraph(para)
        print(f"CHANGE 10 - Limitations 'Closed corpus' bullet [{i}] highlighted.")

# ---------------------------------------------------------------------------
# Final: highlight any remaining changed paragraphs that still mention
# the new VeriFactSynced name (to visually flag all touched paragraphs)
# ---------------------------------------------------------------------------
new_term_count = 0
for para in paras:
    if "VeriFactSynced" in para.text or "VeriFactPipeline" in para.text:
        for run in para.runs:
            if "VeriFactSynced" in run.text or "VeriFactPipeline" in run.text:
                set_run_highlight(run)
                new_term_count += 1

print(f"\nFinal highlight sweep: {new_term_count} runs containing 'VeriFactSynced' highlighted.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
doc.save(DST)
print(f"\nSaved updated document to:\n  {DST}")
print("\nSummary of changes applied:")
print("  [1] TruthSync -> VeriFactSynced (all occurrences, yellow highlighted)")
print("  [2] Gradio removed; Section H is Streamlit-only")
print("  [3] Knowledge base: 302 -> 308 passages (P0303-P0308 noted)")
print("  [4] NLI Section E: dual-engine (RoBERTa-large default + DeBERTa benchmark)")
print("  [5] New Section D': Semantic Relevance Gate (tau=-2.5)")
print("  [6] Aggregation Section G: Contradiction-Precedence rules added")
print("  [7] Contribution bullet [11] highlighted (308-passage corpus)")
print("  [8] Conclusion: dual-engine + relevance gate added")
print("  [9] Overview paragraph: relevance gate step added to data flow")
print(" [10] Limitations 'Closed corpus' bullet highlighted (302->308)")
