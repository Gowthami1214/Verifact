"""
reorganize.py — Phase 1
Moves existing dataset files into the new TruthSync directory structure.
Archives the invalid Layer 3 data.
"""

import os
import shutil

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

DIRS = [
    "layer0_nli",
    "layer1_nli",
    "layer2_retrieval",
    "layer3_claim_extraction",
    "layer4_end_to_end",
    "layer5_heldout",
    "archive",
    "metadata",
]

for d in DIRS:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)
    print(f"  created: {d}/")

# ── Layer 0: FEVER corpus ────────────────────────────────────────────────────
src = os.path.join(BASE, "layer1", "train.jsonl")
dst = os.path.join(BASE, "layer0_nli", "train.jsonl")
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy2(src, dst)
    print(f"  copied layer1/train.jsonl -> layer0_nli/train.jsonl")
else:
    print(f"  layer0_nli/train.jsonl already present or src missing")

# ── Layer 1 NLI: custom NLI dataset ─────────────────────────────────────────
for fname in ("train.csv", "validation.csv", "test.csv"):
    src = os.path.join(BASE, "layer2", fname)
    dst = os.path.join(BASE, "layer1_nli", fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  copied layer2/{fname} -> layer1_nli/{fname}")

# ── Layer 4 end-to-end: existing AI-generated evaluation ────────────────────
src = os.path.join(BASE, "layer2", "ai_generated_evaluation.csv")
dst = os.path.join(BASE, "layer4_end_to_end", "ai_generated_evaluation.csv")
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy2(src, dst)
    print(f"  copied ai_generated_evaluation.csv -> layer4_end_to_end/")

# ── Archive: retire the invalid Layer 3 ─────────────────────────────────────
for fname in (
    "layer3_heldout_evaluation.csv",
    "layer3_evaluation_metadata.csv",
    "layer3_evaluation_statistics.json",
):
    src = os.path.join(BASE, "layer3", fname)
    dst = os.path.join(BASE, "archive", fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  archived layer3/{fname} -> archive/{fname}")

# Write archive note
note_path = os.path.join(BASE, "archive", "ARCHIVE_NOTE.txt")
if not os.path.exists(note_path):
    with open(note_path, "w") as f:
        f.write(
            "ARCHIVE — Layer 3 (original)\n"
            "==============================\n"
            "These files were retired because:\n"
            "  1. Generated texts are template-assembled, not realistic AI outputs.\n"
            "  2. Claim text explicitly embeds label-signaling phrases.\n"
            "  3. [scenario L3_XXXX] identifiers leak claim identity.\n"
            "  4. Evidence is 75% duplicate across claims.\n"
            "  5. The same evidence passage is reused for all three label variants per scenario.\n"
            "\n"
            "Replaced by: layer5_heldout/\n"
            "Archive date: 2026-09-13\n"
        )
    print("  wrote archive/ARCHIVE_NOTE.txt")

print("\nReorganization complete.")
