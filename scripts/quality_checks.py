"""
quality_checks.py -- Phase 5
Runs automated quality checks across all generated dataset layers.
Checks for:
  1. Label-keyword leakage in claim and response text
  2. Exact duplicate claims within and across splits
  3. Missing values in required columns
  4. Label distribution balance
  5. Claims-per-response distribution
  6. Evidence role distribution
  7. Cross-split entity overlap
Writes results to dataset/metadata/leakage_report.json
"""

import os, csv, json, re, sys
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

LEAKAGE_KEYWORDS = [
    r"\bSUPPORTS\b", r"\bREFUTES\b", r"\bNOT_ENOUGH_INFO\b",
    r"does not establish", r"proposition is incorrect",
    r"reference material confirms", r"contrary to the claim",
    r"consistent with the statement", r"cited record does not",
    r"available evidence does not", r"\[scenario", r"\[Evaluation scenario",
    r"\[label=", r"\[class=", r"\[NEI\]",
    r"reference material is consistent", r"factual record supports",
    r"documented account supports", r"established description confirms",
]

LEAKAGE_RE = [re.compile(p, re.IGNORECASE) for p in LEAKAGE_KEYWORDS]

issues = []
report = {}

def check_leakage(text, field, record_id):
    for pattern, compiled in zip(LEAKAGE_KEYWORDS, LEAKAGE_RE):
        if compiled.search(text or ""):
            issues.append({"file": field, "id": record_id, "issue": "label_leakage", "pattern": pattern, "snippet": text[:120]})

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

print("=== TruthSync Quality Checks ===\n")

# ── 1. Layer 1 NLI: train, val, test ────────────────────────────────────────
print("[Layer 1 NLI]")
l1_all_claims = []
for split in ("train.csv", "validation.csv", "test.csv"):
    path = os.path.join(BASE, "layer1_nli", split)
    rows = read_csv(path)
    print(f"  {split}: {len(rows)} rows")
    labels = Counter(r.get("label","") for r in rows)
    print(f"    labels: {dict(labels)}")
    missing = sum(1 for r in rows if not r.get("claim") or not r.get("evidence") or not r.get("label"))
    if missing:
        issues.append({"file": f"layer1_nli/{split}", "issue": "missing_values", "count": missing})
        print(f"    WARNING: {missing} rows with missing values")
    for r in rows:
        check_leakage(r.get("claim",""), f"layer1_nli/{split}:claim", r.get("id","?"))
        l1_all_claims.append(r.get("claim","").lower().strip())

l1_dupes = len(l1_all_claims) - len(set(l1_all_claims))
print(f"  Cross-split duplicate claims: {l1_dupes}")
if l1_dupes:
    issues.append({"file": "layer1_nli", "issue": "duplicate_claims", "count": l1_dupes})
report["layer1_nli"] = {"checked": True, "cross_split_duplicates": l1_dupes}

# ── 2. Layer 2 Retrieval ─────────────────────────────────────────────────────
print("\n[Layer 2 Retrieval]")
pool = read_jsonl(os.path.join(BASE, "layer2_retrieval", "passage_pool.jsonl"))
print(f"  passage_pool.jsonl: {len(pool)} passages")
passage_ids = {p["passage_id"] for p in pool}

cands = read_csv(os.path.join(BASE, "layer2_retrieval", "claim_candidates.csv"))
print(f"  claim_candidates.csv: {len(cands)} rows")
role_dist = Counter(r.get("evidence_role","") for r in cands)
print(f"    evidence roles: {dict(role_dist)}")
for r in cands:
    check_leakage(r.get("claim_text",""), "layer2_retrieval/claim_candidates:claim_text", r.get("claim_id","?"))

ret_test = read_csv(os.path.join(BASE, "layer2_retrieval", "retrieval_test.csv"))
print(f"  retrieval_test.csv: {len(ret_test)} rows")
report["layer2_retrieval"] = {"passages": len(pool), "candidates": len(cands), "retrieval_test_rows": len(ret_test)}

# ── 3. Layer 3 Claim Extraction ──────────────────────────────────────────────
print("\n[Layer 3 Claim Extraction]")
raw = read_jsonl(os.path.join(BASE, "layer3_claim_extraction", "raw_ai_outputs.jsonl"))
print(f"  raw_ai_outputs.jsonl: {len(raw)} documents")
spans = read_jsonl(os.path.join(BASE, "layer3_claim_extraction", "claim_spans.jsonl"))
print(f"  claim_spans.jsonl: {len(spans)} sentence annotations")
factual = sum(1 for s in spans if s.get("is_factual_claim"))
print(f"    factual claims: {factual} ({100*factual//max(len(spans),1)}%)")
test_spans = read_jsonl(os.path.join(BASE, "layer3_claim_extraction", "claim_extraction_test.jsonl"))
print(f"  claim_extraction_test.jsonl: {len(test_spans)} annotations")
for doc in raw:
    check_leakage(doc.get("generated_text",""), "layer3_claim_extraction/raw_ai_outputs:text", doc.get("doc_id","?"))
report["layer3_claim_extraction"] = {"train_docs": len(raw), "train_annotations": len(spans),
                                      "factual_annotations": factual, "test_annotations": len(test_spans)}

# ── 4. Layer 5 Held-Out Benchmark ────────────────────────────────────────────
print("\n[Layer 5 Heldout]")
responses = read_csv(os.path.join(BASE, "layer5_heldout", "heldout_responses.csv"))
print(f"  heldout_responses.csv: {len(responses)} responses")

claims = read_csv(os.path.join(BASE, "layer5_heldout", "heldout_claims.csv"))
print(f"  heldout_claims.csv: {len(claims)} claims")
label_dist = Counter(c.get("label","") for c in claims)
print(f"    label distribution: {dict(label_dist)}")
halluc_dist = Counter(c.get("hallucination_type","") for c in claims if c.get("label") == "REFUTES")
print(f"    hallucination types (REFUTES): {dict(halluc_dist)}")

# Leakage check on all claims and responses
leakage_hits = 0
for c in claims:
    before = len(issues)
    check_leakage(c.get("claim_text",""), "layer5_heldout/claims:claim_text", c.get("claim_id","?"))
    leakage_hits += len(issues) - before
for r in responses:
    check_leakage(r.get("generated_text",""), "layer5_heldout/responses:generated_text", r.get("response_id","?"))

print(f"    label-leakage keyword hits: {leakage_hits}")

# Exact duplicate claims in Layer 5
l5_claim_texts = [c.get("claim_text","").lower().strip() for c in claims]
l5_dupes = len(l5_claim_texts) - len(set(l5_claim_texts))
print(f"    exact duplicate claims: {l5_dupes}")

# Cross-split: Layer 5 vs Layer 1
l5_claim_set = set(l5_claim_texts)
l1_claim_set = set(l1_all_claims)
cross_overlap = len(l5_claim_set & l1_claim_set)
print(f"    cross-split overlap with Layer 1: {cross_overlap}")

# Claims per response distribution
cpr_counter = Counter(r.get("total_claims","0") for r in responses)
print(f"    claims-per-response: {dict(sorted(cpr_counter.items()))}")

# Mixed-label responses
mixed = sum(1 for r in responses if int(r.get("supports_count","0") or 0) + int(r.get("refutes_count","0") or 0) + int(r.get("nei_count","0") or 0) > 0
            and not (int(r.get("supports_count","0") or 0) == int(r.get("total_claims","0") or 0)
                     or int(r.get("refutes_count","0") or 0) == int(r.get("total_claims","0") or 0)
                     or int(r.get("nei_count","0") or 0) == int(r.get("total_claims","0") or 0)))
print(f"    mixed-label responses: {mixed}/{len(responses)} ({100*mixed//max(len(responses),1)}%)")

evidence = read_csv(os.path.join(BASE, "layer5_heldout", "heldout_evidence.csv"))
print(f"  heldout_evidence.csv: {len(evidence)} rows")
erole_dist = Counter(e.get("evidence_role","") for e in evidence)
print(f"    evidence roles: {dict(erole_dist)}")

if l5_dupes:
    issues.append({"file": "layer5_heldout/claims", "issue": "duplicate_claims", "count": l5_dupes})
if cross_overlap:
    issues.append({"file": "layer5_heldout", "issue": "cross_split_overlap", "count": cross_overlap})

report["layer5_heldout"] = {
    "responses": len(responses),
    "claims": len(claims),
    "label_distribution": dict(label_dist),
    "duplicate_claims": l5_dupes,
    "cross_split_overlap_with_layer1": cross_overlap,
    "mixed_label_responses": mixed,
    "mixed_label_pct": round(100*mixed/max(len(responses),1), 1),
    "leakage_keyword_hits": leakage_hits,
    "evidence_rows": len(evidence),
    "evidence_role_distribution": dict(erole_dist),
}

# ── 5. Summary ───────────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"Total issues found: {len(issues)}")
critical = [i for i in issues if i.get("issue") in ("label_leakage", "cross_split_overlap")]
print(f"  Critical (leakage/overlap): {len(critical)}")
print(f"  Other (missing values, duplicates): {len(issues)-len(critical)}")

report["issues"] = issues
report["summary"] = {
    "total_issues": len(issues),
    "critical_issues": len(critical),
}

out_path = os.path.join(BASE, "metadata", "leakage_report.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=True)
print(f"\nLeakage report written to: metadata/leakage_report.json")

if critical:
    print("\nCRITICAL ISSUES -- review the leakage_report.json before using this dataset.")
    sys.exit(1)
else:
    print("\nAll critical checks passed.")
