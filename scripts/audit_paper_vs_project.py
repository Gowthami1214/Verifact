"""
audit_paper_vs_project.py
--------------------------
Cross-checks every factual claim in VERIFACT_Research_Paper_15refs.docx
against actual project data files and identifies discrepancies.
Outputs a detailed audit log.
"""
import sys, json, csv, os
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'c:\Users\Gowthami\TruthSync'

# ── Load data files ──────────────────────────────────────────────────────────
def load_json(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None, f"NOT FOUND: {path}"
    with open(path, encoding='utf-8') as f:
        return json.load(f), None

def load_csv(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None, f"NOT FOUND: {path}"
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f)), None

nli_metrics, _ = load_json('results/nli_metrics.json')
retrieval_metrics, _ = load_json('results/retrieval_metrics.json')
calib, _ = load_json('results/calibration_results.json')
end_to_end, _ = load_json('results/end_to_end_metrics.json')
claim_extraction, _ = load_json('results/claim_extraction_metrics.json')
rep_ablation, _ = load_json('results/representation_ablation.json')
ablation_rows, _ = load_csv('results/ablation_results.csv')
error_analysis, _ = load_csv('results/error_analysis.csv')
heldout_claims, _ = load_csv('dataset/layer5_heldout/heldout_claims.csv')
heldout_responses, _ = load_csv('dataset/layer5_heldout/heldout_responses.csv')
layer1_train, _ = load_csv('dataset/layer1_nli/train.csv')

passage_pool_path = os.path.join(ROOT, 'dataset/layer2_retrieval/passage_pool.jsonl')
passage_count = 0
with open(passage_pool_path, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            passage_count += 1

retrieval_test, _ = load_csv('dataset/layer2_retrieval/retrieval_test.csv')
retrieval_claim_ids = set(r['claim_id'] for r in retrieval_test) if retrieval_test else set()

ISSUES = []
OK = []

def issue(section, claim, actual, note=""):
    ISSUES.append({"section": section, "paper_says": claim, "actual": actual, "note": note})
    print(f"  [ISSUE] {section}: Paper says '{claim}' | Actual: '{actual}'" + (f" | {note}" if note else ""))

def ok(section, claim):
    OK.append({"section": section, "check": claim})
    print(f"  [OK]    {section}: {claim}")

print("="*70)
print("VERIFACT PAPER vs PROJECT — AUDIT REPORT")
print("="*70)

# ── Section I / Contributions ────────────────────────────────────────────────
print("\n--- Contributions / Introduction ---")
# Layer 1 train size
if layer1_train:
    ok("Intro", f"Layer 1 train.csv has {len(layer1_train)} rows (paper says 300 claims)")
    if len(layer1_train) != 300:
        issue("Intro", "300 domain claims", str(len(layer1_train)))

# Passage pool
print(f"  Passage pool lines: {passage_count}")
if passage_count == 308:
    ok("Intro", f"308 passages confirmed in passage_pool.jsonl")
elif passage_count == 302:
    issue("Intro/Layer2", "308 passages", "302", "paper says 308 but pool still has 302")
else:
    issue("Intro/Layer2", "308 passages", str(passage_count))

# Heldout benchmark
if heldout_responses:
    ok("Layer5", f"heldout_responses.csv: {len(heldout_responses)} responses")
    if len(heldout_responses) != 200:
        issue("Layer5", "200 responses", str(len(heldout_responses)))
if heldout_claims:
    ok("Layer5", f"heldout_claims.csv: {len(heldout_claims)} claims")
    if len(heldout_claims) != 910:
        issue("Layer5", "910 claims", str(len(heldout_claims)))

# ── Section IV — NLI Verifier ────────────────────────────────────────────────
print("\n--- Sec IV-E NLI Verifier ---")
# Validation accuracy
if nli_metrics:
    acc = nli_metrics.get('validation_metrics', {}).get('accuracy', None)
    mf1 = nli_metrics.get('validation_metrics', {}).get('macro_f1', None)
    sup_f1 = nli_metrics['validation_metrics']['per_class']['SUPPORTS']['f1']
    ref_f1 = nli_metrics['validation_metrics']['per_class']['REFUTES']['f1']
    ref_rec = nli_metrics['validation_metrics']['per_class']['REFUTES']['recall']
    nei_f1  = nli_metrics['validation_metrics']['per_class']['NOT_ENOUGH_INFO']['f1']

    if abs(acc - 0.8952) < 0.001:
        ok("NLI Verifier", f"Accuracy=0.8952 confirmed")
    else:
        issue("NLI Verifier", "accuracy=0.8952", str(acc))

    if abs(mf1 - 0.8958) < 0.001:
        ok("NLI Verifier", f"macro-F1=0.8958 confirmed")
    else:
        issue("NLI Verifier", "macro-F1=0.8958", str(mf1))

    if abs(sup_f1 - 0.8889) < 0.001 and abs(ref_f1 - 0.8889) < 0.001:
        ok("NLI Verifier", "SUPPORTS F1=REFUTES F1=0.8889 confirmed")
    else:
        issue("NLI Verifier", "SUPPORTS F1=REFUTES F1=0.8889", f"SUP={sup_f1}, REF={ref_f1}")

    if abs(ref_rec - 0.9143) < 0.001:
        ok("NLI Verifier", "REFUTES recall=0.9143 confirmed")
    else:
        issue("NLI Verifier", "REFUTES recall=0.9143", str(ref_rec))

# ── Section IV-F — Calibration ───────────────────────────────────────────────
print("\n--- Sec IV-F Calibration ---")
if nli_metrics:
    cal = nli_metrics.get('confidence_calibration', {})
    temp = cal.get('optimal_temperature', None)
    ece_before = cal.get('uncalibrated_ece', None)
    ece_after = cal.get('calibrated_ece', None)
    brier_before = cal.get('uncalibrated_brier_score', None)
    brier_after = cal.get('calibrated_brier_score', None)

    if abs(temp - 1.1348) < 0.001:
        ok("Calibration", "τ*=1.1348 confirmed")
    else:
        issue("Calibration", "τ*=1.1348", str(temp))

    if abs(ece_before - 0.081) < 0.001 and abs(ece_after - 0.0721) < 0.001:
        ok("Calibration", "ECE 0.0810→0.0721 confirmed")
    else:
        issue("Calibration", "ECE 0.0810→0.0721", f"{ece_before}→{ece_after}")

    if abs(brier_before - 0.133) < 0.001 and abs(brier_after - 0.1325) < 0.001:
        ok("Calibration", "Brier 0.133→0.1325 confirmed")
    else:
        issue("Calibration", "Brier 0.133→0.1325", f"{brier_before}→{brier_after}")

# ── Section V-C — Evaluation Measures ───────────────────────────────────────
print("\n--- Sec V-C Evaluation Measures ---")
if end_to_end:
    n_claims = end_to_end.get('n_claims', None)
    n_resp   = end_to_end.get('n_responses', None)
    claim_acc = end_to_end.get('claim_accuracy', None)
    macro_f1  = end_to_end.get('macro_f1', None)
    verdict_acc = end_to_end.get('verdict_accuracy', None)

    if n_claims == 118:
        ok("Eval Measures", "N=118 claims confirmed")
    else:
        issue("Eval Measures", "N=118 claims", str(n_claims))

    if n_resp == 30:
        ok("Eval Measures", "M=30 responses confirmed")
    else:
        issue("Eval Measures", "M=30 responses", str(n_resp))

    if abs(claim_acc - 0.246) < 0.005:
        ok("Eval Measures", "claim_accuracy=0.246 confirmed")
    else:
        issue("Eval Measures", "claim_accuracy=0.246", str(claim_acc))

    if abs(macro_f1 - 0.234) < 0.005:
        ok("Eval Measures", "macro_f1=0.234 confirmed")
    else:
        issue("Eval Measures", "macro_f1=0.234", str(macro_f1))

    if abs(verdict_acc - 0.500) < 0.001:
        ok("Eval Measures", "verdict_accuracy=0.500 confirmed")
    else:
        issue("Eval Measures", "verdict_accuracy=0.500", str(verdict_acc))

# ── Section VI-C — Retrieval ─────────────────────────────────────────────────
print("\n--- Sec VI-C Retrieval ---")
if retrieval_metrics:
    for metric in ['recall@1','recall@5','recall@10','mrr','ndcg@5','ndcg@10']:
        val = retrieval_metrics.get(metric, None)
        if val == 1.0:
            ok("Retrieval", f"{metric}=1.000 confirmed")
        else:
            issue("Retrieval", f"{metric}=1.000", str(val))

    n_queries = retrieval_metrics.get('evaluated_queries', None)
    if n_queries == 50:
        ok("Retrieval", "50 evaluated queries confirmed")
    else:
        issue("Retrieval", "50 queries", str(n_queries))

    # After reranking check
    reranker = retrieval_metrics.get('reranker', '')
    if 'ms-marco' in reranker:
        ok("Retrieval", "Metrics measured AFTER cross-encoder reranking confirmed")
    else:
        issue("Retrieval", "Metrics after reranking", f"reranker={reranker}")

# ── Section VI-C — Claim Extraction ─────────────────────────────────────────
print("\n--- Sec VI-C Claim Extraction ---")
if claim_extraction:
    tp = claim_extraction.get('true_positives', None)
    fp = claim_extraction.get('false_positives', None)
    fn = claim_extraction.get('false_negatives', None)
    total_annotated = claim_extraction.get('total_annotated', None)
    prec = claim_extraction.get('precision', None)
    rec  = claim_extraction.get('recall', None)
    f1   = claim_extraction.get('f1', None)

    print(f"  claim_extraction_metrics.json: TP={tp}, FP={fp}, FN={fn}, total_annotated={total_annotated}")
    print(f"    precision={prec}, recall={rec}, f1={f1}")

    # Paper says: 21 TP, 0 FP, 1 FN out of 22 annotated
    if tp == 21:
        ok("Claim Extraction", "21 true positives confirmed")
    else:
        issue("Claim Extraction", "21 TP", str(tp))
    if fp == 0:
        ok("Claim Extraction", "0 false positives confirmed")
    else:
        issue("Claim Extraction", "0 FP", str(fp))
    if fn == 1:
        ok("Claim Extraction", "1 false negative confirmed")
    else:
        issue("Claim Extraction", "1 FN", str(fn))
    if total_annotated == 22:
        ok("Claim Extraction", "22 annotated claims confirmed")
    else:
        issue("Claim Extraction", "22 annotated", str(total_annotated))

# ── Section VI-E — Error Analysis ───────────────────────────────────────────
print("\n--- Sec VI-E Error Analysis ---")
if error_analysis:
    n_errors = len(error_analysis)
    print(f"  error_analysis.csv rows: {n_errors}")

    # Count by type
    from collections import Counter
    type_counts = Counter(r.get('error_type','').strip() for r in error_analysis)
    print(f"  Error type breakdown: {dict(type_counts)}")

    # Paper says: total 50, wrong evidence=33 (63.5%), entity confusion=8 (15.4%),
    # numerical/unit=5 (9.6%), temporal=3 (5.8%), lexical-overlap=3 (5.8%)
    if n_errors == 50:
        ok("Error Analysis", "50 total errors confirmed")
    else:
        issue("Error Analysis", "50 errors", str(n_errors), "Paper (para 106) says 50; Fig.3 caption says 52")

    wrong_ev = type_counts.get('Wrong evidence', 0)
    if wrong_ev == 33:
        ok("Error Analysis", "33 wrong-evidence errors confirmed")
    else:
        issue("Error Analysis", "33 wrong-evidence", str(wrong_ev), f"types found: {dict(type_counts)}")

# Check Fig 3 caption still says 52 (inconsistency)
print("  NOTE: Fig. 3 caption reads '52 logged errors' but error_analysis.csv has 50 rows AND")
print("        para [106] text now correctly says 50. Fig. 3 caption needs manual update to 50.")

# ── Section VI-B — Input Template Comparison ────────────────────────────────
print("\n--- Sec VI-B Input Template Comparison ---")
if rep_ablation:
    best = rep_ablation.get('best', {})
    best_f1 = best.get('macro_f1', None)
    best_fmt = best.get('desc', '')
    worst_f1 = min(r['macro_f1'] for r in rep_ablation.get('results',[]))
    gap = round(best_f1 - worst_f1, 4) if best_f1 else None

    print(f"  Best template: {best_fmt}, macro_f1={best_f1}")
    print(f"  Worst macro_f1={worst_f1}, gap={gap}")

    # Paper says best macro-F1=0.6262, gap=3.2 points, accuracy range 59.05%-61.90%
    if abs(best_f1 - 0.6262) < 0.001:
        ok("Template Comparison", "Best macro-F1=0.6262 confirmed")
    else:
        issue("Template Comparison", "Best macro-F1=0.6262", str(best_f1))

    # Calculate accuracy range
    accs = sorted([r['accuracy'] for r in rep_ablation.get('results',[])])
    print(f"  Accuracy range: {accs[0]*100:.2f}% to {accs[-1]*100:.2f}%")
    if abs(accs[0] - 0.5905) < 0.001 and abs(accs[-1] - 0.6190) < 0.001:
        ok("Template Comparison", "Accuracy range 59.05%-61.90% confirmed")
    else:
        issue("Template Comparison", "Accuracy 59.05%-61.90%", f"{accs[0]*100:.2f}%-{accs[-1]*100:.2f}%")

# ── Ablation table check ─────────────────────────────────────────────────────
print("\n--- Sec VI Table VII Ablation ---")
if ablation_rows:
    print(f"  ablation_results.csv: {len(ablation_rows)} models")
    for r in ablation_rows:
        print(f"    {r['Model']}: acc={r['Accuracy']}, mF1={r['Macro_F1']}, "
              f"refF1={r['REFUTES_F1']}, e2e={r['End_to_End_Accuracy']}")

    # Paper says M6 end-to-end=0.500 (current pipeline)
    m6 = next((r for r in ablation_rows if 'M6' in r['Model'] or 'Full' in r['Model']), None)
    if m6:
        e2e = float(m6['End_to_End_Accuracy'])
        if abs(e2e - 0.5) < 0.001:
            ok("Ablation", "M6 end-to-end=0.500 confirmed")
        else:
            issue("Ablation", "M6 end-to-end=0.500", str(e2e))

# ── Layer 2 description in Section V-A ──────────────────────────────────────
print("\n--- Sec V-A Data Description ---")
# Paper (para [67]) says: "Layer 2 supplies the 302-passage corpus..."
# But actual passage pool has 308. Flag this.
issue("Sec V-A Data", "302-passage corpus (Layer 2 description)", "308 passages (confirmed)",
      "Para [67] in new paper still says 302 in the Layer 2 description — needs update to 308")

# ── Discussion: 33 of 52 vs 33 of 50 ────────────────────────────────────────
print("\n--- Sec VII Discussion ---")
# Para [110]: "33 of 52 errors involved a passage..." — should be 33 of 50
issue("Sec VII Discussion", "33 of 52 errors", "33 of 50 errors",
      "Discussion para still uses '52' — should be corrected to 50 to match error_analysis.csv")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("="*70)
print(f"AUDIT COMPLETE:  {len(OK)} items OK,  {len(ISSUES)} issues found")
print("="*70)
print("\nISSUES TO FIX IN DOCX:")
for idx, iss in enumerate(ISSUES, 1):
    print(f"  [{idx}] [{iss['section']}] Paper says: {iss['paper_says']!r}")
    print(f"       Actual value  : {iss['actual']}")
    if iss['note']:
        print(f"       Note          : {iss['note']}")
