"""
generate_metadata.py -- Phase 6
Generates dataset_statistics.json and data_dictionary.json for the TruthSync dataset.
Must be run AFTER all other generation scripts have completed.
"""

import os, csv, json
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

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

def count_jsonl(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

stats = {}

# ── Layer 0 NLI (FEVER) ───────────────────────────────────────────────────────
l0_count = count_jsonl(os.path.join(BASE, "layer0_nli", "train.jsonl"))
stats["layer0_nli"] = {
    "description": "FEVER-format NLI corpus. Evidence is Wikipedia sentence IDs; raw text requires the FEVER Wikipedia dump (~13 GB).",
    "file": "layer0_nli/train.jsonl",
    "records": l0_count,
    "label_distribution_approx": {"SUPPORTS": 80035, "REFUTES": 29775, "NOT_ENOUGH_INFO": 35639},
    "human_verified": False,
    "source_note": "FEVER dataset (Thorne et al., 2018), redistributed subset",
    "caution": "Evidence fields contain Wikipedia doc_id + sentence_index only. Passage text must be recovered from the FEVER Wikipedia dump.",
}
print(f"Layer 0 NLI: {l0_count} records")

# ── Layer 1 NLI (custom) ──────────────────────────────────────────────────────
l1_stats = {}
all_l1 = []
for split in ("train.csv", "validation.csv", "test.csv"):
    rows = read_csv(os.path.join(BASE, "layer1_nli", split))
    labels = dict(Counter(r.get("label","") for r in rows))
    topics = dict(Counter(r.get("topic","") for r in rows))
    difficulty = dict(Counter(r.get("difficulty","") for r in rows))
    claim_types = dict(Counter(r.get("claim_type","") for r in rows))
    l1_stats[split] = {
        "records": len(rows), "labels": labels, "topics": topics,
        "difficulty": difficulty, "claim_types": claim_types,
    }
    all_l1.extend(rows)
    print(f"Layer 1 NLI {split}: {len(rows)} records, labels={labels}")

stats["layer1_nli"] = {
    "description": "Custom NLI dataset with hand-crafted claim/evidence pairs. Used for fine-tuning NLI backbone after pre-training on Layer 0.",
    "splits": l1_stats,
    "total_records": len(all_l1),
    "human_verified": False,
    "source_note": "synthetic_llm_authored",
    "note": "Preamble phrases in claim text ('A verifier is asked to determine whether...') are NLI-style prefixes; they do not encode the gold label.",
}

# ── Layer 2 Retrieval ─────────────────────────────────────────────────────────
pool = read_jsonl(os.path.join(BASE, "layer2_retrieval", "passage_pool.jsonl"))
pool_topics = dict(Counter(p.get("topic","") for p in pool))
pool_sources = dict(Counter(p.get("source_category","") for p in pool))
cands = read_csv(os.path.join(BASE, "layer2_retrieval", "claim_candidates.csv"))
cand_roles = dict(Counter(r.get("evidence_role","") for r in cands))
ret_test = read_csv(os.path.join(BASE, "layer2_retrieval", "retrieval_test.csv"))

stats["layer2_retrieval"] = {
    "description": "Passage pool and retrieval candidate sets for evidence retrieval training and evaluation.",
    "passage_pool": {"records": len(pool), "topic_distribution": pool_topics, "source_distribution": pool_sources},
    "claim_candidates": {"records": len(cands), "candidates_per_claim": 5, "role_distribution": cand_roles},
    "retrieval_test": {"records": len(ret_test), "candidates_per_claim": 10},
    "human_verified": False,
    "source_note": "synthetic_llm_authored",
    "limitation": "Passage pool is ~300 passages, not 50,000. Suitable for prototype retrieval demo; not for production-scale ANN benchmarking.",
}
print(f"Layer 2 Retrieval: {len(pool)} passages, {len(cands)} candidates, {len(ret_test)} test rows")

# ── Layer 3 Claim Extraction ──────────────────────────────────────────────────
raw = read_jsonl(os.path.join(BASE, "layer3_claim_extraction", "raw_ai_outputs.jsonl"))
spans = read_jsonl(os.path.join(BASE, "layer3_claim_extraction", "claim_spans.jsonl"))
test_spans = read_jsonl(os.path.join(BASE, "layer3_claim_extraction", "claim_extraction_test.jsonl"))
raw_topics = dict(Counter(d.get("topic","") for d in raw))
factual_count = sum(1 for s in spans if s.get("is_factual_claim"))
non_factual = len(spans) - factual_count

stats["layer3_claim_extraction"] = {
    "description": "Sentence-level claim extraction annotations on realistic AI-generated paragraphs.",
    "raw_ai_outputs": {"records": len(raw), "topic_distribution": raw_topics},
    "claim_spans_train": {
        "records": len(spans),
        "factual_claim_sentences": factual_count,
        "non_factual_sentences": non_factual,
        "factual_pct": round(100*factual_count/max(len(spans),1), 1),
    },
    "claim_extraction_test": {"records": len(test_spans)},
    "human_verified": False,
    "source_note": "synthetic_llm_authored",
}
print(f"Layer 3 Claim Extraction: {len(raw)} docs, {len(spans)} annotations")

# ── Layer 4 End-to-End ────────────────────────────────────────────────────────
l4_rows = read_csv(os.path.join(BASE, "layer4_end_to_end", "ai_generated_evaluation.csv"))
l4_labels = dict(Counter(r.get("ground_truth_label","") for r in l4_rows))
l4_texts = len(set(r.get("generated_text","") for r in l4_rows))

stats["layer4_end_to_end"] = {
    "description": "End-to-end evaluation set with realistic AI-generated responses containing actual hallucinations. Highest-quality data in the dataset.",
    "file": "layer4_end_to_end/ai_generated_evaluation.csv",
    "records": len(l4_rows),
    "unique_generated_texts": l4_texts,
    "label_distribution": l4_labels,
    "human_verified": False,
    "source_note": "synthetic_llm_authored",
}
print(f"Layer 4 End-to-End: {len(l4_rows)} records")

# ── Layer 5 Heldout ───────────────────────────────────────────────────────────
responses = read_csv(os.path.join(BASE, "layer5_heldout", "heldout_responses.csv"))
claims5 = read_csv(os.path.join(BASE, "layer5_heldout", "heldout_claims.csv"))
evidence5 = read_csv(os.path.join(BASE, "layer5_heldout", "heldout_evidence.csv"))

resp_topics = dict(Counter(r.get("topic","") for r in responses))
resp_styles = dict(Counter(r.get("response_style","") for r in responses))
resp_verdicts = dict(Counter(r.get("response_verdict","") for r in responses))
c5_labels = dict(Counter(c.get("label","") for c in claims5))
c5_types = dict(Counter(c.get("claim_type","") for c in claims5))
c5_diff = dict(Counter(c.get("difficulty","") for c in claims5))
c5_halluc = dict(Counter(c.get("hallucination_type","") for c in claims5 if c.get("label")=="REFUTES"))
ev5_roles = dict(Counter(e.get("evidence_role","") for e in evidence5))

cpr = Counter(r.get("total_claims","") for r in responses)
mixed = sum(1 for r in responses
    if not (r.get("supports_count","0")==r.get("total_claims","0")
            or r.get("refutes_count","0")==r.get("total_claims","0")
            or r.get("nei_count","0")==r.get("total_claims","0")))

stats["layer5_heldout"] = {
    "description": "Held-out evaluation benchmark. Realistic AI-generated responses with no label-signaling text. Never use for training.",
    "heldout_responses": {
        "records": len(responses),
        "topic_distribution": resp_topics,
        "style_distribution": resp_styles,
        "verdict_distribution": resp_verdicts,
        "claims_per_response_distribution": dict(sorted(cpr.items())),
        "mixed_label_responses": mixed,
        "mixed_label_pct": round(100*mixed/max(len(responses),1), 1),
    },
    "heldout_claims": {
        "records": len(claims5),
        "label_distribution": c5_labels,
        "claim_type_distribution": c5_types,
        "difficulty_distribution": c5_diff,
        "hallucination_type_distribution_refutes_only": c5_halluc,
    },
    "heldout_evidence": {
        "records": len(evidence5),
        "candidates_per_claim": 5,
        "role_distribution": ev5_roles,
    },
    "human_verified": False,
    "source_note": "synthetic_llm_authored",
    "usage_restriction": "Must not be used during model training, embedding training, retrieval training, reranking training, hyperparameter tuning, or prompt optimization. Final evaluation only.",
}
print(f"Layer 5 Heldout: {len(responses)} responses, {len(claims5)} claims, {len(evidence5)} evidence rows")

# ── Overall ───────────────────────────────────────────────────────────────────
total_claims = len(all_l1) + len(claims5)
stats["overall"] = {
    "total_nli_records": len(all_l1) + l0_count,
    "total_custom_claims": total_claims,
    "total_passages": len(pool),
    "total_retrieval_candidates": len(cands) + len(ret_test),
    "total_ce_documents": len(raw),
    "total_heldout_responses": len(responses),
    "layers": ["layer0_nli", "layer1_nli", "layer2_retrieval", "layer3_claim_extraction", "layer4_end_to_end", "layer5_heldout"],
    "dataset_version": "2.0",
    "created": "2026-09-13",
    "human_verified": False,
    "source_note": "Layers 1-5 are synthetic_llm_authored. Layer 0 is FEVER corpus.",
}

# ── Write statistics ──────────────────────────────────────────────────────────
stats_path = os.path.join(BASE, "metadata", "dataset_statistics.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=True)
print(f"\ndataset_statistics.json written.")

# ── Data Dictionary ───────────────────────────────────────────────────────────
data_dict = {
    "layer0_nli/train.jsonl": {
        "id": "Integer FEVER claim ID",
        "verifiable": "'VERIFIABLE' or 'NOT VERIFIABLE'",
        "label": "SUPPORTS | REFUTES | NOT ENOUGH INFO",
        "claim": "The factual claim text",
        "evidence": "List of [ann_id, evidence_id, Wikipedia_title, sentence_index] tuples. Raw text must be recovered from the FEVER Wikipedia dump.",
    },
    "layer1_nli/{train,validation,test}.csv": {
        "id": "Integer row ID",
        "claim": "Factual claim with NLI-style preamble. Preamble does not encode the label.",
        "evidence": "Evidence passage text (1-3 sentences)",
        "label": "SUPPORTS | REFUTES | NOT_ENOUGH_INFO",
        "source": "Source category: Wikipedia | scientific | government | official_organization | other",
        "topic": "One of 15 topic categories",
        "difficulty": "easy | medium | hard",
        "claim_type": "Semantic type of claim: person | date | location | number | organization | event | definition | cause_effect | comparison | scientific | relationship",
    },
    "layer2_retrieval/passage_pool.jsonl": {
        "passage_id": "Unique passage identifier (P0001 format)",
        "title": "Short title for the passage content",
        "passage_text": "1-3 sentence factual passage",
        "source_category": "Wikipedia | scientific | government | official_organization | other",
        "topic": "One of 15 topic categories",
        "human_verified": "false -- synthetic data",
        "source_note": "synthetic_llm_authored",
    },
    "layer2_retrieval/claim_candidates.csv": {
        "claim_id": "ID of the claim (matches layer1_nli row ID)",
        "claim_text": "Text of the claim",
        "candidate_passage_id": "Passage ID from passage_pool.jsonl",
        "candidate_passage_text": "Text of the candidate passage",
        "relevance_label": "1 = gold evidence, 0 = non-gold",
        "evidence_role": "GOLD_SUPPORT | GOLD_REFUTE | PARTIAL | TOPICALLY_RELATED | IRRELEVANT",
    },
    "layer3_claim_extraction/raw_ai_outputs.jsonl": {
        "doc_id": "Document identifier (CE_001 format)",
        "topic": "Topic category",
        "prompt_type": "factual_question | open_explanation | comparison | technical | historical | conversational",
        "generated_text": "Realistic AI-generated paragraph (3-6 sentences)",
        "human_verified": "false",
        "source_note": "synthetic_llm_authored",
    },
    "layer3_claim_extraction/claim_spans.jsonl": {
        "doc_id": "Matches raw_ai_outputs.jsonl doc_id",
        "sentence_id": "0-indexed sentence position within the document",
        "sentence_text": "Full sentence text",
        "is_factual_claim": "true if sentence contains an atomic checkable factual proposition",
        "claim_type": "Semantic type when is_factual_claim=true; null otherwise",
        "claim_span": "[start_char, end_char] within sentence_text",
    },
    "layer4_end_to_end/ai_generated_evaluation.csv": {
        "id": "Row ID",
        "generated_text": "AI-generated response text containing realistic hallucinations",
        "claim": "Atomic factual claim extracted from generated_text",
        "ground_truth_label": "SUPPORTS | REFUTES | NOT_ENOUGH_INFO",
        "gold_evidence": "Evidence passage used for NLI classification",
        "source": "Source category",
        "topic": "Topic category",
    },
    "layer5_heldout/heldout_responses.csv": {
        "response_id": "Unique response ID (L5_001 format)",
        "generated_text": "Realistic AI-generated multi-sentence response. No label-signaling text.",
        "topic": "Topic category",
        "response_style": "explanatory_paragraph | bullet_points | comparison | short_answer | historical_narrative | technical | conversational",
        "total_claims": "Integer count of atomic claims in this response",
        "supports_count": "Count of SUPPORTS claims",
        "refutes_count": "Count of REFUTES claims",
        "nei_count": "Count of NOT_ENOUGH_INFO claims",
        "response_verdict": "Fully factual | Mostly factual | Mixed factuality | Mostly inaccurate | Highly inaccurate",
        "factuality_score": "supports_count / total_claims (0.00-1.00)",
        "hallucination_types": "Semicolon-separated list of hallucination types present, or 'none'",
        "human_verified": "false",
        "source_note": "synthetic_llm_authored",
    },
    "layer5_heldout/heldout_claims.csv": {
        "claim_id": "Unique claim ID (L5_001_C1 format)",
        "response_id": "Parent response ID",
        "claim_text": "Atomic factual proposition. No label-signaling text.",
        "claim_type": "Semantic type of claim",
        "label": "SUPPORTS | REFUTES | NOT_ENOUGH_INFO",
        "difficulty": "easy | medium | hard",
        "hallucination_type": "Type of hallucination if REFUTES, else 'none'",
        "gold_evidence_id": "References heldout_evidence.csv evidence_id (gold row only)",
    },
    "layer5_heldout/heldout_evidence.csv": {
        "evidence_id": "Unique evidence row ID",
        "claim_id": "Parent claim ID",
        "passage_text": "Evidence passage text",
        "evidence_role": "GOLD_SUPPORT | GOLD_REFUTE | NEI_EVIDENCE | PARTIAL | TOPICALLY_RELATED | IRRELEVANT",
        "relevance_label": "1 = gold evidence for this claim, 0 = non-gold candidate",
    },
}

dict_path = os.path.join(BASE, "metadata", "data_dictionary.json")
with open(dict_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=True)
print("data_dictionary.json written.")
print("\nMetadata generation complete.")
