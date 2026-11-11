# Factuality Verification Dataset

## Files
- `train.csv` — 240 records
- `validation.csv` — 30 records
- `test.csv` — 30 records
- `ai_generated_evaluation.csv` — mixed-claim AI-generated paragraphs
- `all_claim_evidence.jsonl` — JSONL version of the three-way claim/evidence corpus, with a `split` field
- `dataset_statistics.json` — validation statistics

## Labels
- SUPPORTS: evidence establishes the claim.
- REFUTES: evidence establishes a contradiction.
- NOT_ENOUGH_INFO: evidence is relevant but does not establish truth or falsity.

## Provenance
The evidence passages are curated/paraphrased from well-established facts and are tagged with a source category. No fabricated URLs, article titles, publication identifiers, or citations are included. For a production-grade system, replace/augment these passages with retrieved, versioned passages from real sources and human verification.

## Split integrity
The same factual seed is kept inside one split only, so equivalent propositions do not cross train/validation/test boundaries. Claims are surface-varied with neutral wording to reduce lexical memorization.

## Recommended use
This dataset is suitable as a prototype NLI/factuality benchmark and for validating the pipeline:
claim extraction -> embedding retrieval -> evidence ranking -> NLI verification -> factuality report.
It is intentionally much smaller than a production Transformer training corpus; scale it with real, source-linked retrieval data and multiple independent human annotations.
