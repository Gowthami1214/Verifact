"""
src/nli/nli_classifier.py
--------------------------
Natural Language Inference (NLI) classifier for TruthSync.

Given an (evidence, claim) pair, outputs one of:
    SUPPORTS | REFUTES | NOT_ENOUGH_INFO
along with per-class confidence scores.

Two operating modes:
  - mode="pretrained"  : Uses roberta-large-mnli out of the box (no fine-tuning).
  - mode="finetuned"   : Loads the fine-tuned DeBERTa-v3-small model from
                         models/nli/deberta-v3-small-nli/ after running
                         scripts/train_nli_classifier.py.

The NLI input format follows the standard cross-encoder pattern:
    [CLS] evidence [SEP] claim [SEP]
which conditions the classification on whether the evidence supports the claim.
"""

import os
from typing import List, Dict, Tuple, Union

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Label mappings ────────────────────────────────────────────────────────────
# roberta-large-mnli uses: {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}
# We map those to TruthSync labels:
MNLI_TO_TRUTHSYNC = {0: "REFUTES", 1: "NOT_ENOUGH_INFO", 2: "SUPPORTS"}

# Our fine-tuned model uses the native TruthSync order:
TRUTHSYNC_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PRETRAINED = "roberta-large-mnli"
DEFAULT_BEST       = os.path.join(_ROOT, "models", "deberta_best")
DEFAULT_FINETUNED  = DEFAULT_BEST if os.path.isdir(DEFAULT_BEST) else os.path.join(_ROOT, "models", "nli", "deberta-v3-small-nli")


class NLIClassifier:
    """
    Factuality NLI classifier: (evidence, claim) -> label + confidence scores.

    Usage:
        clf = NLIClassifier(mode="finetuned")
        result = clf.predict(
            evidence="Guido van Rossum created Python in 1991.",
            claim="Python was created by Brendan Eich."
        )
    """

    def __init__(
        self,
        mode: str = "pretrained",
        model_path: str = None,
        max_length: int = 256,
        temperature: float = 1.0,
    ):
        self.mode        = mode
        self.max_length  = max_length
        self.temperature = temperature
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "pretrained":
            path = model_path or DEFAULT_PRETRAINED
            self._label_map = MNLI_TO_TRUTHSYNC
            self._is_mnli   = True
        else:
            if model_path:
                path = model_path
            elif os.path.isdir(DEFAULT_BEST):
                path = DEFAULT_BEST
            else:
                path = DEFAULT_FINETUNED
            self._label_map = {i: l for i, l in enumerate(TRUTHSYNC_LABELS)}
            self._is_mnli   = False

        print(f"[NLI] Loading model ({mode}): {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[NLI] Model ready on {self.device} (Temperature: {self.temperature}).")

    # ── Core prediction ───────────────────────────────────────────────────────

    def _encode(self, evidence: str, claim: str) -> Dict:
        """Tokenize an (evidence, claim) pair using the winning representation format."""
        if self._is_mnli:
            text_a = evidence
            text_b = claim
        else:
            text_a = f"Claim: {claim}"
            text_b = f"Evidence: {evidence}"

        return self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def predict(self, evidence: str, claim: str) -> Dict:
        """
        Classify a single (evidence, claim) pair with calibrated confidence.
        """
        inputs = self._encode(evidence, claim)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits          # (1, num_labels)
            if self.temperature != 1.0:
                logits = logits / self.temperature

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        # Map model output indices to TruthSync labels
        if self._is_mnli:
            scores = {
                "SUPPORTS":        probs[2],  # entailment
                "REFUTES":         probs[0],  # contradiction
                "NOT_ENOUGH_INFO": probs[1],  # neutral
            }
        else:
            scores = {
                label: probs[i]
                for i, label in enumerate(TRUTHSYNC_LABELS)
            }

        best_label = max(scores, key=scores.get)
        return {
            "label":      best_label,
            "confidence": round(scores[best_label], 4),
            "scores":     {k: round(v, 4) for k, v in scores.items()},
        }

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[Dict]:
        """
        Classify a list of (evidence, claim) pairs efficiently in mini-batches.

        Args:
            pairs:      List of (evidence, claim) tuples.
            batch_size: Inference batch size.

        Returns:
            List of prediction dicts (same order as input).
        """
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            evidences = [p[0] for p in batch]
            claims    = [p[1] for p in batch]

            inputs = self.tokenizer(
                evidences,
                claims,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits        # (B, num_labels)

            probs_batch = F.softmax(logits, dim=-1).cpu().tolist()

            for probs in probs_batch:
                if self._is_mnli:
                    scores = {
                        "SUPPORTS":        probs[2],
                        "REFUTES":         probs[0],
                        "NOT_ENOUGH_INFO": probs[1],
                    }
                else:
                    scores = {
                        label: probs[j]
                        for j, label in enumerate(TRUTHSYNC_LABELS)
                    }
                best = max(scores, key=scores.get)
                results.append({
                    "label":      best,
                    "confidence": round(scores[best], 4),
                    "scores":     {k: round(v, 4) for k, v in scores.items()},
                })

        return results

    # ── Convenience helpers ───────────────────────────────────────────────────

    def predict_label_only(self, evidence: str, claim: str) -> str:
        """Return just the predicted label string."""
        return self.predict(evidence, claim)["label"]

    def score_factuality(self, claim_results: List[Dict]) -> Dict:
        """
        Aggregate per-claim NLI results into a response-level factuality report.

        Args:
            claim_results: List of predict() outputs, one per claim.

        Returns:
            {
              "factuality_score":  float,  # supports / total
              "response_verdict":  str,
              "supports_count":    int,
              "refutes_count":     int,
              "nei_count":         int,
              "total_claims":      int,
            }
        """
        total   = len(claim_results)
        sup     = sum(1 for r in claim_results if r["label"] == "SUPPORTS")
        ref     = sum(1 for r in claim_results if r["label"] == "REFUTES")
        nei     = sum(1 for r in claim_results if r["label"] == "NOT_ENOUGH_INFO")
        score   = round(sup / total, 4) if total > 0 else 0.0

        if sup == total:
            verdict = "Fully factual"
        elif sup >= total * 0.8 and ref == 0:
            verdict = "Mostly factual"
        elif ref == total:
            verdict = "Highly inaccurate"
        elif ref > total * 0.5:
            verdict = "Mostly inaccurate"
        else:
            verdict = "Mixed factuality"

        return {
            "factuality_score": score,
            "response_verdict": verdict,
            "supports_count":   sup,
            "refutes_count":    ref,
            "nei_count":        nei,
            "total_claims":     total,
        }
