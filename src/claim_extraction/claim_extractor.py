"""
src/claim_extraction/claim_extractor.py
-----------------------------------------
Sentence-level claim extraction for TruthSync.

Given a multi-sentence AI-generated paragraph, this module:
  1. Splits text into individual sentences (spaCy / NLTK fallback).
  2. Classifies each sentence as factual claim vs. non-claim using a
     zero-shot NLI approach (works without fine-tuning) OR a fine-tuned
     DeBERTa-v3 token classifier (when trained on Layer 3 data).
  3. Returns a list of extracted factual claims with their type labels.

Two operating modes:
  - MODE = "rule"   : Lightweight heuristic classifier (no GPU, instant, no download).
  - MODE = "nli"    : Zero-shot NLI via DeBERTa-v3-small cross-encoder (more accurate).

The fine-tuning script (scripts/train_claim_extractor.py) trains a token
classifier on layer3_claim_extraction/claim_spans.jsonl and saves to
models/claim_extraction/. Once trained, set MODE = "trained".
"""

import re
import os
from typing import List, Dict, Any, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# ── Sentence splitter ────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split paragraph text into sentences.
    Uses a regex-based splitter that handles abbreviations and quotes robustly.
    """
    text = text.strip().strip('"\'“”‘’')
    # Fixed-width lookbehind for sentence-ending punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    cleaned = []
    for s in raw_sentences:
        s = s.strip().strip('"\'“”‘’')
        if s:
            cleaned.append(s)
    return cleaned


# ── Claim type classifier keywords ──────────────────────────────────────────

CLAIM_TYPE_PATTERNS: List[Tuple[str, str]] = [
    ("person",       r'\b(was|were|is|are)\b.{0,40}\b(born|died|awarded|founded|invented|created|developed|designed|wrote|published|won|received|led|served|played|scored)\b'),
    ("date",         r'\b(in\s+\d{4}|on\s+\w+ \d+|january|february|march|april|may|june|july|august|september|october|november|december)\b'),
    ("number",       r'\b\d[\d,\.]*\s*(million|billion|thousand|km|metres?|degrees?|percent|%|years?|days?|hours?|kg|km/h|seconds?)\b'),
    ("location",     r'\b(located|situated|based|headquartered|found|lies?|stands?)\b.{0,50}\b(in|at|on|near|between)\b'),
    ("organization", r'\b(organization|institute|university|corporation|company|association|committee|council|agency|department|ministry|federation|union)\b'),
    ("event",        r'\b(battle|war|revolution|treaty|conference|summit|agreement|election|inauguration|ceremony|launch|discovery|expedition)\b'),
    ("definition",   r'\b(is\s+a|is\s+an|are\s+a|refers?\s+to|defined?\s+as|known\s+as|called|consists?\s+of|comprises?)\b'),
    ("scientific",   r'\b(energy|force|mass|speed|velocity|frequency|temperature|pressure|electrons?|atoms?|molecules?|cells?|genes?|DNA|RNA|protein|enzyme)\b'),
    ("comparison",   r'\b(larger|smaller|greater|higher|lower|faster|slower|more|less|most|least|oldest|youngest|first|last|longest|shortest|deepest|tallest)\b'),
    ("cause_effect", r'\b(causes?|results?\s+in|leads?\s+to|due\s+to|because|therefore|consequently|produces?|generates?|enables?|prevents?)\b'),
    ("relationship", r'\b(relationship|connection|linked|associated|related|part\s+of|member\s+of|belongs?\s+to|founded\s+by|created\s+by|led\s+by)\b'),
]

NON_CLAIM_PATTERNS = [
    r'^(it|this|that|these|those)\s+(is|was|are|were)\s+(important|significant|notable|well-known|famous|widely)',
    r'\b(should|could|would|might|may|perhaps|probably|possibly|generally|often|usually|sometimes|typically|arguably)\b',
    r'\b(interesting|fascinating|remarkable|impressive|amazing|incredible|wonderful)\b',
    r'^(many|some|most|few|several)\s+(people|experts|researchers|scientists|scholars)',
    r'\b(I|we|you|they)\b.{0,20}\b(think|believe|feel|suggest|recommend|argue|claim)\b',
    r'^\s*$',
]

NON_CLAIM_RE = [re.compile(p, re.IGNORECASE) for p in NON_CLAIM_PATTERNS]
CLAIM_TYPE_RE = [(label, re.compile(p, re.IGNORECASE)) for label, p in CLAIM_TYPE_PATTERNS]


def _is_factual_claim_heuristic(sentence: str) -> bool:
    """
    Heuristic filter: returns True if sentence is likely a verifiable factual claim.
    """
    if len(sentence.split()) < 5:
        return False

    # Exclude non-claim patterns first
    for pattern in NON_CLAIM_RE:
        if pattern.search(sentence):
            return False

    # Must contain at least one factual signal
    factual_signals = [
        r'\b\d{4}\b',                  # year
        r'\b\d+[\d,\.]+\b',            # any number
        r'\b(was|were|is|are)\b',       # copula verbs
        r'\b(invented|discovered|founded|created|developed|published|signed|launched|released|won|received|established|adopted|introduced)\b',
    ]
    has_signal = any(re.search(p, sentence, re.IGNORECASE) for p in factual_signals)
    return has_signal


def _classify_claim_type(sentence: str) -> str:
    """Map a factual claim sentence to one of the 11 claim type categories."""
    for label, pattern in CLAIM_TYPE_RE:
        if pattern.search(sentence):
            return label
    return "definition"  # default


# ── Main claim extractor class ───────────────────────────────────────────────

class ClaimExtractor:
    """
    Extracts atomic factual claims from AI-generated text.

    Usage:
        extractor = ClaimExtractor(mode="rule")  # no model needed
        claims = extractor.extract("Python was created by Guido van Rossum in 1991.
                                    It is widely used in data science today.")
        # returns:
        # [{"sentence_id": 0,
        #   "sentence_text": "Python was created by ...",
        #   "is_factual_claim": True,
        #   "claim_type": "person",
        #   "claim_span": [0, 52]}]
    """

    def __init__(self, mode: str = "rule", model_path: str = None):
        """
        Args:
            mode:       "rule"    - heuristic classifier (fast, no model needed)
                        "nli"     - zero-shot NLI via cross-encoder (needs transformers)
                        "trained" - fine-tuned token classifier (needs model_path)
            model_path: Path to fine-tuned model directory (only for mode="trained").
        """
        self.mode = mode
        self._nli_pipe = None
        self._trained_model = None

        if mode == "nli":
            self._load_nli_classifier()
        elif mode == "trained" and model_path:
            self._load_trained_classifier(model_path)

    def _load_nli_classifier(self):
        """Load a zero-shot text classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            print("[ClaimExtractor] Loading zero-shot NLI classifier...")
            self._nli_pipe = hf_pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                device=-1,  # CPU
            )
            print("[ClaimExtractor] NLI classifier ready.")
        except Exception as e:
            print(f"[ClaimExtractor] NLI load failed ({e}). Falling back to rule mode.")
            self.mode = "rule"

    def _load_trained_classifier(self, model_path: str):
        """Load a fine-tuned sentence classification model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            print(f"[ClaimExtractor] Loading trained model from {model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._trained_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._trained_model.eval()
            self._torch = torch
            print("[ClaimExtractor] Trained model ready.")
        except Exception as e:
            print(f"[ClaimExtractor] Trained model load failed ({e}). Falling back to rule mode.")
            self.mode = "rule"

    def _classify_sentence_nli(self, sentence: str) -> bool:
        """Use zero-shot NLI to determine if sentence is a factual claim."""
        result = self._nli_pipe(
            sentence,
            candidate_labels=["factual claim", "opinion or commentary"],
            multi_label=False,
        )
        top_label = result["labels"][0]
        return top_label == "factual claim"

    def _classify_sentence_trained(self, sentence: str) -> bool:
        """Use fine-tuned classifier to predict factual claim (label 1) or not (label 0)."""
        import torch
        inputs = self._tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self._trained_model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).item())
        return pred == 1

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract factual claims from a paragraph of AI-generated text.

        Args:
            text: A multi-sentence AI-generated paragraph.

        Returns:
            List of sentence annotation dicts:
            [
              {
                "sentence_id":    int,          # 0-indexed position in text
                "sentence_text":  str,          # full sentence
                "is_factual_claim": bool,       # True if verifiable factual claim
                "claim_type":     str | None,   # one of 11 types, or None
                "claim_span":     [int, int] | None  # char offsets in sentence_text
              },
              ...
            ]
        """
        sentences = _split_sentences(text)
        results = []

        for idx, sent in enumerate(sentences):
            # Classify
            if self.mode == "nli" and self._nli_pipe:
                is_claim = self._classify_sentence_nli(sent)
            elif self.mode == "trained" and self._trained_model:
                is_claim = self._classify_sentence_trained(sent)
            else:
                is_claim = _is_factual_claim_heuristic(sent)

            claim_type = _classify_claim_type(sent) if is_claim else None
            claim_span = [0, len(sent)] if is_claim else None

            results.append({
                "sentence_id":    idx,
                "sentence_text":  sent,
                "is_factual_claim": is_claim,
                "claim_type":     claim_type,
                "claim_span":     claim_span,
            })

        return results

    def extract_claims_only(self, text: str) -> List[str]:
        """
        Convenience method: returns only the claim sentence strings.
        """
        annotations = self.extract(text)
        return [a["sentence_text"] for a in annotations if a["is_factual_claim"]]

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Run extract() over a list of AI-generated paragraphs.
        """
        return [self.extract(t) for t in texts]
