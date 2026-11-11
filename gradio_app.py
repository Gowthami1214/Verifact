"""
gradio_app.py
---------------
TruthSync Factuality Verification Web Application using Gradio.
Features:
  - Modern responsive UI with hero header and styled glassmorphism/gradient cards
  - Instant clear functionality: clears input text AND resets all outputs simultaneously
  - Real-time output reset when input is emptied
  - Multi-stage neural factuality verification (Claim Extraction -> ChromaDB Dense Retrieval -> Cross-Encoder Reranking -> Calibrated DeBERTa-v3 NLI)
  - Color-coded verdict banners, factuality score meters, and claim breakdown cards

Run:
    python gradio_app.py
"""

import os
import sys
import html

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gradio as gr
from src.pipeline import TruthSyncPipeline

print("[TruthSync Web] Initializing neural verification pipeline...")
pipeline = TruthSyncPipeline(nli_mode="pretrained")
print("[TruthSync Web] Pipeline ready!")

# ── Styling & HTML Templates ──────────────────────────────────────────────────

CSS_STYLES = """
<style>
  /* Global polish */
  .gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  }

  /* Hero Section */
  .hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #111827 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 20px;
    color: #ffffff;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
  }
  .hero-badge {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(129, 140, 248, 0.4);
    border-radius: 20px;
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .hero-title {
    font-size: 26px;
    font-weight: 800;
    margin: 0 0 6px 0;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .hero-sub {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
  }
  .pipeline-pill-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
  }
  .pipeline-pill {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    color: #cbd5e1;
  }

  /* Empty state */
  .empty-state-card {
    background: #ffffff;
    border: 2px dashed #cbd5e1;
    border-radius: 14px;
    padding: 36px 24px;
    text-align: center;
    color: #64748b;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
  }
  .empty-state-icon {
    font-size: 38px;
    margin-bottom: 10px;
  }
  .empty-state-title {
    font-weight: 700;
    font-size: 16px;
    color: #1e293b;
    margin-bottom: 4px;
  }
  .empty-state-desc {
    font-size: 13px;
    color: #64748b;
    max-width: 440px;
    margin: 0 auto;
  }

  /* Overall Verdict Card */
  .verdict-card {
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    color: white;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.16);
  }
  .verdict-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .verdict-title {
    font-size: 22px;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0;
  }
  .verdict-score-badge {
    background: rgba(255, 255, 255, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 18px;
    font-weight: 800;
  }
  .verdict-desc {
    font-size: 13.5px;
    opacity: 0.95;
    margin-bottom: 16px;
    line-height: 1.4;
  }
  .progress-container {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .progress-bar-fill {
    height: 100%;
    background: #ffffff;
    border-radius: 8px;
    transition: width 0.4s ease;
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .stat-tile {
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 10px;
    padding: 8px 10px;
    text-align: center;
  }
  .stat-num {
    font-size: 18px;
    font-weight: 800;
  }
  .stat-lbl {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.92;
  }

  /* Claim Cards */
  .claims-section-title {
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
    margin: 18px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .claim-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
  }
  .claim-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .claim-tag {
    font-size: 12px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .confidence-tag {
    font-size: 11px;
    color: #64748b;
    background: #f1f5f9;
    padding: 4px 9px;
    border-radius: 6px;
    font-weight: 600;
  }
  .claim-text-box {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 12px;
    line-height: 1.5;
  }
  .evidence-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
  }
  .evidence-box.supported {
    background: #f0fdf4;
    border-color: #bbf7d0;
  }
  .evidence-box.refuted {
    background: #fef2f2;
    border-color: #fecaca;
  }
  .evidence-box.nei {
    background: #fffbeb;
    border-color: #fef3c7;
  }
  .evidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .evidence-text {
    font-size: 13px;
    color: #1e293b;
    font-style: italic;
    line-height: 1.5;
  }
  .probabilities-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 11px;
    padding-top: 8px;
    border-top: 1px solid #f1f5f9;
  }
  .prob-pill {
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
    background: #f1f5f9;
  }
</style>
"""

EMPTY_SUMMARY_HTML = """
<div class="empty-state-card">
  <div class="empty-state-icon">🔬</div>
  <div class="empty-state-title">Ready for Factuality Verification</div>
  <div class="empty-state-desc">
    Paste any AI-generated response, summary, or paragraph on the left and click <b>Verify Factuality</b> to evaluate each claim against factual knowledge.
  </div>
</div>
"""

EMPTY_DETAILS_HTML = """
<div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 22px 18px; text-align: center; color: #94a3b8; font-size: 13px;">
  Extracted claims, retrieved evidence passages, and calibrated neural confidence scores will render here.
</div>
"""

VERDICT_CONFIG = {
    "Fully factual": {
        "bg": "linear-gradient(135deg, #10B981 0%, #059669 100%)",
        "icon": "✅",
        "desc": "All extracted claims are verified and corroborated by retrieved factual evidence."
    },
    "Mostly factual": {
        "bg": "linear-gradient(135deg, #14B8A6 0%, #0D9488 100%)",
        "icon": "✨",
        "desc": "The overwhelming majority of claims are supported by facts, with no contradictory hallucinations."
    },
    "Mixed factuality": {
        "bg": "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)",
        "icon": "⚠️",
        "desc": "The text contains a mixture of verified facts alongside refuted claims or unsupported statements."
    },
    "Mostly inaccurate": {
        "bg": "linear-gradient(135deg, #F43F5E 0%, #E11D48 100%)",
        "icon": "❌",
        "desc": "Significant factual contradictions and hallucinations were detected across the response."
    },
    "Highly inaccurate": {
        "bg": "linear-gradient(135deg, #E11D48 0%, #9F1239 100%)",
        "icon": "🚨",
        "desc": "The response consists almost entirely of statements contradicting verified knowledge."
    },
    "No verifiable claims": {
        "bg": "linear-gradient(135deg, #94A3B8 0%, #64748B 100%)",
        "icon": "ℹ️",
        "desc": "No specific factual assertions or propositions were detected in the input text."
    }
}

CLAIM_CONFIG = {
    "SUPPORTS": {
        "badge_bg": "#dcfce7",
        "badge_color": "#15803d",
        "border": "#86efac",
        "icon": "✅ SUPPORTS",
        "box_cls": "supported",
        "hdr_color": "#166534"
    },
    "REFUTES": {
        "badge_bg": "#fee2e2",
        "badge_color": "#b91c1c",
        "border": "#fca5a5",
        "icon": "❌ REFUTES",
        "box_cls": "refuted",
        "hdr_color": "#991b1b"
    },
    "NOT_ENOUGH_INFO": {
        "badge_bg": "#fef3c7",
        "badge_color": "#b45309",
        "border": "#fde68a",
        "icon": "❓ NOT ENOUGH INFO",
        "box_cls": "nei",
        "hdr_color": "#92400e"
    }
}


# ── Verification Function ─────────────────────────────────────────────────────

def verify_text(input_text):
    if not input_text or not input_text.strip():
        return EMPTY_SUMMARY_HTML, EMPTY_DETAILS_HTML, None

    # Strip outer quotes or accidental wrapping
    cleaned_input = input_text.strip()
    report = pipeline.verify(cleaned_input, return_all_candidates=True)

    verdict = report.get("response_verdict", "No verifiable claims")
    score = report.get("factuality_score", 0.0) * 100
    sup = report.get("supports_count", 0)
    ref = report.get("refutes_count", 0)
    nei = report.get("nei_count", 0)
    tot = report.get("total_claims", 0)

    cfg = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["No verifiable claims"])

    # 1. Generate Overall Verdict Banner
    summary_html = f"""
    <div class="verdict-card" style="background: {cfg['bg']};">
      <div class="verdict-header">
        <div class="verdict-title">{cfg['icon']} {html.escape(verdict)}</div>
        <div class="verdict-score-badge">{score:.1f}% Factuality</div>
      </div>
      <div class="verdict-desc">{cfg['desc']}</div>
      <div class="progress-container">
        <div class="progress-bar-fill" style="width: {max(5.0, score):.1f}%;"></div>
      </div>
      <div class="stats-grid">
        <div class="stat-tile">
          <div class="stat-num">{sup} / {tot}</div>
          <div class="stat-lbl">🟢 Verified (Supports)</div>
        </div>
        <div class="stat-tile">
          <div class="stat-num">{ref} / {tot}</div>
          <div class="stat-lbl">🔴 Hallucinations (Refutes)</div>
        </div>
        <div class="stat-tile">
          <div class="stat-num">{nei} / {tot}</div>
          <div class="stat-lbl">🟡 Unverified (No Info)</div>
        </div>
      </div>
    </div>
    """

    # 2. Generate Detailed Claim Breakdown Cards
    if not report.get("claims"):
        details_html = """
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; color: #64748b; font-size: 13.5px;">
          No verifiable claim sentences could be isolated from the text. Try inputting declarative factual statements.
        </div>
        """
    else:
        cards = ['<div class="claims-section-title">🔍 Claim-by-Claim Neural Verification Breakdown</div>']
        for i, c in enumerate(report["claims"], 1):
            lbl = c.get("label", "NOT_ENOUGH_INFO")
            conf = c.get("confidence", 0.0) * 100
            ctype = c.get("claim_type", "general")
            claim_str = html.escape(c.get("claim_text", ""))
            ev_str = html.escape(c.get("evidence_text", "No relevant evidence found in knowledge store."))
            ev_score = c.get("evidence_score", 0.0)

            c_cfg = CLAIM_CONFIG.get(lbl, CLAIM_CONFIG["NOT_ENOUGH_INFO"])
            scores = c.get("scores", {})
            p_sup = scores.get("SUPPORTS", 0.0) * 100
            p_ref = scores.get("REFUTES", 0.0) * 100
            p_nei = scores.get("NOT_ENOUGH_INFO", 0.0) * 100

            cards.append(f"""
            <div class="claim-card">
              <div class="claim-top-row">
                <span class="claim-tag" style="background: {c_cfg['badge_bg']}; color: {c_cfg['badge_color']}; border: 1px solid {c_cfg['border']};">
                  {c_cfg['icon']}
                </span>
                <div>
                  <span class="confidence-tag">Category: <b>{html.escape(ctype)}</b></span>
                  <span class="confidence-tag" style="margin-left: 6px;">Confidence: <b>{conf:.1f}%</b></span>
                </div>
              </div>

              <div class="claim-text-box">
                "{claim_str}"
              </div>

              <div class="evidence-box {c_cfg['box_cls']}">
                <div class="evidence-header" style="color: {c_cfg['hdr_color']};">
                  <span>📖 Retrieved Factual Evidence</span>
                  <span>Relevance Score: {ev_score:.3f}</span>
                </div>
                <div class="evidence-text">"{ev_str}"</div>
              </div>

              <div class="probabilities-row">
                <span style="font-weight: 600; color: #64748b; margin-right: 4px;">Model Probabilities:</span>
                <span class="prob-pill" style="color: #15803d;">Supports: {p_sup:.1f}%</span>
                <span class="prob-pill" style="color: #b91c1c;">Refutes: {p_ref:.1f}%</span>
                <span class="prob-pill" style="color: #b45309;">Not Enough Info: {p_nei:.1f}%</span>
              </div>
            </div>
            """)
        details_html = "".join(cards)

    return summary_html, details_html, report


def clear_all():
    """Clear both input and all outputs simultaneously."""
    return "", EMPTY_SUMMARY_HTML, EMPTY_DETAILS_HTML, None


def on_text_changed(text):
    """If the input is cleared by deleting text, automatically clear the outputs."""
    if not text or not text.strip():
        return EMPTY_SUMMARY_HTML, EMPTY_DETAILS_HTML, None
    return gr.no_change(), gr.no_change(), gr.no_change()


# ── UI Construction ───────────────────────────────────────────────────────────

with gr.Blocks(title="TruthSync — AI Factuality Verification") as demo:
    # Inject styling
    gr.HTML(CSS_STYLES)

    # Hero Banner
    gr.HTML("""
    <div class="hero-container">
      <span class="hero-badge">TruthSync · Neural Factuality System</span>
      <h1 class="hero-title">🔍 AI-Generated Text Factuality Verification</h1>
      <p class="hero-sub">
        Detect AI hallucinations and verify factual accuracy across AI-generated paragraphs using deep retrieval, cross-encoder reranking, and temperature-calibrated natural language inference.
      </p>
      <div class="pipeline-pill-row">
        <span class="pipeline-pill">⚡ Step 1: Claim Proposition Extractor</span>
        <span class="pipeline-pill">📚 Step 2: Dense MiniLM-L6-v2 Retrieval</span>
        <span class="pipeline-pill">🎯 Step 3: MS-MARCO Cross-Encoder Reranker</span>
        <span class="pipeline-pill">🧠 Step 4: Fine-Tuned DeBERTa-v3-small NLI (T=1.135)</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # Left Column: Input & Controls
        with gr.Column(scale=5):
            input_box = gr.Textbox(
                label="Enter AI-Generated Text or Paragraph to Verify:",
                placeholder="Paste an AI-generated paragraph, answer, or summary here...",
                lines=6,
            )

            with gr.Row():
                verify_btn = gr.Button("🚀 Verify Factuality", variant="primary", scale=3)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            gr.Markdown("### 💡 Quick Test Scenarios & Edge Cases")
            
            ex_1 = gr.Button("1️⃣ Mixed: Web Inventor + Python Hallucination", size="sm")
            ex_2 = gr.Button("2️⃣ Fully Factual: Marie Curie Nobel Prizes", size="sm")
            ex_3 = gr.Button("3️⃣ Blatant Hallucination: Berlin Wall in Paris", size="sm")
            ex_4 = gr.Button("4️⃣ Quotes Edge Case: Albert Einstein Violin", size="sm")

        # Right Column: Output Results
        with gr.Column(scale=6):
            summary_output = gr.HTML(value=EMPTY_SUMMARY_HTML, label="Factuality Verdict")
            details_output = gr.HTML(value=EMPTY_DETAILS_HTML, label="Evidence Breakdown")

    with gr.Accordion("📦 Complete Diagnostic JSON Payload", open=False):
        json_output = gr.JSON(label="Full Pipeline Output")

    # Wire verify button
    verify_btn.click(
        fn=verify_text,
        inputs=[input_box],
        outputs=[summary_output, details_output, json_output],
    )

    # Wire clear button: clears input AND all outputs
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_box, summary_output, details_output, json_output],
    )

    # Wire input_box change: when user deletes text, reset outputs
    input_box.change(
        fn=on_text_changed,
        inputs=[input_box],
        outputs=[summary_output, details_output, json_output],
    )

    # Wire quick preset buttons
    def set_ex1():
        txt = "Tim Berners-Lee invented the World Wide Web in 1989 at CERN. He also invented the Python programming language in 1991."
        s, d, j = verify_text(txt)
        return txt, s, d, j

    def set_ex2():
        txt = "Marie Curie was the first woman to win a Nobel Prize. She received two Nobel Prizes, one in Physics and one in Chemistry."
        s, d, j = verify_text(txt)
        return txt, s, d, j

    def set_ex3():
        txt = "The Berlin Wall was built in 1961 and fell on November 9, 1989. It divided the city of Paris during the Cold War."
        s, d, j = verify_text(txt)
        return txt, s, d, j

    def set_ex4():
        txt = '"Albert Einstein was an avid fan of classical music and played the violin from childhood."'
        s, d, j = verify_text(txt)
        return txt, s, d, j

    ex_1.click(fn=set_ex1, outputs=[input_box, summary_output, details_output, json_output])
    ex_2.click(fn=set_ex2, outputs=[input_box, summary_output, details_output, json_output])
    ex_3.click(fn=set_ex3, outputs=[input_box, summary_output, details_output, json_output])
    ex_4.click(fn=set_ex4, outputs=[input_box, summary_output, details_output, json_output])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
