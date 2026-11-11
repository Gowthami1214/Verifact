"""
app.py
------
TruthSync — Web Interface for Factuality Verification of AI-Generated Text.

Built with Streamlit.
Run:
    streamlit run app.py
"""

import os, sys
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.pipeline import TruthSyncPipeline

# Page configuration
st.set_page_config(
    page_title="TruthSync — AI Factuality Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .verdict-factual {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #31C48D;
    }
    .verdict-mixed {
        background-color: #FEF08A;
        color: #713F12;
        border: 1px solid #FACC15;
    }
    .verdict-inaccurate {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F98080;
    }
    .claim-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background-color: #F9FAFB;
        margin-bottom: 0.8rem;
    }
    .badge-supports {
        background-color: #10B981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-refutes {
        background-color: #EF4444;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-nei {
        background-color: #F59E0B;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading TruthSync Neural Pipeline...")
def get_pipeline(nli_mode: str = "pretrained"):
    return TruthSyncPipeline(nli_mode=nli_mode)


def main():
    st.markdown('<div class="main-header">🔍 TruthSync: AI Factuality Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Verify whether factual claims in AI-generated text are supported or refuted by retrieved factual evidence.</div>',
        unsafe_allow_html=True
    )

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        model_choice = st.selectbox(
            "🧠 NLI Inference Engine:",
            [
                "RoBERTa-large-MNLI (Recommended for Open Facts)",
                "DeBERTa-v3-small (Domain Fine-Tuned)",
            ],
            index=0,
            help="RoBERTa-large is trained on 400k+ pairs for robust open-ended world facts. DeBERTa-v3 is fine-tuned on the research benchmark."
        )
        active_mode = "pretrained" if "RoBERTa" in model_choice else "finetuned"

        st.markdown("**Core Models Active:**")
        st.markdown("- **Claim Extractor**: Rule & Syntactic Parser")
        st.markdown("- **Embedder**: `all-MiniLM-L6-v2` (384-d)")
        st.markdown("- **Vector Database**: ChromaDB (Cosine k-NN)")
        st.markdown("- **Reranker**: `ms-marco-MiniLM-L-6-v2`")
        st.markdown(f"- **NLI Verifier**: `{'roberta-large-mnli' if active_mode == 'pretrained' else 'deberta-v3-small'}`")
        st.markdown("- **Relevance Gating**: Enabled (Threshold: -2.5)")
        st.markdown("---")
        st.markdown("### 💡 Quick Examples")
        
        sample_texts = {
            "Example 1: Mixed Web & Python (50% Factual)": (
                "Tim Berners-Lee invented the World Wide Web in 1989 at CERN. "
                "He also invented the Python programming language in 1991."
            ),
            "Example 2: Cold War & Berlin Wall (Refutation)": (
                "The Berlin Wall was built in 1961 and fell on November 9, 1989. "
                "It divided the city of Paris during the Cold War."
            ),
            "Example 3: Marie Curie Nobel Prizes": (
                "Marie Curie was the first woman to win a Nobel Prize. "
                "She received two Nobel Prizes, one in Physics and one in Chemistry."
            ),
            "Example 4: Database & Tech Hallucinations": (
                "Relational database management was proposed by Edgar F. Codd in 1970. "
                "Relational database operates exclusively by generating nuclear fission energy."
            )
        }

        selected_sample = st.selectbox("Choose a pre-set sample:", ["Custom Input"] + list(sample_texts.keys()))

    # Text Input area with Clear support
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "last_report" not in st.session_state:
        st.session_state["last_report"] = None

    if selected_sample != "Custom Input":
        st.session_state["input_text"] = sample_texts[selected_sample]

    user_text = st.text_area(
        "Enter AI-Generated Response or Paragraph to Verify:",
        value=st.session_state["input_text"],
        height=140,
        key="user_text_widget",
        placeholder="Paste text generated by an LLM here to analyze its factual accuracy...",
    )
    # Sync widget value
    st.session_state["input_text"] = user_text

    col1, col2, col3 = st.columns([2, 1, 5])
    with col1:
        verify_clicked = st.button("🚀 Verify Factuality", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked or not user_text.strip():
        st.session_state["last_report"] = None
        if clear_clicked:
            st.session_state["input_text"] = ""
            st.rerun()

    if verify_clicked and user_text.strip():
        pipeline = get_pipeline(active_mode)
        with st.spinner("Extracting atomic claims, querying ChromaDB vector store, and verifying evidence..."):
            st.session_state["last_report"] = pipeline.verify(user_text, return_all_candidates=True)

    report = st.session_state.get("last_report")
    if report is not None:
        st.markdown("---")
        st.subheader("📊 Factuality Verification Report")

        # Summary cards
        v_class = "verdict-mixed"
        if report["response_verdict"] in ("Fully factual", "Mostly factual"):
            v_class = "verdict-factual"
        elif report["response_verdict"] in ("Mostly inaccurate", "Highly inaccurate"):
            v_class = "verdict-inaccurate"

        st.markdown(
            f'<div class="verdict-box {v_class}">'
            f'Response Verdict: {report["response_verdict"]} '
            f'(Score: {report["factuality_score"]*100:.1f}%)'
            f'</div>',
            unsafe_allow_html=True
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Claims", report["total_claims"])
        m_col2.metric("Supported", report["supports_count"], delta="Verified Facts", delta_color="normal")
        m_col3.metric("Refuted", report["refutes_count"], delta="-Hallucinations", delta_color="inverse")
        m_col4.metric("Not Enough Info", report["nei_count"], delta="Unverifiable", delta_color="off")

        st.markdown("### 📝 Claim-by-Claim Evidence Breakdown")

        if not report["claims"]:
            st.info("No factual claims detected in the input.")
        else:
            for i, c in enumerate(report["claims"], 1):
                badge_class = "badge-nei"
                badge_icon = "❓"
                if c["label"] == "SUPPORTS":
                    badge_class = "badge-supports"
                    badge_icon = "✅"
                elif c["label"] == "REFUTES":
                    badge_class = "badge-refutes"
                    badge_icon = "❌"

                with st.expander(
                    f"{badge_icon} Claim {i}: {c['claim_text'][:70]}...  [{c['label']}]",
                    expanded=True
                ):
                    c_col1, c_col2, c_col3 = st.columns([3, 1, 1])
                    with c_col1:
                        st.markdown(f"**Claim Statement**: {c['claim_text']}")
                        st.markdown(f"**Claim Type**: `{c['claim_type'] or 'general'}`")
                    with c_col2:
                        st.markdown(f"**Verdict**: <span class='{badge_class}'>{c['label']}</span>", unsafe_allow_html=True)
                    with c_col3:
                        st.markdown(f"**Confidence**: `{c['confidence']*100:.1f}%`")

                    st.markdown("---")
                    st.markdown("**Retrieved Factual Evidence**:")
                    if c["evidence_text"]:
                        st.info(f"📄 *\"{c['evidence_text']}\"*")
                        st.caption(f"Relevance Rank Score: `{c['evidence_score']:.3f}`")
                    else:
                        st.warning("No relevant evidence passage found in knowledge base.")

                    # Probabilities breakdown
                    scores = c.get("scores", {})
                    if scores:
                        st.markdown("**Per-Class Probabilities:**")
                        p_col1, p_col2, p_col3 = st.columns(3)
                        p_col1.progress(scores.get("SUPPORTS", 0.0), text=f"Supports: {scores.get('SUPPORTS', 0.0)*100:.1f}%")
                        p_col2.progress(scores.get("REFUTES", 0.0), text=f"Refutes: {scores.get('REFUTES', 0.0)*100:.1f}%")
                        p_col3.progress(scores.get("NOT_ENOUGH_INFO", 0.0), text=f"Not Enough Info: {scores.get('NOT_ENOUGH_INFO', 0.0)*100:.1f}%")

                    # Alternative candidates
                    if c.get("candidates") and len(c["candidates"]) > 1:
                        with st.expander("🔎 View Alternative Retrieved Candidates"):
                            for r_idx, cand in enumerate(c["candidates"][1:], 2):
                                st.markdown(f"**Candidate #{r_idx}** (Score: `{cand.get('rerank_score', cand.get('score', 0)):.3f}`):")
                                st.write(cand.get("passage_text", ""))

    elif verify_clicked and not user_text.strip():
        st.warning("Please enter some text or select an example from the sidebar.")


main()
