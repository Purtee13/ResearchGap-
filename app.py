"""ResearchGap: multimodal multi-agent research assistant (Streamlit UI)."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from researchgap.config import GEMMA_MODELS, Settings
from researchgap.demo import demo_result
from researchgap.llm import LLMError
from researchgap.pdf_processing import ParsedPaper, parse_pdf, render_page
from researchgap.pipeline import run_pipeline
from researchgap.report import cite, to_markdown
from researchgap.sample_papers import PAPERS, build_sample_papers
from researchgap.schemas import EvidenceRef, RunResult

st.set_page_config(page_title="ResearchGap", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
.rg-badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600;}
.rg-supported {background:#e3f6e8;color:#1b6b34;}
.rg-weak {background:#fff4d6;color:#8a5a00;}
.rg-rejected {background:#fde4e4;color:#9b1c1c;}
.rg-card {border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:10px 14px;margin-bottom:8px;}
.rg-muted {opacity:.7;font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

PAPER_IDS = ["A", "B", "C"]
SAMPLE_NAMES = {v["filename"] for v in PAPERS.values()}


@st.cache_data(show_spinner=False)
def _parse(pid: str, name: str, data: bytes) -> ParsedPaper:
    return parse_pdf(pid, name, data)


@st.cache_data(show_spinner=False, max_entries=256)
def _render(data: bytes, page: int, dpi: int, highlight: tuple[str, ...] = ()) -> bytes:
    return render_page(data, page, dpi=dpi, fmt="png", highlight=list(highlight))


# ---------------------------------------------------------------- sidebar
env = Settings.from_env()
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Live (Gemma 4)", "Offline demo"], help=(
        "Offline demo shows precomputed output for the bundled synthetic sample papers, without calling a model."))
    api_key = env.api_key
    if mode.startswith("Live"):
        if env.api_key:
            st.success("GEMINI_API_KEY loaded from environment")
        else:
            api_key = st.text_input("Gemini API key", type="password",
                                    help="Kept in this browser session only. Get one at aistudio.google.com")
        model_choice = st.selectbox("Gemma 4 model", GEMMA_MODELS + ["custom"],
                                    index=GEMMA_MODELS.index(env.model) if env.model in GEMMA_MODELS else len(GEMMA_MODELS))
        model = st.text_input("Custom model id", env.model) if model_choice == "custom" else model_choice
        max_images = st.slider("Page images per paper", 0, 12, env.max_images_per_paper)
        dpi = st.slider("Render DPI", 72, 200, env.render_dpi, step=8)
        baseline = st.checkbox("Also run text-only baseline", True,
                               help="Runs the Paper Analyst a second time without images, for the text vs multimodal comparison.")
    st.divider()
    st.caption("Gaps are relative to the uploaded papers only. ResearchGap never claims that nobody has studied a question.")

settings = Settings(api_key=api_key or "")
if mode.startswith("Live"):
    settings.model, settings.max_images_per_paper, settings.render_dpi = model, max_images, dpi
    settings.run_text_only_baseline = baseline

# ---------------------------------------------------------------- header + inputs
st.title("🔎 ResearchGap")
st.markdown("Upload up to three papers. Four Gemma 4 agents read the text **and the figures**, compare the evidence, "
            "propose research gaps, and verify every gap against the exact page it came from.")

c1, c2 = st.columns([3, 1])
with c1:
    uploads = st.file_uploader("Research papers (PDF, up to 3)", type="pdf", accept_multiple_files=True)
with c2:
    st.write("")
    st.write("")
    if st.button("Use sample papers", use_container_width=True,
                 help="Three short synthetic papers about AI code review, built to show what figures reveal."):
        st.session_state["files"] = build_sample_papers()
        st.session_state.pop("result", None)

if uploads:
    files = [(u.name, u.getvalue()) for u in uploads[:3]]
    if len(uploads) > 3:
        st.warning("Only the first three PDFs are used.")
    if files != st.session_state.get("files"):
        st.session_state["files"] = files
        st.session_state.pop("result", None)

files = st.session_state.get("files", [])
papers = [_parse(pid, name, data) for pid, (name, data) in zip(PAPER_IDS, files)]

if not papers:
    st.info("Upload PDFs or click **Use sample papers** to begin.")
    st.stop()

with st.expander(f"📄 {len(papers)} paper(s) loaded: text extracted, figure/table pages detected", expanded=False):
    cols = st.columns(len(papers))
    for col, p in zip(cols, papers):
        with col:
            st.markdown(f"**Paper {p.paper_id}** · {p.title}")
            vis = p.visual_pages(settings.max_images_per_paper or 1)
            st.caption(f"{p.page_count} pages · {sum(len(pg.text) for pg in p.pages):,} chars · "
                       f"{len(vis)} figure/table page(s) sent to Gemma")
            for pg in vis[:2]:
                st.image(_render(p.pdf_bytes, pg.number, 60), caption=f"p.{pg.number}: " + ("; ".join(pg.captions)[:120] or "visual content"))

is_sample = {p.filename for p in papers} == SAMPLE_NAMES
run_label = "Run agents" if mode.startswith("Live") else "Show offline demo"
if st.button(run_label, type="primary"):
    if mode.startswith("Live"):
        if not settings.api_key:
            st.error("Add a Gemini API key in the sidebar or in `.env` first.")
            st.stop()
        with st.status("Running ResearchGap agents...", expanded=True) as status:
            bar = st.progress(0.0)

            def progress(msg: str, frac: float):
                status.write(msg)
                bar.progress(min(frac, 1.0))

            try:
                st.session_state["result"] = run_pipeline(papers, settings, progress)
                status.update(label="Analysis complete", state="complete", expanded=False)
            except LLMError as e:
                status.update(label="Analysis failed", state="error")
                st.error(str(e))
                st.stop()
    else:
        if not is_sample:
            st.error("Offline demo only covers the bundled sample papers. Click **Use sample papers**, or switch to Live mode.")
            st.stop()
        st.session_state["result"] = demo_result(papers)

result: RunResult | None = st.session_state.get("result")
if result is None:
    st.stop()

by_id = {p.paper_id: p for p in papers}
if result.is_demo:
    st.warning("Offline demo: precomputed illustrative output for the synthetic sample papers. "
               "Citation checks below are real; the agent outputs are not a live Gemma run.")

# ---------------------------------------------------------------- results
verdicts = [v.verdict for v in result.verifications.values()]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers", len(result.analyses))
m2.metric("Gaps proposed", len(result.gaps))
m3.metric("Verified (supported)", verdicts.count("supported"))
m4.metric("Rejected by Verifier", verdicts.count("rejected"))

tab_gaps, tab_matrix, tab_visual, tab_tvm, tab_papers, tab_export = st.tabs(
    ["🧭 Research gaps", "📊 Comparison matrix", "📈 Experimental results", "👁️ Text-only vs multimodal",
     "📄 Paper analyses", "⬇️ Export"])


def evidence_block(ref: EvidenceRef, key: str, status: str | None = None, note: str = ""):
    icon = {"found": "✅", "page_only": "🟡", "not_found": "❌", "invalid_ref": "❌"}.get(status or "", "•")
    kind = {"figure": "🖼️ figure", "table": "🧮 table", "text": "📝 text"}[ref.source_type]
    st.markdown(f"{icon} **Paper {ref.paper_id} — page {ref.page}"
                + (f" — {ref.figure_or_table}" if ref.figure_or_table else "") + f"** · {kind}")
    st.markdown(f"> {ref.quote_or_observation}")
    if note:
        st.caption(note)
    paper = by_id.get(ref.paper_id)
    if paper and ref.page and 1 <= ref.page <= paper.page_count:
        if st.toggle("Show source page", key=key):
            hl = (ref.quote_or_observation,) if ref.source_type == "text" else ()
            st.image(_render(paper.pdf_bytes, ref.page, 90, hl),
                     caption=f"Paper {ref.paper_id}, page {ref.page}" + (" (quote highlighted)" if hl else ""))


with tab_gaps:
    st.caption("Click a gap to see the exact evidence behind it. Each citation was checked automatically against "
               "the PDF (✅ located, 🟡 page exists but not matched verbatim, ❌ not found) and then judged by the Verifier agent.")
    show_rejected = st.checkbox("Show gaps rejected by the Verifier", value=True)
    order = {"supported": 0, "weak": 1, "rejected": 2}
    for gap in sorted(result.gaps, key=lambda g: order.get(result.verifications[g.id].verdict, 3)):
        v = result.verifications[gap.id]
        if v.verdict == "rejected" and not show_rejected:
            continue
        badge = {"supported": "Evidence-supported", "weak": "Needs investigation", "rejected": "Rejected"}[v.verdict]
        with st.expander(f"{gap.id} · {gap.title}  —  {badge}" + ("  · 🖼️ from figures" if gap.derived_from_visual else ""),
                         expanded=v.verdict == "supported" and gap.id == result.gaps[0].id):
            st.markdown(f"<span class='rg-badge rg-{v.verdict}'>{badge} · {v.confidence:.0%}</span> "
                        f"<span class='rg-muted'>{gap.gap_type.replace('_', ' ')}</span>", unsafe_allow_html=True)
            st.markdown(f"### {gap.question}")
            st.write(gap.rationale)
            st.markdown("**Supporting evidence**")
            checks = {c.index: c for c in v.evidence_checks}
            for i, ref in enumerate(gap.evidence):
                c = checks.get(i)
                with st.container(border=True):
                    evidence_block(ref, f"ev-{gap.id}-{i}", c.status if c else None, c.note if c else "")
            st.markdown("**Suggested experiment**")
            st.info(gap.suggested_experiment)
            st.markdown(f"**Verifier:** {v.reasons}")
            st.caption("This is a gap in the uploaded paper set, not proof that no prior research exists. "
                       "Confidence depends on the completeness of the uploaded literature.")

with tab_matrix:
    comp = result.comparison
    if comp.matrix:
        df = pd.DataFrame([{"Dimension": r.dimension, **{f"Paper {p}": r.values.get(p, "—") for p in result.analyses}}
                           for r in comp.matrix]).set_index("Dimension")
        st.dataframe(df, use_container_width=True)
    if comp.metric_conflicts:
        st.subheader("⚠️ Metric definitions differ")
        for m in comp.metric_conflicts:
            st.warning(f"**{m.metric}** ({', '.join(m.papers)}): {m.explanation}")
    a_col, c_col = st.columns(2)
    with a_col:
        st.subheader("Agreements")
        for pt in comp.agreements:
            st.markdown(f"- **{', '.join(pt.papers)}**: {pt.description}")
            for e in pt.evidence:
                st.caption("  " + cite(e))
    with c_col:
        st.subheader("Contradictions")
        for pt in comp.contradictions:
            st.markdown(f"- **{', '.join(pt.papers)}**: {pt.description}")
            for e in pt.evidence:
                st.caption("  " + cite(e))
    if comp.unstudied_dimensions:
        st.subheader("Not studied in the uploaded papers")
        for u in comp.unstudied_dimensions:
            st.markdown(f"- {u}")

with tab_visual:
    rows = [{"Paper": pid, "Metric": r.metric, "Condition": r.condition, "Method": r.method, "Value": r.value,
             "Unit": r.unit or "", "Source": r.source.figure_or_table or f"p.{r.source.page} text",
             "From": r.source.source_type}
            for pid, a in result.analyses.items() for r in a.reported_results]
    if not rows:
        st.info("No numeric results were extracted.")
    else:
        df = pd.DataFrame(rows)
        metric = st.selectbox("Metric", sorted(df["Metric"].unique()))
        sub = df[df["Metric"] == metric]
        conflict = next((m for m in result.comparison.metric_conflicts if m.metric.lower() == metric.lower()), None)
        if conflict:
            st.warning(f"'{metric}' is defined differently across papers: {conflict.explanation}")
        fig = px.bar(sub, x="Condition", y="Value", color="Method", facet_col="Paper", barmode="group",
                     text="Value", hover_data=["Source", "From"])
        fig.update_xaxes(matches=None, title=None)
        fig.update_layout(height=430, margin=dict(t=40, b=10), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Values with From = figure/table were read by Gemma from the rendered page images.")
        st.dataframe(sub, use_container_width=True, hide_index=True)

with tab_tvm:
    if not result.text_only_analyses:
        st.info("Enable 'Also run text-only baseline' in the sidebar to compare against a text-only analysis.")
    st.markdown("What a text-only reader concludes vs what Gemma 4 sees after inspecting the actual figures.")
    for tv in result.text_vs_visual:
        st.subheader(f"Paper {tv.paper_id}: {result.analyses[tv.paper_id].title}")
        left, right = st.columns(2)
        with left:
            st.markdown("**📝 Text-only analysis concludes**")
            for f in tv.text_only_findings or ["(baseline not run)"]:
                st.markdown(f"- {f}")
        with right:
            st.markdown("**🖼️ Multimodal analysis adds**")
            for f in tv.multimodal_only_insights or ["No figure-only insights."]:
                st.markdown(f"- {f}")
        for vf in tv.contradictions_revealed:
            st.error(f"**{vf.figure_or_table} contradicts the text.** Claim: “{vf.related_text_claim}”. "
                     f"Figure shows: {vf.observation}")
            if st.toggle("Show figure", key=f"tvm-{tv.paper_id}-{vf.figure_or_table}"):
                p = by_id.get(tv.paper_id)
                if p and 1 <= vf.page <= p.page_count:
                    st.image(_render(p.pdf_bytes, vf.page, 90), caption=f"Paper {tv.paper_id}, page {vf.page}")
        st.divider()

with tab_papers:
    for pid, a in result.analyses.items():
        with st.expander(f"Paper {pid}: {a.title}", expanded=False):
            st.markdown(f"**Research question:** {a.research_question}")
            st.markdown(f"**Method:** {a.method}  ·  **Participants:** {a.participants if a.participants is not None else 'n/a'}")
            if a.datasets_or_settings:
                st.markdown("**Datasets / settings:** " + ", ".join(a.datasets_or_settings))
            if a.metrics:
                st.markdown("**Metrics**")
                st.table(pd.DataFrame([m.model_dump() for m in a.metrics]))
            st.markdown("**Key findings**")
            for f in a.key_findings:
                st.markdown(f"- {f.statement}  " + " ".join(f"`{e.paper_id} p.{e.page}{' ' + e.figure_or_table if e.figure_or_table else ''}`" for e in f.evidence))
            st.markdown("**Limitations**")
            for f in a.limitations:
                st.markdown(f"- {f.statement}")
            if a.visual_findings:
                st.markdown("**Visual findings**")
                for vf in a.visual_findings:
                    st.markdown(f"- **{vf.figure_or_table}** (p.{vf.page}, {vf.relation_to_text}): {vf.observation}")
            st.json(a.model_dump(), expanded=False)

with tab_export:
    st.download_button("Download report (Markdown)", to_markdown(result), "researchgap_report.md", "text/markdown")
    st.download_button("Download raw results (JSON)", result.model_dump_json(indent=2), "researchgap_results.json",
                       "application/json")
    st.caption(f"Model: {result.model}")
