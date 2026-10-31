"""Orchestrates the agents: analyse -> compare -> find gaps -> verify."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .agents import Comparator, GapFinder, PaperAnalyst, Verifier
from .config import Settings
from .llm import GemmaClient
from .pdf_processing import ParsedPaper, quote_in_page
from .schemas import (
    EvidenceCheck,
    PaperAnalysis,
    ResearchGap,
    RunResult,
    TextVsVisual,
    Verification,
)

Progress = Callable[[str, float], None]


def check_evidence(gap: ResearchGap, papers: dict[str, ParsedPaper]) -> list[EvidenceCheck]:
    """Deterministic citation check: does the paper/page exist, and is the quote really there?"""
    checks = []
    for i, ref in enumerate(gap.evidence):
        paper = papers.get(ref.paper_id)
        if paper is None:
            checks.append(EvidenceCheck(index=i, status="invalid_ref", note=f"No paper '{ref.paper_id}' was uploaded."))
            continue
        if not ref.page or not 1 <= ref.page <= paper.page_count:
            checks.append(EvidenceCheck(index=i, status="invalid_ref",
                                        note=f"Page {ref.page} does not exist (paper has {paper.page_count})."))
            continue
        page = paper.pages[ref.page - 1]
        if ref.source_type == "text":
            if quote_in_page(paper.pdf_bytes, ref.page, ref.quote_or_observation):
                checks.append(EvidenceCheck(index=i, status="found", note="Quote located on the cited page."))
            else:
                checks.append(EvidenceCheck(index=i, status="page_only", note="Page exists but the quote was not matched verbatim."))
        else:
            label = (ref.figure_or_table or "").lower().replace(".", "")
            caption_hit = label and any(label in c.lower().replace(".", "") for c in page.captions)
            if caption_hit:
                checks.append(EvidenceCheck(index=i, status="found", note=f"{ref.figure_or_table} caption is on this page."))
            elif page.has_visual:
                checks.append(EvidenceCheck(index=i, status="page_only", note="Page has a figure/table, caption not matched."))
            else:
                checks.append(EvidenceCheck(index=i, status="not_found", note="No figure or table detected on this page."))
    return checks


def combine_verdict(gap: ResearchGap, checks: list[EvidenceCheck], llm_verdict: str, confidence: float,
                    reasons: str) -> Verification:
    """The deterministic check can veto the LLM: a gap with no locatable evidence is rejected."""
    usable = [c for c in checks if c.status in ("found", "page_only")]
    verdict = llm_verdict
    if not usable:
        verdict, confidence = "rejected", min(confidence, 0.1)
        reasons = "No cited evidence could be located in the uploaded papers. " + reasons
    elif verdict == "supported" and not any(c.status == "found" for c in checks):
        verdict = "weak"
        reasons = "Cited pages exist, but no quote or caption matched exactly. " + reasons
    return Verification(gap_id=gap.id, verdict=verdict, confidence=confidence, reasons=reasons, evidence_checks=checks)


def build_text_vs_visual(analyses: dict[str, PaperAnalysis], text_only: dict[str, PaperAnalysis]) -> list[TextVsVisual]:
    out = []
    for pid, mm in analyses.items():
        txt = text_only.get(pid)
        out.append(
            TextVsVisual(
                paper_id=pid,
                text_only_findings=[f.statement for f in (txt.key_findings if txt else [])],
                multimodal_only_insights=[
                    f"{v.figure_or_table} (p.{v.page}): {v.observation}"
                    for v in mm.visual_findings if v.relation_to_text in ("adds_detail", "contradicts")
                ],
                contradictions_revealed=[v for v in mm.visual_findings if v.relation_to_text == "contradicts"],
            )
        )
    return out


def run_pipeline(papers: list[ParsedPaper], settings: Settings, progress: Progress = lambda m, f: None) -> RunResult:
    llm = GemmaClient(settings)
    by_id = {p.paper_id: p for p in papers}
    analyst = PaperAnalyst(llm)

    jobs = [(p, True) for p in papers]
    if settings.run_text_only_baseline:
        jobs += [(p, False) for p in papers]
    analyses: dict[str, PaperAnalysis] = {}
    text_only: dict[str, PaperAnalysis] = {}
    progress(f"Paper Analyst reading {len(papers)} paper(s)...", 0.05)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(analyst.run, p, mm, settings.max_images_per_paper, settings.render_dpi): (p, mm)
            for p, mm in jobs
        }
        for done, fut in enumerate(as_completed(futures), 1):
            p, mm = futures[fut]
            (analyses if mm else text_only)[p.paper_id] = fut.result()
            progress(f"Analysed paper {p.paper_id} ({'multimodal' if mm else 'text-only'})", 0.05 + 0.45 * done / len(jobs))
    analyses = {p.paper_id: analyses[p.paper_id] for p in papers}

    progress("Comparator building the cross-paper matrix...", 0.55)
    comparison = Comparator(llm).run(analyses)

    progress("Gap Finder proposing research gaps...", 0.7)
    gaps = GapFinder(llm).run(analyses, comparison)

    verifier = Verifier(llm)
    verifications: dict[str, Verification] = {}
    for i, gap in enumerate(gaps, 1):
        progress(f"Verifier checking gap {gap.id}...", 0.75 + 0.23 * i / max(1, len(gaps)))
        checks = check_evidence(gap, by_id)
        if any(c.status in ("found", "page_only") for c in checks):
            out = verifier.run(gap, checks, by_id, settings.render_dpi)
            verifications[gap.id] = combine_verdict(gap, checks, out.verdict, out.confidence, out.reasons)
        else:
            verifications[gap.id] = combine_verdict(gap, checks, "rejected", 0.0, "")

    progress("Done", 1.0)
    return RunResult(
        papers={p.paper_id: p.filename for p in papers},
        analyses=analyses,
        text_only_analyses=text_only,
        comparison=comparison,
        gaps=gaps,
        verifications=verifications,
        text_vs_visual=build_text_vs_visual(analyses, text_only),
        model=settings.model,
    )
