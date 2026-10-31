"""The four ResearchGap agents. One Gemma 4 model, a different prompt per role."""
from __future__ import annotations

import json

from .llm import GemmaClient, ImagePart
from .pdf_processing import ParsedPaper, render_page
from .schemas import (
    Comparison,
    EvidenceCheck,
    GapList,
    PaperAnalysis,
    ResearchGap,
    VerifierOutput,
)

GROUNDING_RULES = """Rules:
- Use only the material provided. Never use outside knowledge about these papers or the field.
- Every finding, limitation and result must cite evidence with paper_id and a 1-based page number.
- For text evidence, quote_or_observation must be a short VERBATIM quote (5-25 words) copied from that page.
- For figure/table evidence, set source_type to "figure" or "table", set figure_or_table, and describe what is shown.
- If something is not reported, leave it null or empty. Do not guess."""


def paper_images(paper: ParsedPaper, limit: int, dpi: int) -> list[ImagePart]:
    images = []
    for p in paper.visual_pages(limit):
        caps = "; ".join(p.captions) or "no caption detected"
        images.append(
            ImagePart(
                label=f"Paper {paper.paper_id}, page {p.number} ({caps})",
                data=render_page(paper.pdf_bytes, p.number, dpi=dpi),
            )
        )
    return images


class PaperAnalyst:
    """Agent 1: extracts research question, method, findings, limitations and results."""

    def __init__(self, llm: GemmaClient):
        self.llm = llm

    def run(self, paper: ParsedPaper, multimodal: bool, max_images: int, dpi: int) -> PaperAnalysis:
        images = paper_images(paper, max_images, dpi) if multimodal else []
        if multimodal:
            mode_note = (
                f"You also receive {len(images)} rendered page images containing the paper's figures and tables. "
                "Read the charts and tables yourself: note values per condition, trends, and anything the written "
                "text does not mention. For every figure/table you inspect, add a visual_findings entry and say "
                "whether it agrees with, contradicts, or adds detail to a specific written claim. A claim like "
                "'improved across all conditions' that a chart shows is false for one condition is a contradiction. "
                "Put numbers you read from charts or tables in reported_results with source_type figure/table."
            )
        else:
            mode_note = (
                "TEXT-ONLY MODE: you only have the extracted text, no images. You cannot see figures. "
                "Leave visual_findings empty and only report numbers that appear in the text."
            )
        prompt = f"""You are the Paper Analyst in ResearchGap, a research assistant that compares scientific papers.
Analyze paper {paper.paper_id} (file: {paper.filename}). Set paper_id to "{paper.paper_id}" and mode to "{'multimodal' if multimodal else 'text'}".

{mode_note}

Extract: title, research_question, method, participants (number of human participants if any),
datasets_or_settings, metrics with the paper's own definition of each, key_findings, limitations
(stated by the authors AND ones evident from the design, e.g. small sample), and reported_results.

{GROUNDING_RULES}

Paper text with page markers:
{paper.tagged_text()}"""
        analysis = self.llm.structured(prompt, PaperAnalysis, images)
        analysis.paper_id = paper.paper_id
        analysis.mode = "multimodal" if multimodal else "text"
        if not multimodal:
            analysis.visual_findings = []
        return analysis


class Comparator:
    """Agent 2: cross-paper evidence comparison."""

    def __init__(self, llm: GemmaClient):
        self.llm = llm

    def run(self, analyses: dict[str, PaperAnalysis]) -> Comparison:
        ids = list(analyses)
        payload = json.dumps({k: v.model_dump() for k, v in analyses.items()}, indent=1)
        prompt = f"""You are the Cross-Paper Evidence Comparator in ResearchGap.
You receive structured analyses of papers {', '.join(ids)}. Compare them.

1. matrix: 6-10 rows. Include Focus, Method, Participants/data, each important metric as "<metric> measured?",
   and any dimension that some papers study and others do not. values maps every paper id ({', '.join(ids)}) to a short cell.
2. agreements and contradictions across papers, each with evidence copied from the analyses.
   A figure in one paper contradicting another paper's claim counts.
3. metric_conflicts: metrics with the same name but different definitions, so numbers cannot be compared directly.
4. unstudied_dimensions: relevant variables, populations, conditions or interventions that NONE of these papers examine.
   Phrase them as "Not studied in the uploaded papers: ...".

Only use the analyses below. Copy evidence refs from them; do not invent pages or quotes.

Analyses:
{payload}"""
        return self.llm.structured(prompt, Comparison)


class GapFinder:
    """Agent 3: proposes research gaps with traceable evidence."""

    def __init__(self, llm: GemmaClient):
        self.llm = llm

    def run(self, analyses: dict[str, PaperAnalysis], comparison: Comparison, max_gaps: int = 5) -> list[ResearchGap]:
        payload = json.dumps(
            {"analyses": {k: v.model_dump() for k, v in analyses.items()}, "comparison": comparison.model_dump()},
            indent=1,
        )
        prompt = f"""You are the Research Gap Investigator in ResearchGap.
Using the paper analyses and cross-paper comparison, propose up to {max_gaps} research gaps.

Good gaps come from: interventions none of the papers test, results that are inconsistent across conditions
(especially ones only visible in figures), limitations no paper addresses, metric definitions that prevent
comparison, and populations or datasets that are missing.

For each gap:
- id "G1", "G2", ...; a short title; the open question phrased as a question.
- rationale: say "not studied in the uploaded papers", never "nobody has studied this".
- evidence: 1-4 refs copied exactly from the analyses (paper_id, page, figure_or_table, quote_or_observation, source_type).
  A gap with no identifiable evidence must not be proposed.
- suggested_experiment: concrete design (arms, sample, metric).
- derived_from_visual: true if it relies on a figure/table observation.

Material:
{payload}"""
        gaps = self.llm.structured(prompt, GapList).gaps
        for i, gap in enumerate(gaps, 1):
            gap.id = f"G{i}"
        return gaps


class Verifier:
    """Agent 4: independent check that each gap is supported by the cited evidence."""

    def __init__(self, llm: GemmaClient):
        self.llm = llm

    def run(self, gap: ResearchGap, checks: list[EvidenceCheck], papers: dict[str, ParsedPaper],
            dpi: int) -> VerifierOutput:
        excerpts, images, seen = [], [], set()
        for ref, check in zip(gap.evidence, checks):
            paper = papers.get(ref.paper_id)
            if not paper or not ref.page or not 1 <= ref.page <= paper.page_count:
                continue
            key = (ref.paper_id, ref.page)
            if key in seen:
                continue
            seen.add(key)
            excerpts.append(f"[Paper {ref.paper_id}, page {ref.page}]\n{paper.pages[ref.page - 1].text[:3500]}")
            if ref.source_type in ("figure", "table") and len(images) < 4:
                images.append(
                    ImagePart(f"Paper {ref.paper_id}, page {ref.page}", render_page(paper.pdf_bytes, ref.page, dpi=dpi))
                )
        check_lines = "\n".join(
            f"- evidence[{c.index}] ({gap.evidence[c.index].paper_id} p.{gap.evidence[c.index].page}): {c.status}. {c.note}"
            for c in checks
        )
        prompt = f"""You are the independent Verification Agent in ResearchGap. Be skeptical.
Decide whether the proposed research gap is genuinely supported by the cited evidence from the uploaded papers.

verdict:
- "supported": the cited pages/figures clearly show what the gap claims, and the question is genuinely open within these papers.
- "weak": partly supported, overstated, or the evidence is indirect.
- "rejected": the evidence does not exist, does not say this, or the papers actually address the question.
Also check figure interpretations against the paper's own tables and written conclusions where possible.
confidence is 0-1. reasons: 1-3 sentences naming the specific pages/figures you checked.

Proposed gap:
{gap.model_dump_json(indent=1)}

Automatic citation checks (quote matched on page = found; page exists but quote not matched = page_only):
{check_lines or '- no evidence cited'}

Cited page text:
{chr(10).join(excerpts) or '(none)'}"""
        return self.llm.structured(prompt, VerifierOutput, images)
