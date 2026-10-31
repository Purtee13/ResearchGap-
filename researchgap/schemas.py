"""Pydantic models shared by every agent.

Every claim an agent makes carries one or more EvidenceRef objects so the UI can
trace a research gap back to the exact page, figure, or table it came from.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    paper_id: str = Field(description="Paper label, e.g. 'A', 'B' or 'C'")
    page: int | None = Field(default=None, description="1-based page number in that paper")
    figure_or_table: str | None = Field(default=None, description="e.g. 'Figure 2' or 'Table 1', if any")
    quote_or_observation: str = Field(
        description="A short verbatim quote from the page text, or what was observed in the figure/table"
    )
    source_type: Literal["text", "figure", "table"] = "text"


class Finding(BaseModel):
    statement: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class MetricDefinition(BaseModel):
    name: str
    definition: str = Field(description="How the paper defines or computes this metric")


class ResultPoint(BaseModel):
    """One numeric result read from text, a table, or a chart."""

    metric: str
    condition: str = Field(description="Dataset, codebase, subgroup or setting the number belongs to")
    method: str = Field(description="System or arm that produced the number, e.g. 'Baseline'")
    value: float
    unit: str | None = None
    source: EvidenceRef


class VisualFinding(BaseModel):
    figure_or_table: str
    page: int
    observation: str = Field(description="What the figure/table actually shows")
    relation_to_text: Literal["agrees", "contradicts", "adds_detail", "unclear"]
    related_text_claim: str | None = Field(
        default=None, description="The written claim this observation agrees with or contradicts"
    )


class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    research_question: str
    method: str
    participants: int | None = None
    datasets_or_settings: list[str] = Field(default_factory=list)
    metrics: list[MetricDefinition] = Field(default_factory=list)
    key_findings: list[Finding] = Field(default_factory=list)
    limitations: list[Finding] = Field(default_factory=list)
    reported_results: list[ResultPoint] = Field(default_factory=list)
    visual_findings: list[VisualFinding] = Field(default_factory=list)
    visual_evidence: list[str] = Field(default_factory=list, description="Figures/tables the analysis relied on")
    mode: Literal["text", "multimodal"] = "multimodal"


class MatrixRow(BaseModel):
    dimension: str = Field(description="Row label, e.g. 'Method' or 'Review time measured?'")
    values: dict[str, str] = Field(description="paper_id -> short cell value")


class CrossPaperPoint(BaseModel):
    description: str
    papers: list[str]
    evidence: list[EvidenceRef] = Field(default_factory=list)


class MetricConflict(BaseModel):
    metric: str
    papers: list[str]
    explanation: str = Field(description="Why the numbers cannot be compared directly")


class Comparison(BaseModel):
    matrix: list[MatrixRow] = Field(default_factory=list)
    agreements: list[CrossPaperPoint] = Field(default_factory=list)
    contradictions: list[CrossPaperPoint] = Field(default_factory=list)
    metric_conflicts: list[MetricConflict] = Field(default_factory=list)
    unstudied_dimensions: list[str] = Field(
        default_factory=list, description="Things none of the uploaded papers examine"
    )


GapType = Literal[
    "untested_intervention",
    "inconsistent_results",
    "unaddressed_limitation",
    "metric_mismatch",
    "population_or_dataset",
    "other",
]


class ResearchGap(BaseModel):
    id: str = Field(description="Short id such as 'G1'")
    title: str
    question: str = Field(description="The open research question, phrased as a question")
    rationale: str = Field(description="Why the uploaded papers leave this open")
    gap_type: GapType = "other"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    suggested_experiment: str
    derived_from_visual: bool = Field(
        default=False, description="True if the gap depends on something only visible in a figure/table"
    )


class GapList(BaseModel):
    gaps: list[ResearchGap] = Field(default_factory=list)


class EvidenceCheck(BaseModel):
    index: int
    status: Literal["found", "page_only", "not_found", "invalid_ref"]
    note: str = ""


class Verification(BaseModel):
    gap_id: str
    verdict: Literal["supported", "weak", "rejected"]
    confidence: float = Field(ge=0, le=1)
    reasons: str
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    verdict: Literal["supported", "weak", "rejected"]
    confidence: float = Field(ge=0, le=1)
    reasons: str


class TextVsVisual(BaseModel):
    paper_id: str
    text_only_findings: list[str]
    multimodal_only_insights: list[str]
    contradictions_revealed: list[VisualFinding]


class RunResult(BaseModel):
    papers: dict[str, str] = Field(description="paper_id -> file name")
    analyses: dict[str, PaperAnalysis]
    text_only_analyses: dict[str, PaperAnalysis] = Field(default_factory=dict)
    comparison: Comparison
    gaps: list[ResearchGap]
    verifications: dict[str, Verification]
    text_vs_visual: list[TextVsVisual] = Field(default_factory=list)
    model: str
    is_demo: bool = False
