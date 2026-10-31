"""Precomputed, hand-written output for the bundled synthetic sample papers.

Used by "Offline demo" mode so the dashboard can be explored without an API key.
It shows what the pipeline is designed to produce; it is NOT a live model run,
and the UI labels it that way.
"""
from __future__ import annotations

from .pdf_processing import ParsedPaper
from .pipeline import build_text_vs_visual, check_evidence, combine_verdict
from .schemas import (
    Comparison,
    CrossPaperPoint,
    EvidenceRef,
    Finding,
    MatrixRow,
    MetricConflict,
    MetricDefinition,
    PaperAnalysis,
    ResearchGap,
    ResultPoint,
    RunResult,
    VisualFinding,
)


def T(pid: str, page: int, quote: str) -> EvidenceRef:
    return EvidenceRef(paper_id=pid, page=page, quote_or_observation=quote, source_type="text")


def F(pid: str, page: int, label: str, obs: str, kind: str = "figure") -> EvidenceRef:
    return EvidenceRef(paper_id=pid, page=page, figure_or_table=label, quote_or_observation=obs, source_type=kind)


def R(metric, cond, method, value, unit, src) -> ResultPoint:
    return ResultPoint(metric=metric, condition=cond, method=method, value=value, unit=unit, source=src)


def _analyses(multimodal: bool) -> dict[str, PaperAnalysis]:
    mode = "multimodal" if multimodal else "text"
    fig2 = F("A", 2, "Figure 2", "AI condition beats Baseline on web service (78 vs 62) and data pipeline (74 vs 58) but is worse on embedded firmware (64 vs 71).")
    a = PaperAnalysis(
        paper_id="A", mode=mode,
        title="Do LLM Suggestions Help Code Reviewers? A Controlled Experiment",
        research_question="Do inline LLM suggestions improve defect detection and review speed in code review?",
        method="Controlled within-subject experiment with seeded defects",
        participants=30,
        datasets_or_settings=["Web service", "Data pipeline", "Embedded firmware"],
        metrics=[MetricDefinition(name="accuracy", definition="Percentage of seeded defects that a reviewer correctly identified"),
                 MetricDefinition(name="review time", definition="Minutes from opening the pull request to submitting the verdict")],
        key_findings=[
            Finding(statement="AI assistance improved defect detection accuracy across all three codebases.",
                    evidence=[T("A", 1, "AI assistance improved defect detection accuracy across all three codebases")]),
            Finding(statement="Overall accuracy rose from 63.7% to 72.0% with AI assistance.",
                    evidence=[T("A", 2, "AI assistance improved accuracy from 63.7% to 72.0%")]),
            Finding(statement="Review time fell by about 21% (18.4 to 14.5 minutes).",
                    evidence=[F("A", 2, "Table 1", "Baseline 18.4 min, AI condition 14.5 min", "table")]),
        ],
        limitations=[
            Finding(statement="Small sample of 30 participants from a single company.",
                    evidence=[T("A", 2, "The sample of 30 participants is small")]),
            Finding(statement="Over-trust in incorrect suggestions was not studied.",
                    evidence=[T("A", 2, "We did not study whether reviewers over-trust incorrect suggestions")]),
        ],
        reported_results=[
            R("accuracy", "Overall", "Baseline", 63.7, "%", T("A", 2, "AI assistance improved accuracy from 63.7% to 72.0%")),
            R("accuracy", "Overall", "AI condition", 72.0, "%", T("A", 2, "AI assistance improved accuracy from 63.7% to 72.0%")),
        ],
        visual_evidence=["Figure 2", "Table 1"],
    )
    b = PaperAnalysis(
        paper_id="B", mode=mode,
        title="Reading AI-Written Code: An Eye-Tracking Study of Code Reviewers",
        research_question="Do reviewers read and judge AI-written code differently from human-written code?",
        method="Eye-tracking study, blinded to code source",
        participants=24,
        datasets_or_settings=["16 short Python code changes (8 human, 8 AI)"],
        metrics=[MetricDefinition(name="accuracy", definition="Percentage of accept/reject decisions that matched the ground-truth label"),
                 MetricDefinition(name="review time", definition="Minutes per code change"),
                 MetricDefinition(name="fixation duration", definition="Total fixation seconds on changed lines")],
        key_findings=[
            Finding(statement="Decision accuracy was similar for human-written (81%) and AI-written (79%) code.",
                    evidence=[T("B", 2, "Decision accuracy was 81% for human-written changes and 79% for AI-written changes")]),
            Finding(statement="Reviewers spent less time reading AI-generated code.",
                    evidence=[T("B", 2, "Reviewers spent less time reading AI-generated code")]),
        ],
        limitations=[Finding(statement="Students only, Python only, and short changes under 60 lines.",
                             evidence=[T("B", 2, "Participants were students, all code was Python")])],
        reported_results=[
            R("accuracy", "Human-written code", "Human reviewers", 81, "%", T("B", 2, "Decision accuracy was 81% for human-written changes")),
            R("accuracy", "AI-written code", "Human reviewers", 79, "%", T("B", 2, "79% for AI-written changes")),
        ],
        visual_evidence=["Figure 1"],
    )
    c = PaperAnalysis(
        paper_id="C", mode=mode,
        title="Can Reviewers Catch Bad Agent Patches? A Benchmark Evaluation",
        research_question="How accurately can an LLM review model judge patches produced by coding agents?",
        method="Benchmark evaluation on 500 agent-generated patches",
        participants=None,
        datasets_or_settings=["500 agent patches for real GitHub issues (Python)", "Small / medium / large patches"],
        metrics=[MetricDefinition(name="accuracy", definition="Percentage of patches where the review model's approve/reject verdict matches the hidden test outcome")],
        key_findings=[
            Finding(statement="The review model reached 81% accuracy overall.",
                    evidence=[T("C", 1, "The review model achieved 81% accuracy overall")]),
            Finding(statement="The authors claim the model is robust to patch size.",
                    evidence=[T("C", 2, "Accuracy remains high across patch sizes")]),
        ],
        limitations=[
            Finding(statement="Only Python repositories.", evidence=[T("C", 2, "Only Python repositories were included")]),
            Finding(statement="The reviewer had no access to execution traces or test logs.",
                    evidence=[T("C", 2, "the review model did not have access to the agent's execution traces or test logs")]),
            Finding(statement="Review time was not measured.", evidence=[T("C", 1, "We did not measure review time")]),
        ],
        reported_results=[R("accuracy", "Overall", "Review model", 81, "%", T("C", 2, "Overall accuracy was 81%"))],
        visual_evidence=["Figure 3"],
    )
    if multimodal:
        fig1 = F("B", 2, "Figure 1", "Incorrect decisions on AI-written code had only 17 s of fixation, far below every other cell (48, 39, 35 s).")
        fig3 = F("C", 2, "Figure 3", "Accuracy falls from 90% (small) to 79% (medium) to 50% (large patches), i.e. chance level for large patches.")
        a.visual_findings = [VisualFinding(figure_or_table="Figure 2", page=2, observation=fig2.quote_or_observation,
                                           relation_to_text="contradicts", related_text_claim="improved defect detection accuracy across all three codebases")]
        a.reported_results += [
            R("accuracy", "Web service", "Baseline", 62, "%", fig2), R("accuracy", "Web service", "AI condition", 78, "%", fig2),
            R("accuracy", "Data pipeline", "Baseline", 58, "%", fig2), R("accuracy", "Data pipeline", "AI condition", 74, "%", fig2),
            R("accuracy", "Embedded firmware", "Baseline", 71, "%", fig2), R("accuracy", "Embedded firmware", "AI condition", 64, "%", fig2),
        ]
        a.key_findings.append(Finding(statement="Figure 2 shows AI assistance lowered accuracy on embedded firmware (71% to 64%).", evidence=[fig2]))
        b.visual_findings = [VisualFinding(figure_or_table="Figure 1", page=2, observation=fig1.quote_or_observation,
                                           relation_to_text="adds_detail", related_text_claim="Reviewers spent less time reading AI-generated code")]
        b.reported_results += [
            R("fixation (s)", "Correct decisions", "Human-written", 48, "s", fig1), R("fixation (s)", "Incorrect decisions", "Human-written", 39, "s", fig1),
            R("fixation (s)", "Correct decisions", "AI-written", 35, "s", fig1), R("fixation (s)", "Incorrect decisions", "AI-written", 17, "s", fig1),
        ]
        b.key_findings.append(Finding(statement="Wrong decisions on AI code coincide with very low attention (17 s fixation).", evidence=[fig1]))
        c.visual_findings = [VisualFinding(figure_or_table="Figure 3", page=2, observation=fig3.quote_or_observation,
                                           relation_to_text="contradicts", related_text_claim="Accuracy remains high across patch sizes")]
        c.reported_results += [R("accuracy", "Small patches", "Review model", 90, "%", fig3),
                               R("accuracy", "Medium patches", "Review model", 79, "%", fig3),
                               R("accuracy", "Large patches", "Review model", 50, "%", fig3)]
        c.key_findings.append(Finding(statement="Figure 3 shows accuracy collapses to 50% on large patches.", evidence=[fig3]))
    return {"A": a, "B": b, "C": c}


def demo_result(papers: list[ParsedPaper]) -> RunResult:
    analyses = _analyses(True)
    text_only = _analyses(False)
    comparison = Comparison(
        matrix=[
            MatrixRow(dimension="Focus", values={"A": "AI-assisted review", "B": "Human review of AI code", "C": "Agent-generated patches"}),
            MatrixRow(dimension="Method", values={"A": "Controlled experiment", "B": "Eye-tracking study", "C": "Benchmark evaluation"}),
            MatrixRow(dimension="Participants / data", values={"A": "30 professionals", "B": "24 students", "C": "500 patches"}),
            MatrixRow(dimension="Accuracy measured?", values={"A": "Yes (defects found)", "B": "Yes (decision correct)", "C": "Yes (verdict = tests)"}),
            MatrixRow(dimension="Review time measured?", values={"A": "Yes", "B": "Yes", "C": "No"}),
            MatrixRow(dimension="Languages / domains", values={"A": "Web, data, firmware", "B": "Python", "C": "Python"}),
            MatrixRow(dimension="Results split by condition?", values={"A": "Yes (Fig 2)", "B": "Yes (Fig 1)", "C": "Yes (Fig 3)"}),
            MatrixRow(dimension="Agent execution traces studied?", values={"A": "No", "B": "No", "C": "No (listed as limitation)"}),
        ],
        agreements=[CrossPaperPoint(description="AI involvement reduces the time humans spend reviewing.", papers=["A", "B"],
                                    evidence=[F("A", 2, "Table 1", "Review time 18.4 vs 14.5 min", "table"),
                                              T("B", 2, "Reviewers spent less time reading AI-generated code")])],
        contradictions=[CrossPaperPoint(
            description="Both A and C claim uniform benefits, but their own figures show performance drops in one condition (firmware in A, large patches in C).",
            papers=["A", "C"],
            evidence=[F("A", 2, "Figure 2", "AI condition worse on embedded firmware (64 vs 71)"),
                      F("C", 2, "Figure 3", "Accuracy 50% on large patches")])],
        metric_conflicts=[MetricConflict(metric="accuracy", papers=["A", "B", "C"],
                                         explanation="A counts seeded defects found, B counts correct accept/reject decisions, C compares model verdicts with test outcomes. The 72%, 79% and 81% figures are not comparable.")],
        unstudied_dimensions=["Not studied in the uploaded papers: giving reviewers agent execution traces or test logs.",
                              "Not studied in the uploaded papers: over-trust in incorrect AI suggestions.",
                              "Not studied in the uploaded papers: non-Python or safety-critical code with AI review."],
    )
    gaps = [
        ResearchGap(id="G1", title="Execution traces for reviewing agent patches",
                    question="Does access to coding-agent execution traces improve the accuracy of AI code reviewers?",
                    rationale="Not studied in the uploaded papers. Paper C names the missing traces as a limitation, and no paper gives reviewers traces.",
                    gap_type="untested_intervention", suggested_experiment="Review the 500-patch benchmark twice: diff only vs diff plus execution-trace summary; compare accuracy by patch size.",
                    evidence=[T("C", 2, "the review model did not have access to the agent's execution traces or test logs"),
                              T("C", 1, "The review model sees only the issue text and the code diff")]),
        ResearchGap(id="G2", title="Where AI review help turns into harm",
                    question="Under what codebase conditions does LLM assistance stop improving, or start hurting, defect detection?",
                    rationale="Paper A claims gains across all codebases, but Figure 2 shows accuracy dropped on embedded firmware. The uploaded papers do not explain why.",
                    gap_type="inconsistent_results", derived_from_visual=True,
                    suggested_experiment="Replicate A with several low-level / safety-critical codebases and log which suggestion types were accepted when defects were missed.",
                    evidence=[F("A", 2, "Figure 2", "AI condition worse on embedded firmware (64 vs 71) while better on the other two codebases"),
                              T("A", 1, "AI assistance improved defect detection accuracy across all three codebases")]),
        ResearchGap(id="G3", title="AI review of large patches",
                    question="How can AI reviewers maintain accuracy on large agent-generated patches?",
                    rationale="Paper C reports robustness to patch size, but Figure 3 shows accuracy of 50% on patches over 200 lines. Paper B only used changes under 60 lines.",
                    gap_type="inconsistent_results", derived_from_visual=True,
                    suggested_experiment="Compare whole-diff review with chunked or test-guided review on large patches, with balanced patch-size strata.",
                    evidence=[F("C", 2, "Figure 3", "Accuracy 90% small, 79% medium, 50% large"),
                              T("B", 2, "code changes were short (under 60 lines)")]),
        ResearchGap(id="G4", title="Attention and over-trust in AI-written code",
                    question="Does reduced visual attention to AI-written code cause reviewers to approve faulty AI changes?",
                    rationale="Paper B's Figure 1 links wrong decisions on AI code with very low fixation, and Paper A did not study over-trust. Causality is untested in the uploaded papers.",
                    gap_type="unaddressed_limitation", derived_from_visual=True,
                    suggested_experiment="Eye-tracking experiment that labels code source vs hides it, with seeded defects, measuring fixation and approval of faulty changes.",
                    evidence=[F("B", 2, "Figure 1", "Incorrect AI-code decisions: 17 s fixation vs 35-48 s elsewhere"),
                              T("A", 2, "We did not study whether reviewers over-trust incorrect suggestions")]),
        ResearchGap(id="G5", title="Universal benefit across all languages",
                    question="Do AI reviewers already work equally well in every programming language?",
                    rationale="Proposed to show the verifier: none of the cited text exists in the papers.",
                    gap_type="population_or_dataset",
                    suggested_experiment="(rejected example)",
                    evidence=[T("B", 3, "AI review is equally accurate for all programming languages")]),
    ]
    by_id = {p.paper_id: p for p in papers}
    llm_verdicts = {
        "G1": ("supported", 0.86, "Paper C p.2 states the reviewer lacked execution traces; p.1 confirms diff-only input. No paper tests traces."),
        "G2": ("supported", 0.82, "Figure 2 (A p.2) shows 71 to 64 on firmware, contradicting the 'all three codebases' claim on p.1."),
        "G3": ("supported", 0.8, "Figure 3 (C p.2) shows 50% on large patches; B's changes were under 60 lines, so neither covers large patches well."),
        "G4": ("weak", 0.55, "Figure 1 shows the association, but with 24 students and 8 AI changes the effect may not be robust; still untested causally."),
        "G5": ("rejected", 0.05, "The cited page does not exist."),
    }
    verifications = {g.id: combine_verdict(g, check_evidence(g, by_id), *llm_verdicts[g.id]) for g in gaps}
    return RunResult(
        papers={p.paper_id: p.filename for p in papers},
        analyses=analyses, text_only_analyses=text_only, comparison=comparison, gaps=gaps,
        verifications=verifications, text_vs_visual=build_text_vs_visual(analyses, text_only),
        model="offline demo (precomputed)", is_demo=True,
    )
