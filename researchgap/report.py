"""Markdown export of a run."""
from __future__ import annotations

from .schemas import EvidenceRef, RunResult


def cite(ref: EvidenceRef) -> str:
    where = f"Paper {ref.paper_id}, p.{ref.page}" + (f", {ref.figure_or_table}" if ref.figure_or_table else "")
    return f"{where}: \"{ref.quote_or_observation}\""


def to_markdown(r: RunResult) -> str:
    out = ["# ResearchGap report", ""]
    if r.is_demo:
        out += ["> Offline demo output for the bundled synthetic papers. Not a live model run.", ""]
    out += [f"Model: `{r.model}`", "", "## Papers"]
    for pid, a in r.analyses.items():
        out += [f"### Paper {pid}: {a.title}", f"- File: {r.papers.get(pid, '')}",
                f"- Research question: {a.research_question}", f"- Method: {a.method}",
                f"- Participants: {a.participants if a.participants is not None else 'n/a'}",
                "- Findings:"] + [f"  - {f.statement}" for f in a.key_findings] + \
               ["- Limitations:"] + [f"  - {f.statement}" for f in a.limitations] + [""]
    out += ["## Comparison matrix", "", "| Dimension | " + " | ".join(r.analyses) + " |",
            "|---|" + "---|" * len(r.analyses)]
    for row in r.comparison.matrix:
        out.append(f"| {row.dimension} | " + " | ".join(row.values.get(p, "") for p in r.analyses) + " |")
    if r.comparison.metric_conflicts:
        out += ["", "### Metric definition conflicts"]
        out += [f"- **{m.metric}** ({', '.join(m.papers)}): {m.explanation}" for m in r.comparison.metric_conflicts]
    out += ["", "## Research gaps", "",
            "_Gaps are relative to the uploaded papers only, not proof that no prior research exists._", ""]
    for g in r.gaps:
        v = r.verifications.get(g.id)
        out += [f"### {g.id}. {g.title} [{v.verdict if v else 'unverified'}]", f"**{g.question}**", "",
                g.rationale, "", "Evidence:"] + [f"- {cite(e)}" for e in g.evidence] + \
               ["", f"Suggested experiment: {g.suggested_experiment}"]
        if v:
            out += ["", f"Verifier ({v.confidence:.0%}): {v.reasons}"]
        out.append("")
    return "\n".join(out)
