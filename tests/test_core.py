import pytest

from researchgap.demo import demo_result
from researchgap.llm import GemmaClient, extract_json
from researchgap.pdf_processing import parse_pdf, quote_in_page
from researchgap.sample_papers import build_sample_papers
from researchgap.schemas import GapList


@pytest.fixture(scope="module")
def papers():
    return [parse_pdf(pid, name, data) for pid, (name, data) in zip("ABC", build_sample_papers())]


def test_parse_detects_figures(papers):
    a = papers[0]
    assert a.page_count == 2
    labels = [c.split(":")[0] for c in a.pages[1].captions]
    assert labels == ["Figure 2", "Table 1"]
    assert [p.number for p in a.visual_pages(6)] == [2]
    assert "[Page 2]" in a.tagged_text()


def test_quote_lookup(papers):
    assert quote_in_page(papers[2].pdf_bytes, 2, "Accuracy remains high across patch sizes")
    assert not quote_in_page(papers[2].pdf_bytes, 2, "reviewers loved the tool very much indeed")
    assert not quote_in_page(papers[2].pdf_bytes, 9, "Accuracy remains high")


def test_wrong_page_citation_is_corrected(papers):
    from researchgap.pipeline import check_evidence
    from researchgap.schemas import EvidenceRef, ResearchGap

    # the definition is on page 1, but the agent cites page 2
    ref = EvidenceRef(paper_id="A", page=2, quote_or_observation="percentage of seeded defects that a reviewer correctly identified")
    gap = ResearchGap(id="G1", title="t", question="q", rationale="r", suggested_experiment="e", evidence=[ref])
    [check] = check_evidence(gap, {p.paper_id: p for p in papers})
    assert check.status == "found" and ref.page == 1


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('prefix {"a": {"b": 2}} suffix') == '{"a": {"b": 2}}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_demo_verification(papers):
    r = demo_result(papers)
    v = r.verifications
    assert v["G1"].verdict == "supported"
    assert all(c.status == "found" for c in v["G1"].evidence_checks)
    assert v["G5"].verdict == "rejected"  # cites a page that does not exist
    assert r.text_vs_visual[0].contradictions_revealed  # Figure 2 contradicts paper A's text


def test_pipeline_orchestration_with_fake_model(monkeypatch, papers):
    """Runs the real pipeline with the model call replaced by canned demo objects."""
    from researchgap import pipeline
    from researchgap.config import Settings
    from researchgap.schemas import Comparison, PaperAnalysis, VerifierOutput

    demo = demo_result(papers)
    seen_images = []

    def fake_structured(self, prompt, model_cls, images=None, max_repairs=2):
        seen_images.extend(images or [])
        if model_cls is PaperAnalysis:
            pid = prompt.split("Analyze paper ")[1][0]
            mm = "mode to \"multimodal\"" in prompt
            return (demo.analyses if mm else demo.text_only_analyses)[pid].model_copy(deep=True)
        if model_cls is Comparison:
            return demo.comparison
        if model_cls is GapList:
            return GapList(gaps=[g.model_copy(deep=True) for g in demo.gaps])
        if model_cls is VerifierOutput:
            return VerifierOutput(verdict="supported", confidence=0.9, reasons="ok")
        raise AssertionError(model_cls)

    monkeypatch.setattr(GemmaClient, "__init__", lambda self, s: None)
    monkeypatch.setattr(GemmaClient, "structured", fake_structured)
    r = pipeline.run_pipeline(papers, Settings(api_key="test"))
    assert set(r.analyses) == {"A", "B", "C"} and set(r.text_only_analyses) == {"A", "B", "C"}
    assert r.verifications["G5"].verdict == "rejected"  # deterministic check vetoes the model
    assert r.verifications["G1"].verdict == "supported"
    assert any("Paper A, page 2" in im.label for im in seen_images)


def test_structured_repairs_bad_json(monkeypatch):
    client = GemmaClient.__new__(GemmaClient)
    replies = iter(["not json", '{"gaps": []}'])
    monkeypatch.setattr(client, "_generate", lambda prompt, images: next(replies), raising=False)
    assert client.structured("x", GapList).gaps == []
