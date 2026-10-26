import pytest
import json
import yaml
from pathlib import Path

from src.classifier import Classifier, InvalidModelOutput
from src.schema import Category
from src.prompt_loader import load_prompt
from evals.dataset import load_dataset, GoldenCase
from evals.runner import run_case, CaseResult, save_run
from evals.metrics import (
    compute_strict_accuracy,
    compute_ambiguous_accuracy,
    compute_invalid_rate,
    compute_report,
    AccuracyStats,
    EvalReport,
)
from evals.compare import (
    verify_dataset_match,
    find_case_flips,
    evaluate_regression,
)

class FakeLLMClient:
    def __init__(self, response: str = ""):
        self.response = response
        self.calls = []  

    def complete(self, system: str, messages: list[dict], model: str, temperature: float) -> str:
        self.calls.append({
            "system": system,
            "messages": messages,
            "model": model,
            "temperature": temperature
        })
        return self.response

@pytest.fixture
def config():
    return load_prompt("v1")

@pytest.mark.parametrize(
    "bad_reply",
    [
        "Sure! Here is the JSON: {}",                             
        '{"category": "payments", "summary": "Charged twice."}',  
        '{"category": "billing"}',                                
    ]
)
def test_bad_model_output_raises_invalid_model_output(config, bad_reply):
    fake = FakeLLMClient(response=bad_reply)
    classifier = Classifier(client=fake, config=config)

    with pytest.raises(InvalidModelOutput) as exc_info:
        classifier.classify("My bill is wrong.")

    assert exc_info.value.raw_json == bad_reply
    assert exc_info.value.raw_output == bad_reply

def test_valid_reply_parses_successfully(config):
    fake = FakeLLMClient(response='{"category": "billing", "summary": "Charged twice."}')
    classifier = Classifier(client=fake, config=config)
    result = classifier.classify("I was charged twice on my card.")
    assert result.category == Category.billing
    assert result.summary == "Charged twice."

def test_empty_email_raises_value_error_and_does_not_call_llm(config):
    fake = FakeLLMClient()
    classifier = Classifier(client=fake, config=config)
    with pytest.raises(ValueError):
        classifier.classify("   ")
    
    assert len(fake.calls) == 0

def test_message_structure_has_few_shots_and_target_email_last(config):
    fake = FakeLLMClient(response='{"category": "general", "summary": "Hello"}')
    classifier = Classifier(client=fake, config=config)
    classifier.classify("My target email")
    assert len(fake.calls) == 1
    messages = fake.calls[0]["messages"]
    expected_len = (2 * len(config.few_shot_examples)) + 1
    assert len(messages) == expected_len
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "My target email"

def test_load_resolve_v1():
    config = load_prompt("v1")
    assert config.version == "v1"
    assert len(config.few_shot_examples) > 0

def test_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_prompt("v99")

def test_loader_rejects_broken_yaml(tmp_path):
    broken_file = tmp_path / "broken.yaml"
    broken_file.write_text("invalid: yaml: [unclosed brackets", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_prompt(broken_file)

def test_bad_path(tmp_path: str | Path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"id": "TC-99", "email": "test", "expected_category": "payments", "notes": "x", "source": "s"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_dataset(bad_file)
    assert "line 1" in str(exc_info.value).lower()

def test_run_case(config):
    fake = FakeLLMClient(response='{"category": "billing", "summary": "Refund"}')
    classifier = Classifier(client=fake, config=config)
    case = GoldenCase(id="TC-01", email="Need refund", expected_category=Category.billing, tags=["clear"], notes="n", source="s")
    res = run_case(classifier=classifier, case=case)
    assert res.valid_output is True
    assert res.predicted_category == Category.billing
    assert res.expected_category == Category.billing
    assert res.tags == ["clear"]
    assert res.latency_ms > 0

def test_run_case_invalid_output(config):
    fake = FakeLLMClient(response='{"category": "Policy", "summary": "Refund"}')
    classifier = Classifier(client=fake, config=config)
    case = GoldenCase(id="TC-01", email="Need refund", expected_category=Category.billing, tags=["clear"], notes="n", source="s")
    res = run_case(classifier=classifier, case=case)
    assert res.valid_output is False
    assert res.predicted_category is None
    assert res.raw_output == '{"category": "Policy", "summary": "Refund"}'
    assert res.expected_category == Category.billing
    assert res.latency_ms > 0
    assert res.error is not None

def test_accuracy_ignores_ambiguous_and_penalizes_missing():
    cases = [
        GoldenCase(id="TC-01", email="Refund please", expected_category=Category.billing, notes="n", source="s"),
        GoldenCase(id="TC-02", email="App crashed", expected_category=Category.technical, notes="n", source="s"),
        GoldenCase(id="TC-03", email="Charged but app broke", expected_category=Category.billing, acceptable_categories=[Category.billing, Category.technical], notes="n", source="s"),
    ]

    results = [
        CaseResult(id="TC-01", expected_category=Category.billing, predicted_category=Category.billing, valid_output=True, latency_ms=100.0),
        CaseResult(id="TC-02", expected_category=Category.technical, predicted_category=Category.general, valid_output=True, latency_ms=100.0),
        CaseResult(id="TC-03", expected_category=Category.billing, predicted_category=Category.general, valid_output=True, latency_ms=100.0),
    ]

    metrics = compute_strict_accuracy(cases, results)
    assert isinstance(metrics, AccuracyStats)
    assert metrics.total == 2
    assert metrics.correct == 1
    assert metrics.accuracy == 0.5

    # Test missing result is counted in denominator as failure
    metrics_partial = compute_strict_accuracy(cases, results[:1])
    assert metrics_partial.total == 2
    assert metrics_partial.correct == 1
    assert metrics_partial.accuracy == 0.5

def test_save_run(tmp_path, config):
    cases = [
        GoldenCase(id="TC-01", email="Refund please", expected_category=Category.billing, tags=["clear"], notes="n", source="s"),
    ]
    results = [
        CaseResult(id="TC-01", expected_category=Category.billing, tags=["clear"], predicted_category=Category.billing, valid_output=True, latency_ms=100.0),
    ]
    ds_file = tmp_path / "dataset.jsonl"
    ds_file.write_text('{"id": "TC-01", "email": "Refund please", "expected_category": "billing", "notes": "n", "source": "s"}\n', encoding="utf-8")

    out_file = tmp_path / "runs" / "test_run.json"
    saved = save_run(
        output_path=out_file,
        prompt_config=config,
        dataset_path=ds_file,
        cases=cases,
        results=results,
    )

    assert saved.exists()
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert "metadata" in data
    assert "metrics" in data
    assert "results" in data
    assert data["metadata"]["prompt_version"] == config.version
    assert data["metadata"]["prompt_sha256"] is not None
    assert data["metadata"]["dataset_sha256"] is not None
    assert len(data["results"]) == 1
    assert data["results"][0]["expected_category"] == "billing"
    assert data["metrics"]["strict_accuracy"]["accuracy"] == 1.0

def test_verify_dataset_match():
    base = {"metadata": {"dataset_sha256": "abc123hash"}}
    cand_same = {"metadata": {"dataset_sha256": "abc123hash"}}
    cand_diff = {"metadata": {"dataset_sha256": "different_hash"}}

    verify_dataset_match(base, cand_same)

    with pytest.raises(ValueError, match="Dataset mismatch"):
        verify_dataset_match(base, cand_diff)

def test_find_case_flips_detects_regression_and_improvement():
    base = {
        "results": [
            {"id": "TC-01", "expected_category": "billing", "predicted_category": "billing", "valid_output": True, "tags": []},
            {"id": "TC-02", "expected_category": "technical", "predicted_category": "general", "valid_output": True, "tags": []},
        ]
    }
    cand = {
        "results": [
            {"id": "TC-01", "expected_category": "billing", "predicted_category": "technical", "valid_output": True, "tags": []},
            {"id": "TC-02", "expected_category": "technical", "predicted_category": "technical", "valid_output": True, "tags": []},
        ]
    }

    flips = find_case_flips(base, cand)
    assert len(flips["regressions"]) == 1
    assert flips["regressions"][0]["id"] == "TC-01"
    assert len(flips["improvements"]) == 1
    assert flips["improvements"][0]["id"] == "TC-02"

def test_evaluate_regression_gate():
    flips_pass = {"regressions": [], "improvements": [{"id": "TC-01"}]}
    deltas_pass = {"strict_accuracy_delta": 0.05}
    gate_pass = evaluate_regression(flips_pass, deltas_pass, max_regressions=0, min_strict_delta=0.0)
    assert gate_pass["passed"] is True
    assert len(gate_pass["reasons"]) == 0

    flips_fail = {"regressions": [{"id": "TC-01"}], "improvements": []}
    deltas_fail = {"strict_accuracy_delta": -0.05}
    gate_fail = evaluate_regression(flips_fail, deltas_fail, max_regressions=0, min_strict_delta=0.0)
    assert gate_fail["passed"] is False
    assert len(gate_fail["reasons"]) == 2

