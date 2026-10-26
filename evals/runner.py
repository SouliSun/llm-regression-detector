from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

import json
import hashlib
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional
from pydantic import BaseModel

from src.schema import Category, PromptConfig
from src.classifier import Classifier, InvalidModelOutput
from evals.dataset import GoldenCase
from evals.metrics import compute_report, EvalReport

class CaseResult(BaseModel):
    id: str
    expected_category: Category
    tags: list[str] = []
    predicted_category: Optional[Category] = None
    valid_output: bool
    raw_output: Optional[str] = None
    latency_ms: float
    error: Optional[str] = None

def run_case(classifier: Classifier, case: GoldenCase) -> CaseResult:
    start = perf_counter()
    try:
        c = classifier.classify(case.email)
        latency = (perf_counter() - start) * 1000
        return CaseResult(
            id=case.id,
            expected_category=case.expected_category,
            tags=case.tags,
            predicted_category=c.category,
            valid_output=True,
            latency_ms=latency,
        )
    except InvalidModelOutput as e:
        latency = (perf_counter() - start) * 1000
        raw = getattr(e, "raw_output", getattr(e, "raw_json", ""))
        return CaseResult(
            id=case.id,
            expected_category=case.expected_category,
            tags=case.tags,
            valid_output=False,
            raw_output=raw,
            error=str(e.reason) if hasattr(e, "reason") else str(e),
            latency_ms=latency,
        )

def run_all(classifier: Classifier, cases: list[GoldenCase]) -> list[CaseResult]:
    results = []
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        res = run_case(classifier, case)
        results.append(res)
        status = res.predicted_category.value if res.valid_output and res.predicted_category else "INVALID"
        print(f"[{i}/{total}] {res.id} -> {status} ({res.latency_ms:.0f}ms)")
    return results

def save_run(
    output_path: str | Path,
    prompt_config: PromptConfig,
    dataset_path: str | Path,
    cases: list[GoldenCase],
    results: list[CaseResult],
    report: Optional[EvalReport] = None,
    prompt_path: Optional[str | Path] = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds_path = Path(dataset_path)
    dataset_hash = hashlib.sha256(ds_path.read_bytes()).hexdigest() if ds_path.exists() else None

    prompt_hash = None
    if prompt_path and Path(prompt_path).exists():
        prompt_hash = hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()
    else:
        # Fallback to hashing prompt config content
        prompt_content = f"{prompt_config.system_prompt}\n{[ex.model_dump_json() for ex in prompt_config.few_shot_examples]}"
        prompt_hash = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()

    if report is None:
        report = compute_report(cases, results)

    payload = {
        "metadata": {
            "prompt_version": prompt_config.version,
            "prompt_sha256": prompt_hash,
            "model": prompt_config.model,
            "temperature": prompt_config.temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(ds_path),
            "dataset_sha256": dataset_hash,
        },
        "metrics": report.model_dump(),
        "results": [r.model_dump() for r in results],
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out
