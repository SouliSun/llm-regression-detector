import argparse

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

from src.prompt_loader import load_prompt
from evals.dataset import load_dataset
from src.llm_client import OpenAIClient
from src.classifier import Classifier
from evals.runner import run_all, save_run
from evals.metrics import compute_report


def parse_args():
    parse = argparse.ArgumentParser(description="Run LLM evaluation on a dataset")

    parse.add_argument("--dataset",default="data/dataset.jsonl")
    parse.add_argument("--prompt", default="v1")
    parse.add_argument("--out", default="results/v1_baseline.json")

    return parse.parse_args()

def print_report(cases, results, report):
    print("\n" + "=" * 60)
    print("                    EVALUATION REPORT")
    print("=" * 60)

    sa = report.strict_accuracy
    aa = report.ambiguous_accuracy
    ir = report.invalid_rate

    print(f"Strict Accuracy     : {sa.accuracy * 100:5.1f}%  ({sa.correct}/{sa.total})")
    print(f"Within Acceptable   : {aa.accuracy * 100:5.1f}%  ({aa.correct}/{aa.total}) [ambiguous set]")
    print(f"Invalid Output Rate : {ir.rate * 100:5.1f}%  ({ir.invalid}/{ir.total})")

    print("\n--- Accuracy by Category (Strict) ---")
    for cat, stats in sorted(report.by_category.items()):
        print(f"  {cat:<12} : {stats.accuracy * 100:5.1f}%  (n={stats.total}, correct={stats.correct})")

    case_map = {c.id: c for c in cases}
    failures = []
    for r in results:
        c = case_map.get(r.id)
        if not c or c.acceptable_categories:
            continue
        if not r.valid_output or r.predicted_category != c.expected_category:
            failures.append((c, r))

    print(f"\n--- Failure Details ({len(failures)} cases) ---")
    for c, r in failures:
        pred = r.predicted_category.value if r.valid_output and r.predicted_category else "INVALID"
        print(f"  [{c.id}] Expected: {c.expected_category.value:<10} | Predicted: {pred:<10} | {c.email[:60]}...")

    print("=" * 60)

def main():
    args = parse_args()
    config = load_prompt(args.prompt)
    cases = load_dataset(args.dataset)

    client = OpenAIClient()
    classifier = Classifier(client=client, config=config)

    results = run_all(classifier, cases)

    report  = compute_report(cases, results)

    save_run(args.out, config, args.dataset, cases, results, report=report)

    print_report(cases, results, report)
    
if __name__ == "__main__":
    main()