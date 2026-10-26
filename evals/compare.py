from pathlib import Path
import json
import argparse
import sys

def load_run(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def verify_dataset_match(baseline: dict, candidate: dict):
   base_hash = baseline.get("metadata", {}).get("dataset_sha256")
   cand_hash = candidate.get("metadata", {}).get("dataset_sha256")

   if not base_hash or base_hash != cand_hash:
        raise ValueError(
            f"Dataset mismatch!\n"
            f"  Baseline  dataset hash: {base_hash}\n"
            f"  Candidate dataset hash: {cand_hash}\n"
            f"Cannot compare runs evaluated on different datasets."
        )
def is_correct(result: dict) -> bool:
    if not result:
        return False
    return (
        result.get("valid_output") is True
        and result.get("predicted_category") == result.get("expected_category")
    )

def find_case_flips(baseline: dict, candidate: dict) -> dict:
    regressions = []
    improvements = []

    cand_map = {r["id"]: r for r in candidate.get("results", [])}

    for base in baseline.get("results", []):
        cand = cand_map.get(base["id"])

        base_ok = is_correct(base)
        cand_ok = is_correct(cand)

        if base_ok and not cand_ok:
            regressions.append({
                "id": base["id"],
                "expected": base.get("expected_category"),
                "baseline": base.get("predicted_category"),
                "candidate": cand.get("predicted_category") if cand else "Missing",
                "tags": base.get("tags", [])
            })

        elif not base_ok and cand_ok:
            improvements.append({
                "id": cand["id"],
                "expected": cand.get("expected_category"),
                "baseline": base.get("predicted_category"),
                "candidate": cand.get("predicted_category") if cand else "Missing",
                "tags": cand.get("tags", [])
            })

    return {
        "regressions": regressions,
        "improvements": improvements
    }

def compute_metric_deltas(baseline: dict, candidate: dict) -> dict:
    base_acc = baseline["metrics"]["strict_accuracy"]["accuracy"]
    cand_acc = candidate["metrics"]["strict_accuracy"]["accuracy"]
    delta_strict = cand_acc - base_acc

    category_deltas = {}
    base_cats = baseline["metrics"].get("by_category", {})
    cand_cats = candidate["metrics"].get("by_category", {})

    tags_deltas = {}
    base_tags = baseline["metrics"].get("by_tag", {})
    cand_tags = candidate["metrics"].get("by_tag", {})

    for cat in set(base_cats.keys()) | set(cand_cats.keys()):
        b_acc = base_cats.get(cat, {}).get("accuracy", 0.0)
        c_acc = cand_cats.get(cat, {}).get("accuracy", 0.0)
        category_deltas[cat] = c_acc - b_acc

    for tag in set(base_tags.keys()) | set(cand_tags.keys()):
        b_acc = base_tags.get(tag, {}).get("accuracy", 0.0)
        c_acc = cand_tags.get(tag, {}).get("accuracy", 0.0)
        tags_deltas[tag] = c_acc - b_acc

    return {
        "strict_accuracy_delta": delta_strict,
        "category_deltas": category_deltas,
        "tag_deltas": tags_deltas,
    }

def evaluate_regression(flips: dict, deltas: dict, max_regressions: int = 0, min_strict_delta: float = 0.0) -> dict:
    num_regressions = len(flips["regressions"])
    strict_delta = deltas["strict_accuracy_delta"]

    reasons = []
    if num_regressions > max_regressions:
        reasons.append(f"Found {num_regressions} regression (max allowed: {max_regressions})")
    if strict_delta < min_strict_delta:
        reasons.append(f"Strict accuracy dropped by {abs(strict_delta) * 100:.1f}%")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }

def print_comparison_report(baseline: dict, candidate: dict, flips: dict, deltas: dict, gate: dict):
    print("\n" + "=" * 65)
    print("                 REGRESSION COMPARISON REPORT")
    print("=" * 65)
    base_meta = baseline.get("metadata", {})
    cand_meta = candidate.get("metadata", {})
    print(f"Baseline : {base_meta.get('prompt_version')} ({base_meta.get('model')})")
    print(f"Candidate: {cand_meta.get('prompt_version')} ({cand_meta.get('model')})")
   
    b_acc = baseline["metrics"]["strict_accuracy"]["accuracy"] * 100
    c_acc = candidate["metrics"]["strict_accuracy"]["accuracy"] * 100
    d_acc = deltas["strict_accuracy_delta"] * 100
    sign = "+" if d_acc >= 0 else ""
    print(f"\nStrict Accuracy : {b_acc:5.1f}% -> {c_acc:5.1f}% ({sign}{d_acc:.1f}%)")
  
    print("\n--- Category Deltas ---")
    base_cats = baseline["metrics"].get("by_category", {})
    cand_cats = candidate["metrics"].get("by_category", {})
    for cat, delta in sorted(deltas["category_deltas"].items()):
        b_c = base_cats.get(cat, {}).get("accuracy", 0.0) * 100
        c_c = cand_cats.get(cat, {}).get("accuracy", 0.0) * 100
        s = "+" if delta >= 0 else ""
        print(f"  {cat:<12} : {b_c:5.1f}% -> {c_c:5.1f}% ({s}{delta * 100:.1f}%)")
    
    regs = flips.get("regressions", [])
    print(f"\n--- Regressions ({len(regs)} cases broken) ---")
    if not regs:
        print("  None! No previously working cases were broken.")
    else:
        for r in regs:
            print(f"  [REGRESSION] {r['id']} | Expected: {r['expected']} | Base: {r['baseline']} -> Cand: {r['candidate']}")
    
    imps = flips.get("improvements", [])
    print(f"\n--- Improvements ({len(imps)} cases fixed) ---")
    if not imps:
        print("  None.")
    else:
        for im in imps:
            print(f"  [IMPROVEMENT] {im['id']} | Expected: {im['expected']} | Base: {im['baseline']} -> Cand: {im['candidate']}")
    
    print("\n" + "=" * 65)
    if gate["passed"]:
        print(">>> VERDICT: [PASS] Candidate is safe to release! <<<")
    else:
        print(">>> VERDICT: [FAIL] Regressions detected! Candidate rejected. <<<")
        for reason in gate["reasons"]:
            print(f"    - {reason}")
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser("Compare baseline and candidate evaluation runs to detect regressions")
    parser.add_argument("--baseline", default="results/v1_baseline.json")
    parser.add_argument("--candidate", default="results/v1_run2.json")
    parser.add_argument("--max-regressions", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)

    args = parser.parse_args()

    base = load_run(args.baseline)
    cand= load_run(args.candidate)
    verify_dataset_match(base, cand)

    flips = find_case_flips(base, cand)
    deltas = compute_metric_deltas(base, cand)
    gate = evaluate_regression(flips, deltas, args.max_regressions, args.min_delta)

    print_comparison_report(base, cand, flips, deltas, gate)
    sys.exit(0 if gate["passed"] else 1)

if __name__ == "__main__":
    main()