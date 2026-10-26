from __future__ import annotations
from typing import TYPE_CHECKING
from pydantic import BaseModel
from src.schema import Category

if TYPE_CHECKING:
    from evals.dataset import GoldenCase
    from evals.runner import CaseResult

class AccuracyStats(BaseModel):
    accuracy: float
    correct: int
    total: int

class InvalidStats(BaseModel):
    rate: float
    invalid: int
    total: int

class EvalReport(BaseModel):
    strict_accuracy: AccuracyStats
    ambiguous_accuracy: AccuracyStats
    invalid_rate: InvalidStats
    by_category: dict[str, AccuracyStats]
    by_tag: dict[str, AccuracyStats]
    confusion_matrix: dict[str, dict[str, int]]

def compute_strict_accuracy(cases: list[GoldenCase], results: list[CaseResult]) -> AccuracyStats:
    results_map = {res.id: res for res in results}
    correct = 0
    total = 0
    for case in cases:
        if not case.acceptable_categories:
            total += 1
            res = results_map.get(case.id)
            if res and res.valid_output and res.predicted_category == case.expected_category:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return AccuracyStats(accuracy=accuracy, correct=correct, total=total)

def compute_ambiguous_accuracy(cases: list[GoldenCase], results: list[CaseResult]) -> AccuracyStats:
    results_map = {res.id: res for res in results}
    correct = 0
    total = 0
    for case in cases:
        if case.acceptable_categories:
            total += 1
            res = results_map.get(case.id)
            if res and res.valid_output and res.predicted_category in case.acceptable_categories:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return AccuracyStats(accuracy=accuracy, correct=correct, total=total)

def compute_invalid_rate(results: list[CaseResult]) -> InvalidStats:
    total = len(results)
    invalid = sum(1 for res in results if not res.valid_output)
    rate = invalid / total if total > 0 else 0.0
    return InvalidStats(rate=rate, invalid=invalid, total=total)

def compute_accuracy_by_category(cases: list[GoldenCase], results: list[CaseResult]) -> dict[str, AccuracyStats]:
    results_map = {res.id: res for res in results}
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        if case.acceptable_categories:
            continue
        cat = case.expected_category.value
        if cat not in counts:
            counts[cat] = {"correct": 0, "total": 0}
        counts[cat]["total"] += 1
        res = results_map.get(case.id)
        if res and res.valid_output and res.predicted_category == case.expected_category:
            counts[cat]["correct"] += 1

    return {
        cat: AccuracyStats(
            accuracy=(data["correct"] / data["total"] if data["total"] > 0 else 0.0),
            correct=data["correct"],
            total=data["total"],
        )
        for cat, data in counts.items()
    }

def compute_accuracy_by_tag(cases: list[GoldenCase], results: list[CaseResult]) -> dict[str, AccuracyStats]:
    results_map = {res.id: res for res in results}
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        if case.acceptable_categories:
            continue
        res = results_map.get(case.id)
        is_correct = bool(res and res.valid_output and res.predicted_category == case.expected_category)
        for tag in case.tags:
            if tag not in counts:
                counts[tag] = {"correct": 0, "total": 0}
            counts[tag]["total"] += 1
            if is_correct:
                counts[tag]["correct"] += 1

    return {
        tag: AccuracyStats(
            accuracy=(data["correct"] / data["total"] if data["total"] > 0 else 0.0),
            correct=data["correct"],
            total=data["total"],
        )
        for tag, data in counts.items()
    }

def compute_confusion_matrix(cases: list[GoldenCase], results: list[CaseResult]) -> dict[str, dict[str, int]]:
    results_map = {res.id: res for res in results}
    categories = [c.value for c in Category]
    columns = categories + ["invalid"]
    matrix = {exp: {pred: 0 for pred in columns} for exp in categories}

    for case in cases:
        expected = case.expected_category.value
        res = results_map.get(case.id)
        if res and res.valid_output and res.predicted_category:
            predicted = res.predicted_category.value
        else:
            predicted = "invalid"
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1

    return matrix

def compute_report(cases: list[GoldenCase], results: list[CaseResult]) -> EvalReport:
    return EvalReport(
        strict_accuracy=compute_strict_accuracy(cases, results),
        ambiguous_accuracy=compute_ambiguous_accuracy(cases, results),
        invalid_rate=compute_invalid_rate(results),
        by_category=compute_accuracy_by_category(cases, results),
        by_tag=compute_accuracy_by_tag(cases, results),
        confusion_matrix=compute_confusion_matrix(cases, results),
    )
