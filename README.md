# Model Regression Detection System

Automated regression-testing harness for our customer support email classifier (`billing`, `technical`, `account`, `general`). It catches silent regressions when prompts or models change by comparing case-level results against a locked golden dataset, so improvements in one area don't secretly break working cases elsewhere.

---

## Quick Setup

### 1. Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_key_here
# or OPENAI_API_KEY=your_key_here
```

### 3. Verify it works
Run the offline test suite (runs in ~2s, no API keys needed):
```bash
pytest
```

---

## Daily Commands

```bash
# 1. Run an eval for a prompt version
python -m evals.run --prompt v1 --out results/v1_baseline.json

# 2. Compare a candidate prompt against baseline
python -m evals.compare --baseline results/v1_baseline.json --candidate results/v2_candidate.json
```
- **Exit 0**: All clean, safe to merge.
- **Exit 1**: Regressions detected, PR gets blocked.

---

## Adding Cases to the Golden Dataset

The golden dataset lives in `data/dataset.jsonl`. Each line is a single JSON record:

```json
{
  "id": "TC-121",
  "email": "I was charged twice for my subscription this morning.",
  "expected_category": "billing",
  "acceptable_categories": null,
  "tags": ["clear", "short"],
  "notes": "Duplicate billing charge refund request",
  "source": "customer_support"
}
```

### Guidelines for new cases:
- **The Tie-Break Rule**: If an email mentions multiple issues, classify by **what the customer needs fixed first**.
  - *Example*: "I can't log in to update my credit card" $\rightarrow$ `account` (they can't pay until they're logged in).
- **Ambiguous Cases**: If an email is truly a 50/50 split between two categories, list both under `"acceptable_categories": ["billing", "technical"]`. These are tracked separately so normal LLM variance doesn't trigger false-positive regression alerts.
- **No Prompt Leakage**: Never paste examples from `prompts/*.yaml` into the dataset.

---

## Adjusting CI/CD Thresholds

By default, CI blocks merges on **any** regression (`0` allowed) or accuracy drop. When doing a major model migration or intentional behavior shift, you can adjust these thresholds via CLI flags:

```bash
python -m evals.compare \
  --baseline results/v1_baseline.json \
  --candidate results/v2_candidate.json \
  --max-regressions 2 \
  --min-delta -0.01
```

- `--max-regressions`: How many previously-correct cases are allowed to flip (default: `0`).
- `--min-delta`: Minimum change in strict accuracy allowed (default: `0.0`). Set to `-0.01` to tolerate 1% noise.

These flags can also be configured directly in `.github/workflows/v1.yml`.

---

## Architecture Decisions & Why

- **Case-Level Flips > Top-Line Accuracy**: A prompt can go from 95% to 95% accuracy while breaking 10 high-value customer workflows and accidentally fixing 10 edge cases. Tracking individual case transitions (`PASS -> FAIL`) is the only way to catch silent regressions.
- **Pure Metric Functions**: All metrics logic in `evals/metrics.py` takes plain lists/dicts with zero network or file dependencies. This makes metrics instantly testable and allows re-analyzing past run artifacts without paying for LLM tokens.
- **Denominator Includes Missing Cases**: Metrics loop over expected cases rather than returned results. If an API call drops or crashes, that case is scored as a failure instead of disappearing and inflating the accuracy score.
- **Dataset & Prompt Fingerprinting**: Run files save SHA-256 hashes of both the dataset and prompt config. If someone silently edits the dataset between runs, the comparison step immediately catches the integrity mismatch.
- **Provider Refusal Handling**: When test emails look like scams or phishing, LLM safety guardrails often return raw refusals that break JSON schemas. The client catches these (`json_validate_failed`), records them as invalid outputs, and keeps the test suite running instead of crashing mid-batch.

