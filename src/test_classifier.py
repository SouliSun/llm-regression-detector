from schema import ClassificationResult
from pydantic import ValidationError

test_cases = [
    {"billing", "Charged twice"},
    {"payments", "Charged twice"},
    {"billing", ""}
]

for cat, summary in test_cases:
    try:
        res = ClassificationResult(category=cat,summary= summary)
        print(f"✅ PASSED: category='{cat}', summary='{summary}' -> {res}")
    except ValidationError as e:
        print(f"❌ REJECTED: category='{cat}', summary='{summary}'")
        for err in e.errors():
            print(f"   ↳ {err['loc'][0]}: {err['msg']}")