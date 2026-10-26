import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

from src.classifier import Classifier, InvalidModelOutput
from src.llm_client import OpenAIClient
from src.prompt_loader import load_prompt

config = load_prompt("v1")
config.model = "openai/gpt-oss-20b"

classifier = Classifier(client = OpenAIClient(), config = config)

emails = [
    ("billing",   "Hi, I was charged twice this month. Please refund one."),
    ("technical", "The app crashes every time I open the export page."),
    ("account",   "I forgot my password and the reset email never arrives."),
    ("general",   "Do you offer discounts for nonprofits?"),
    ("AMBIGUOUS", "I got charged after my trial, but the cancel button doesn't work."),
]

for cat, text in emails:
    try:
        r = classifier.classify(text)
        print(f"[{cat}] -> {r.category.value} | {r.summary}")
    except InvalidModelOutput as e:
        print(f"[{cat}] -> INVALID OUTPUT: {e.raw_json!r}")