from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])
from src.schema import Category

class GoldenCase(BaseModel):
    id: str
    email: str
    expected_category: Category
    acceptable_categories: Optional[List[Category]] = None
    tags: List[str] = []
    notes: str
    source: str

def load_dataset(path: str | Path) -> list[GoldenCase]:
    case = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found!")
    
    with open(path, "r", encoding="utf-8") as f :
        content = f.readlines()    
        for line_no, raw_line in enumerate(content, start=1):
            line = raw_line.strip()
            if line :
                try:
                    case.append(GoldenCase.model_validate_json(line))
                except Exception as e:
                    raise ValueError(f"Error on line {line_no} of {path}: {e}") from e
    return case

if __name__ == "__main__":
    cases = load_dataset("data/dataset.jsonl")
    print(f"Loaded {len(cases)} cases successfully. First ID: {cases[0].id}")