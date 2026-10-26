from pathlib import Path
from typing import Any, Dict, List, Union
import yaml

from schema import PromptConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

def load_prompt(version_or_path : Union[str, Path] = "v1") -> PromptConfig:
    if isinstance(version_or_path, str) and not version_or_path.endswith(("yaml", "yml")):
        file_path = PROMPTS_DIR / f"{version_or_path}.yaml"
    else:
        file_path = Path(version_or_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    return PromptConfig.model_validate(raw_data)

if __name__ == "__main__":
    config = load_prompt("v1")
    print(f"✅ Loaded prompt version: {config.version}")
    print(f"✅ Model placeholder: {config.model}")
    print(f"✅ Temperature: {config.temperature}")
    print(f"✅ Loaded {len(config.few_shot_examples)} few-shot examples")