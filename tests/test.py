import pytest
from src.classifier import Classifier, InvalidModelOutput
from src.schema import Category
from src.prompt_loader import load_prompt

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

import yaml

def test_loader_rejects_broken_yaml(tmp_path):
    broken_file = tmp_path / "broken.yaml"
    broken_file.write_text("invalid: yaml: [unclosed brackets", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_prompt(broken_file)