from llm_client import LLMClient
from schema import PromptConfig, ClassificationResult
from pydantic import ValidationError

class InvalidModelOutput(Exception):
    """The model replied, but the reply is not a valid ClassificationResult."""
 
    def __init__(self, raw_json: str, reason: str) -> None:
        super().__init__(f"Invalid model output: {reason}")
        self.raw_json = raw_json
        self.raw_output = raw_json
        self.reason = reason
 

class Classifier():
    def __init__(self, client: LLMClient, config: PromptConfig):
        self.client = client
        self.config = config

    def _build_messages(self, email: str) -> list[dict]:
        messages = []
        for ex in self.config.few_shot_examples:
            messages.append({
                "role": "user",
                "content": ex.input_text.strip()
            })
            messages.append({
                "role": "assistant",
                "content": ex.expected_output.model_dump_json()
            })

        messages.append({
            "role": "user",
            "content": email.strip()
        })
        return messages

    def classify(self, email: str) -> ClassificationResult:
        if not email or not email.strip():
            raise ValueError("Email cannot be empty")
        
        messages = self._build_messages(email)
        
        raw_json = self.client.complete(
            system = self.config.system_prompt,
            messages = messages,
            model = self.config.model,
            temperature= self.config.temperature
        )
        try:
            return ClassificationResult.model_validate_json(raw_json)
        except ValidationError as e:
            raise InvalidModelOutput(raw_json, str(e)) from e

if __name__ == "__main__":
    from prompt_loader import load_prompt
    from llm_client import OpenAIClient

    config = load_prompt("v1")
    config.model = "openai/gpt-oss-120b"

    client = OpenAIClient()
    classifier = Classifier(client=client, config=config)

    test_email = "Hi team, we were billed twice on our invoice this month. Can we get a refund?"
    result = classifier.classify(test_email)
    print("--- Classification Result ---")
    print(f"Category : {result.category.value}")
    print(f"Summary  : {result.summary}")