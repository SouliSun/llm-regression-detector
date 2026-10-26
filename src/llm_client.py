from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from dotenv import load_dotenv
import os
import time
from typing import Dict, List, Optional, Protocol
from pydantic import BaseModel, Field

from schema import ClassificationResult, PromptConfig

class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], model: str, temperature: float) -> str:
        pass

class OpenAIClient():
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            print("API key loaded successfully!!")
        else:
            raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0
            )
        
    def complete(self, system: str, messages: list[dict], model: str, temperature: float) -> str:
        delays = [1, 2, 4]
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, *messages],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("No Response generated!")
                
                return content
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == 3:
                    raise  
                time.sleep(delays[attempt])


if __name__ == "__main__":
    client = OpenAIClient()
    response = client.complete(
        system="You classify customer support emails. Output JSON only: {\"category\": \"billing\" | \"technical\" | \"account\" | \"general\", \"summary\": \"one sentence\"}",
        messages=[{"role": "user", "content": "I was charged $50 twice for my subscription this morning."}],
        model="openai/gpt-oss-120b",
        temperature=0.0
    )

    print("Real API Output:")
    print(response)

