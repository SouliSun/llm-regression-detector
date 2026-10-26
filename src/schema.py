from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    general = "general"

class ClassificationResult(BaseModel):
    category : Category = Field(
        description= "The predicted category for the email"
    )
    summary : str = Field(
        min_length=1,
        description= "Concise summary of the user issue"
    )

class FewShotExample(BaseModel):
    input_text : str =  Field(..., min_length=1, description="Sample user query/ticket")
    expected_output: ClassificationResult
    
class PromptConfig(BaseModel):
    version : str = Field(..., description="Prompt version or git commit hash, e.g. 'v1.2.0'")
    model : str = Field(default="openai/gpt-oss-20b", description="Model Identifier")
    temperature: float = Field(default=0.0, ge = 0.0, le = 2.0)
    system_prompt: str = Field(..., min_length=1)
    few_shot_examples : List[FewShotExample] = Field(default_factory=list)