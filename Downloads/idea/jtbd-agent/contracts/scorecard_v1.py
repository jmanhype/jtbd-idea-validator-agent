from pydantic import BaseModel, Field, ConfigDict
from typing import List

class Criterion(BaseModel):
    name: str
    score: float = Field(ge=0, le=10)
    rationale: str

class ScorecardV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    target_id: str
    scheme: str = "v1"
    criteria: List[Criterion]
    total: float = Field(ge=0, le=10)
