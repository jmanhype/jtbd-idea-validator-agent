from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class AssumptionV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    assumption_id: str
    text: str
    level: int = Field(ge=1, le=3, description="1=observed,2=educated,3=strategic")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = []
    validation_exp_id: Optional[str] = None
