from pydantic import BaseModel, Field, ConfigDict
from typing import List

class Experiment(BaseModel):
    exp_id: str
    hypothesis: str
    design: str
    metric: str
    success: str
    evi: float

class Stage(BaseModel):
    stage: str
    budget: float
    experiments: List[Experiment]

class ValidationPlanV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    plan_id: str
    stages: List[Stage]
