from pydantic import BaseModel, ConfigDict
from typing import Dict, List

class JobV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    job_id: str
    statement: str
    forces: Dict[str, List[str]]  # push/pull/anxiety/inertia
