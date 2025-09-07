from pydantic import BaseModel, ConfigDict
from typing import List

class Section(BaseModel):
    title: str
    md: str

class Asset(BaseModel):
    id: str
    type: str
    path: str

class GammaDocV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    doc_id: str
    sections: List[Section]
    assets: List[Asset]
