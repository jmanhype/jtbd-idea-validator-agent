from pydantic import BaseModel, ConfigDict
class InnovationLayerV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    layer_id: str
    type: str
    trigger: str
    effect: str
