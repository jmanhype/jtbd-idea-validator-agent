from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any

class IdeaV1(BaseModel):
    model_config = ConfigDict(extra='allow', frozen=True, strict=False)
    
    # Required core fields
    idea_id: str = Field(..., description="URN")
    title: str
    hunches: List[str]
    
    # Optional context (flexible format)
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    
    # Optional rich business context
    problem_statement: Optional[str] = None
    solution_overview: Optional[str] = None
    target_customer: Optional[Dict[str, Any]] = None
    value_propositions: Optional[List[str]] = None
    competitive_landscape: Optional[List[str]] = None
    revenue_streams: Optional[List[str]] = None
    key_metrics: Optional[List[str]] = None
    risks_and_challenges: Optional[List[str]] = None
    go_to_market_strategy: Optional[Dict[str, Any]] = None
    
    # Legacy support for nested idea structure
    idea: Optional[Dict[str, Any]] = None
