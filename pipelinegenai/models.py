from pydantic import BaseModel
from typing import List, Optional, Tuple

class Claim(BaseModel):
    claim_id: Optional[str] = None
    text: str
    entities: List[str]
    anchor_span: Optional[Tuple[int,int]] = None
    type: str  # FACT|EVENT|CAUSE_EFFECT|TRAIT
    source_backstory_id: Optional[str] = None
    model_version: Optional[str] = None
    prompt_template_hash: Optional[str] = None

class ClaimDecision(BaseModel):
    claim_id: str
    label: str  # SUPPORT|CONTRADICT|INSUFFICIENT
    evidence_ids: List[str]
    rationale: str
    model_version: Optional[str] = None
    prompt_template_hash: Optional[str] = None
    validated: bool = False
    validation_failures: List[str] = []

class CauseEffectCheck(BaseModel):
    claim_id: str
    contradiction: bool
    evidence_ids: List[str]
    rationale: str
