from typing import List, Optional, Tuple, Any, Dict

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class Claim(BaseModel):
    claim_id: Optional[str] = None
    text: str = ""
    entities: List[str] = []
    anchor_span: Optional[Tuple[int, int]] = None
    type: str = "FACT"  # FACT|EVENT|CAUSE_EFFECT|TRAIT
    source_backstory_id: Optional[str] = None
    model_version: Optional[str] = None
    prompt_template_hash: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'claim_id'):
            self.claim_id = kwargs.get('claim_id')
        if not hasattr(self, 'text'):
            self.text = kwargs.get('text', '')
        if not hasattr(self, 'entities'):
            self.entities = kwargs.get('entities', [])
        if not hasattr(self, 'anchor_span'):
            self.anchor_span = kwargs.get('anchor_span')
        if not hasattr(self, 'type'):
            self.type = kwargs.get('type', 'FACT')
        if not hasattr(self, 'source_backstory_id'):
            self.source_backstory_id = kwargs.get('source_backstory_id')
        if not hasattr(self, 'model_version'):
            self.model_version = kwargs.get('model_version')
        if not hasattr(self, 'prompt_template_hash'):
            self.prompt_template_hash = kwargs.get('prompt_template_hash')


class ClaimDecision(BaseModel):
    claim_id: str = ""
    label: str = "INSUFFICIENT"  # SUPPORT|CONTRADICT|INSUFFICIENT
    evidence_ids: List[str] = []
    rationale: str = ""
    model_version: Optional[str] = None
    prompt_template_hash: Optional[str] = None
    validated: bool = False
    validation_failures: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'claim_id'):
            self.claim_id = kwargs.get('claim_id', '')
        if not hasattr(self, 'label'):
            self.label = kwargs.get('label', 'INSUFFICIENT')
        if not hasattr(self, 'evidence_ids'):
            self.evidence_ids = kwargs.get('evidence_ids', [])
        if not hasattr(self, 'rationale'):
            self.rationale = kwargs.get('rationale', '')
        if not hasattr(self, 'model_version'):
            self.model_version = kwargs.get('model_version')
        if not hasattr(self, 'prompt_template_hash'):
            self.prompt_template_hash = kwargs.get('prompt_template_hash')
        if not hasattr(self, 'validated'):
            self.validated = kwargs.get('validated', False)
        if not hasattr(self, 'validation_failures'):
            self.validation_failures = kwargs.get('validation_failures', [])


class CauseEffectCheck(BaseModel):
    claim_id: str = ""
    contradiction: bool = False
    evidence_ids: List[str] = []
    rationale: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'claim_id'):
            self.claim_id = kwargs.get('claim_id', '')
        if not hasattr(self, 'contradiction'):
            self.contradiction = kwargs.get('contradiction', False)
        if not hasattr(self, 'evidence_ids'):
            self.evidence_ids = kwargs.get('evidence_ids', [])
        if not hasattr(self, 'rationale'):
            self.rationale = kwargs.get('rationale', '')
