"""Minimal LLM provider stub.

Replace or extend `llm_call` to call your preferred provider (OpenAI, Anthropic, etc.).
This stub intentionally returns safe, deterministic JSON outputs so the pipeline can run
in `parallel` mode without external credentials.
"""
import json
from typing import Callable


def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """A minimal, deterministic LLM-call stub.

    For production, replace this with a provider implementation that returns
    the model's raw text output (string). The function should accept (system, user, temperature)
    and return a string containing JSON only, as expected by the prompts in `prompts/`.

    For safety in the repository, the stub inspects the system prompt and returns
    a conservative JSON response that validates against schemas.
    """
    sp = (system_prompt or "").lower()
    # Claim extractor: return empty claims by default
    if 'claim extractor' in sp:
        return json.dumps({"claims": []})
    # Reasoner: default to INSUFFICIENT
    if 'reasoner' in sp:
        return json.dumps({"label": "INSUFFICIENT", "evidence_ids": [], "rationale": "no evidence provided"})
    # Cause-effect checker: default to no contradiction
    if 'cause' in sp and 'checker' in sp:
        return json.dumps({"contradiction": False, "evidence_ids": [], "rationale": "no contradiction in provided chunks"})
    # Fallback safe output
    return json.dumps({})
