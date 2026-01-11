# pipeline_nli

This pipeline validates whether a backstory is compatible with a novel using Natural Language Inference (NLI).

## Why NLI?
NLI models explicitly reason about whether a hypothesis (the claim) is entailed, contradicted, or neutral given a premise (novel text). This makes them a natural fit for contradiction detection: contradictions indicate a claim that is incompatible with the novel.

## How this differs from the Pathway+heuristic approach
- This pipeline uses NLI classification on textual claim vs. retrieved context pairs, rather than handcrafted heuristics over retrieved facts.
- Pathway (or a deterministic TF-IDF fallback) is used only for retrieval/indexing; inference is done with a pretrained transformer NLI model.
- All steps are deterministic and traceable: claim → chunk → NLI label.

## Strengths
- More semantic verification (captures paraphrases, implicit support) compared to simple keyword heuristics.
- Traceability: each decision can be traced back to the contributing chunks and NLI labels.

## Limitations
- NLI models can be sensitive to premise length and domain shifts.
- Requires good retrieval: if relevant evidence isn't retrieved, the NLI model cannot support or contradict the claim.
- No generative LLMs used—this preserves determinism but limits reasoning complexity.

## How to run
1. Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
# If you plan to use Pathway, install it:
# pip install pathway
```

2. Run full pipeline:

```bash
python pipeline_nli/run_pipeline.py
```

3. Run steps individually:

```bash
python pipeline_nli/run_pipeline.py --steps step1 step2
```

Artifacts are created under `artifacts/pipeline_nli/`.

## Notes
- This pipeline attempts to use `pathway` for indexing if available; otherwise it falls back to a deterministic TF-IDF/overlap retriever.
- To change NLI model, edit `pipeline_nli/step0_config.py` (NLI_MODEL).
