# Other experiments

The folders above `experiments/` hold the code of the best submission of each phase. This folder holds the code of the other
experiments and submissions: Arch A, C and D, context-aware decoding, diagnostics, unit tests, knowledge-distillation teachers, the
Phase-7 cross-encoder, and the Phase-3 last-day rule experiments.

The layout mirrors the repository root. To run any of it, merge it into the phase folders from the repository root:

```bash
cp -r experiments/phase[1-7] .
```

This adds files and restores the full model registry in `phase1/models/__init__.py`. Nothing else is replaced.

- `run_commands/phaseN_runs.sh`: the recorded arguments of every Phase 2-7 run, generated from the saved run configurations.
- `phase1/scripts/`: the shell scripts behind the Phase-1 submissions and evaluations.
- `phase1/sft/llm_*.py` and `label_specifics_llm.py`: a local labelling pilot over 200 pairs, not used in any submitted model.
