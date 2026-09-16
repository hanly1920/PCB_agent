# PCB Agent MVP

This layer turns the existing `pcbplace` placement model into a deterministic, replayable agent workflow:

`bundle -> DSL -> placement candidates -> scoring -> KiCad write-back -> routing adapter -> KiCad DRC -> repair loop -> report`

## What is implemented

- Pydantic v2 schemas for design bundles, DSL, tool results, candidates, DRC and agent state.
- Qwen3 client for Ollama and OpenAI-compatible endpoints, with schema validation and one repair pass.
- Offline heuristic DSL compiler for reproducible tests.
- Checkpoint validation and real CUDA `pcbplace.infer_layout` integration.
- Deterministic multi-candidate mock placement for CPU smoke tests.
- KiCad footprint `(at x y rotation)` write-back that preserves all other board text.
- Freerouting wrapper with explicit DSN export / SES import adapters.
- KiCad JSON DRC runner and parser; exit code 5 is treated as a valid report.
- Deterministic state machine, repair policy, `state.json`, `run_history.jsonl`, `metrics.json` and `report.md`.
- CLI, FastAPI skeleton and unit/smoke tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Place the supplied checkpoint at a path of your choice; it is intentionally not duplicated in this source archive.

## Prove the whole workflow on any CPU

```bash
python -m pcb_agent.cli.pcb_agent run   --bundle design_bundle.example   --out runs/smoke   --mock
```

Expected final status: `EXPORT_RESULT`. Artifacts appear in `runs/smoke/`.

## Validate the supplied checkpoint

```bash
python -m pcb_agent.cli.pcb_agent validate-checkpoint   --ckpt /path/to/latest_w4000_k3000.pt
```

## Compile natural language to DSL

```bash
python -m pcb_agent.cli.pcb_agent compile-dsl   --text "J1靠左，U1居中，去耦电容靠近U1"
```

The default config uses the mock compiler. Use `config.example.yaml` and set `llm.provider: ollama` for local Qwen3.

## Real placement / DRC run

```bash
python -m pcb_agent.cli.pcb_agent run   --bundle /path/to/design_bundle   --out runs/real   --config config.yaml   --ckpt /path/to/latest_w4000_k3000.pt   --device cuda
```

The bundled `pcbplace.infer_layout` is CUDA-only. A CPU run must use `--mock` unless the base model code is extended.

## Freerouting integration boundary

KiCad CLI does not provide a universally reliable headless DSN/SES roundtrip across installations. Therefore real routing requires two configured adapter commands:

- `router.export_dsn_cmd`: board -> DSN
- `router.import_ses_cmd`: board + SES -> routed board

The agent invokes Freerouting between them and still uses KiCad DRC as the final acceptance gate. This avoids inventing a non-existent KiCad command.

## Tests

```bash
PYTHONPATH=. pytest -q tests_agent
```

## Remaining engineering work

- Supply installation-specific DSN/SES adapters or a KiCad 9 IPC plugin.
- Add stochastic sampling/local-freeze support to `pcbplace.infer_layout` for stronger candidate diversity.
- Map all DSL hard constraints into `PlacementEnv` masks rather than objective overrides only.
- Add real board preview export and richer congestion/routing metrics.
- Exercise integration tests on a machine with KiCad 9, Freerouting, Ollama Qwen3 and CUDA.
