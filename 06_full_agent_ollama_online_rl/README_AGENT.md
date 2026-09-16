# PCB Agent

This layer turns the existing `pcbplace` placement model into a deterministic, replayable end-to-end PCB agent workflow. The agent can start from a bundle directory, a bundle manifest, or a `.kicad_pcb` board; it generates or reuses a structured task, compiles user intent to LayoutDSL, creates legal placement candidates, writes the selected layout back to KiCad, optionally routes and runs DRC, asks a local LLM layout critic to analyze the result, tunes replay parameters, performs a run-local online policy-gradient/self-imitation update from the selected trajectory, saves a new session checkpoint, and then runs another placement round or exports the best acceptable board:

`bundle -> DSL -> placement candidates -> scoring -> KiCad write-back -> routing adapter -> KiCad DRC -> LLM tuning -> online checkpoint update -> replay loop -> report`

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
- Flexible input: `--input` accepts a bundle directory, bundle manifest, or direct `.kicad_pcb` board path.
- Automatic tool fallback: when Freerouting adapters or KiCad DRC are unavailable and fallback is enabled, the agent exports a placement-level artifact instead of aborting the run.
- Round artifacts for review: each round writes `candidates.json`, `candidate_ranking.json`, `selected_candidate.json`, and, when needed, `repair_plan.json`.
- Acceptance modes recorded in `metrics.json`: `drc_clean`, `placement_accepted_without_drc`, or `best_effort_after_max_rounds`.
- Local Ollama/Qwen3 critic emits a validated `TuningPlan`; coordinate-like keys are rejected, so the LLM can tune constraints and replay parameters but cannot directly place components.
- Online checkpoint update: the selected candidate is converted into a temporary single-board replay trajectory, fresh sampled rollouts are collected on the current board, policy-gradient and self-imitation losses update the model weights, and a new session checkpoint is saved under `online_checkpoints/` without overwriting the base checkpoint.

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
python -m pcb_agent.cli.pcb_agent run   --input design_bundle.example   --out runs/smoke   --mock
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

## Real placement / DRC run with online policy update

```bash
python -m pcb_agent.cli.pcb_agent run   --input /path/to/design_or_board   --out runs/real   --config config.agent.example.yaml   --device cuda
```

The bundled `pcbplace.infer_layout` and online replay update are CUDA-only. A CPU run must use `--mock` or `--no-online-finetune` unless the base model code is extended.

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

- Supply installation-specific DSN/SES adapters or a KiCad 9 IPC plugin for production route+DRC closure.
- Map all DSL hard constraints into `PlacementEnv` masks rather than objective overrides only.
- Add real board preview export and richer congestion/routing metrics.
- Exercise integration tests on a machine with KiCad 9, Freerouting, Ollama Qwen3 and CUDA.

## Placement-only automatic fallback

For machines without a configured Freerouting DSN/SES adapter or KiCad DRC setup, the agent can still run automatically and export a reviewed placement artifact:

```bash
python -m pcb_agent.cli.pcb_agent run \
  --input /path/to/board.kicad_pcb \
  --out runs/place_only \
  --config config.yaml \
  --ckpt /path/to/latest_w4000_k3000.pt \
  --device cuda \
  --skip-routing \
  --skip-drc
```

Use `--require-drc-clean` and `--no-tool-fallback` when a production run must fail unless routing and DRC complete successfully.

## Local Ollama/Qwen3 critic

The default example config uses:

```yaml
llm:
  provider: ollama
  endpoint: http://localhost:11434
  model: qwen3:8b
placement:
  checkpoint_path: <path-to-checkpoint>/train-floorplan-mixed-b3_3-reply20.pt
online_finetune:
  enabled: true
  default_policy_updates: 1
```

The LLM is a critic and tuning planner only. It is not allowed to emit coordinates; its JSON output is validated as `TuningPlan`, and coordinate-like keys are rejected before the plan is applied.

## Online checkpoint update

Online finetuning is controlled by `online_finetune` in `config.agent.example.yaml`. The base checkpoint remains unchanged. Each update writes a new checkpoint such as:

```text
runs/real/online_checkpoints/round_00_online_policy.pt
```

The orchestrator then points the next placement round to that new checkpoint. Use `--no-online-finetune` for an ablation that keeps the policy frozen while still allowing LLM-guided DSL and replay-parameter tuning.
