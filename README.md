# PCB_agent

This repository contains ordered code snapshots for the StructPlace-PCB / PCB agent development sequence.

## Version timeline

1. `01_chip_centered_order` — chip-centered placement order.
2. `02_interface_anchor_order` — interface/anchor-centered placement order.
3. `03_edge_module_order` — edge/module-aware intermediate version.
4. `04_six_stage_module_aware_order` — six-stage module-aware placement order.
5. `05_six_stage_edge_interface_anchor` — six-stage order with edge/interface anchor handling.
6. `06_full_agent_ollama_online_rl` — latest full PCB agent version with local Ollama LLM analysis and online replay/RL workflow.

Only source code and lightweight documentation are tracked. Checkpoints, datasets, logs, inference outputs, virtual environments, archives, spreadsheets, and generated figures are excluded.

## Notes

The code snapshots are kept as separate directories to make the ablation timeline explicit. Each directory has its own `VERSION_README.md` describing its position in the timeline.
