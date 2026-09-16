# Full agent with Ollama-guided online replay

This is the latest agent-oriented version in the development timeline.

It is intended to connect the trained placement policy with a PCB-agent workflow, including task parsing, LayoutDSL-style intent handling, layout analysis, replay-based optimization, and local LLM feedback through an Ollama deployment such as qwen3:8b.

Large local artifacts are intentionally excluded: checkpoints, datasets, logs, inference outputs, temporary files, virtual environments, spreadsheets, and generated figures.
