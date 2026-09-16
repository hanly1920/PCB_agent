# Fine Modules v2

Run a full rebuild and audit:

```bash
PYTHONPATH=. python tools/refine_modules_and_check_dataset.py --data-root data --splits train,infer
```

The tool regenerates each board from `data/audits/<split>/*.source.json` where available.  It writes fresh `data/<split>/*.json`, fresh audit JSON, and a report.  Train boards are checked with a full expert-guided legal rollout; infer boards have no expert labels, so their built-in smoke check covers an initial legal prefix.

Key rules:

- Physical connectors are isolated from relay/power/core anchors.
- Interface support must share a direct signal net with its connector.
- Narrow edge priors require a physical interface with an explicit side constraint.
- Uncertain interface/power placement gets a broad low-confidence prior rather than a false edge prior.
