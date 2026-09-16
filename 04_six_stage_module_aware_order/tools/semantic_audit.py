from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _semantic_value(comp: Dict[str, Any], key: str, default: Any = None) -> Any:
    if comp.get(key) not in (None, ""):
        return comp.get(key)
    sem = comp.get("semantic") if isinstance(comp.get("semantic"), dict) else {}
    if sem.get(key) not in (None, ""):
        return sem.get(key)
    layout = comp.get("layout") if isinstance(comp.get("layout"), dict) else {}
    if layout.get(key) not in (None, ""):
        return layout.get(key)
    return default


def _has(comp: Dict[str, Any], key: str) -> bool:
    return _semantic_value(comp, key, None) not in (None, "", [], {})


def audit_file(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    comps = data.get("components") or []
    n = len(comps)
    cov = {
        "semantic_class": sum(1 for c in comps if _has(c, "semantic_class")),
        "module_id": sum(1 for c in comps if _has(c, "module_id")),
        "functional_group": sum(1 for c in comps if _has(c, "functional_group")),
        "align_group": sum(1 for c in comps if _has(c, "align_group")),
        "align_axis": sum(1 for c in comps if _has(c, "align_axis")),
        "same_side_group": sum(1 for c in comps if _has(c, "same_side_group")),
        "boundary_order": sum(1 for c in comps if _has(c, "boundary_order")),
        "critical_neighbors": sum(1 for c in comps if _has(c, "critical_neighbors")),
        "semantic_strength": sum(1 for c in comps if _has(c, "semantic_strength")),
        "constraint_source": sum(1 for c in comps if _has(c, "constraint_source")),
        "constraint_level": sum(1 for c in comps if _has(c, "constraint_level")),
    }
    needs_review = sum(1 for c in comps if bool((_semantic_value(c, "semantic_review", {}) or {}).get("needs_review", False)) or bool(_semantic_value(c, "needs_review", False)))
    levels = Counter(str(_semantic_value(c, "constraint_level", "soft") or "soft").lower() for c in comps)
    sources = Counter(str(_semantic_value(c, "constraint_source", "auto") or "auto").lower() for c in comps)
    return {
        "file": str(path),
        "n_components": n,
        "coverage": cov,
        "coverage_ratio": {k: (v / n if n else 0.0) for k, v in cov.items()},
        "needs_review_count": needs_review,
        "constraint_level_counts": dict(levels),
        "constraint_source_counts": dict(sources),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_n = sum(int(r.get("n_components", 0)) for r in rows)
    keys = sorted({k for r in rows for k in (r.get("coverage") or {})})
    cov = {k: sum(int((r.get("coverage") or {}).get(k, 0)) for r in rows) for k in keys}
    levels = Counter()
    sources = Counter()
    for r in rows:
        levels.update(r.get("constraint_level_counts") or {})
        sources.update(r.get("constraint_source_counts") or {})
    return {
        "n_files": len(rows),
        "n_components": total_n,
        "coverage": cov,
        "coverage_ratio": {k: (v / total_n if total_n else 0.0) for k, v in cov.items()},
        "needs_review_count": sum(int(r.get("needs_review_count", 0)) for r in rows),
        "constraint_level_counts": dict(levels),
        "constraint_source_counts": dict(sources),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="JSON task glob, e.g. data/**/*.json")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.glob, recursive=True))
    rows = [audit_file(p) for p in paths]
    payload = {"summary": summarize(rows), "files": rows}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
