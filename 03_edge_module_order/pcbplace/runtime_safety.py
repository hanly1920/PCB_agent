from __future__ import annotations

from typing import Any, Mapping, Set

RUNTIME_PRIOR_SCHEMA_VERSION = "runtime_prior_v1"
RUNTIME_SHAPE_HINT_SCHEMA_VERSION = "runtime_shape_hint_v1"

ALLOWED_RUNTIME_PRIOR_SCHEMA_VERSIONS: Set[str] = {RUNTIME_PRIOR_SCHEMA_VERSION}
ALLOWED_RUNTIME_SHAPE_HINT_SCHEMA_VERSIONS: Set[str] = {RUNTIME_SHAPE_HINT_SCHEMA_VERSION}

# Runtime prior sources produced by the built-in rule generator.  Keep this
# deliberately narrow: new generators must opt in by adding a new schema/source.
ALLOWED_RUNTIME_PRIOR_SOURCE_PREFIXES = ("rule_v1_", "rule_v2_", "rule_v3_")
ALLOWED_RUNTIME_PRIOR_SOURCES = {
    "prior_region_existing",
    "prior_region",
}

# Runtime shape hints are either derived from the prior generator or synthesized
# inside PlacementEnv from leakage-safe prior/module metadata.
ALLOWED_RUNTIME_SHAPE_HINT_SOURCE_PREFIXES = ("prior_rule_v1_", "prior_rule_v2_", "prior_rule_v3_", "runtime_derived_")
ALLOWED_RUNTIME_SHAPE_HINT_SOURCES = {
    "prior_region_existing",
    "runtime_derived_from_prior_region",
}


def _is_safe_bool(payload: Mapping[str, Any]) -> bool:
    return payload.get("leakage_safe") is True


def _schema_version(payload: Mapping[str, Any]) -> str:
    return str(payload.get("schema_version") or payload.get("version_tag") or "")


def _source(payload: Mapping[str, Any]) -> str:
    return str(payload.get("source") or "")


def is_runtime_prior_allowed(prior: Mapping[str, Any]) -> bool:
    if not _is_safe_bool(prior):
        return False
    if _schema_version(prior) not in ALLOWED_RUNTIME_PRIOR_SCHEMA_VERSIONS:
        return False
    src = _source(prior)
    if "expert" in src.lower():
        return False
    return src in ALLOWED_RUNTIME_PRIOR_SOURCES or src.startswith(ALLOWED_RUNTIME_PRIOR_SOURCE_PREFIXES)


def is_runtime_shape_hint_allowed(hint: Mapping[str, Any]) -> bool:
    if not _is_safe_bool(hint):
        return False
    if _schema_version(hint) not in ALLOWED_RUNTIME_SHAPE_HINT_SCHEMA_VERSIONS:
        return False
    src = _source(hint)
    low = src.lower()
    if "expert" in low or str(hint.get("derived_from") or "").lower() == "expert":
        return False
    return src in ALLOWED_RUNTIME_SHAPE_HINT_SOURCES or src.startswith(ALLOWED_RUNTIME_SHAPE_HINT_SOURCE_PREFIXES)


def describe_runtime_prior_violation(prior: Mapping[str, Any], *, path: str = "prior_region") -> str:
    if not _is_safe_bool(prior):
        return f"{path}.leakage_safe must be true"
    if _schema_version(prior) not in ALLOWED_RUNTIME_PRIOR_SCHEMA_VERSIONS:
        return (
            f"{path}.schema_version must be one of "
            f"{sorted(ALLOWED_RUNTIME_PRIOR_SCHEMA_VERSIONS)}"
        )
    src = _source(prior)
    if "expert" in src.lower():
        return f"{path}.source must not reference expert labels"
    return f"{path}.source is not whitelisted: {src!r}"


def describe_runtime_shape_hint_violation(hint: Mapping[str, Any], *, path: str = "shape_hint") -> str:
    if not _is_safe_bool(hint):
        return f"{path}.leakage_safe must be true"
    if _schema_version(hint) not in ALLOWED_RUNTIME_SHAPE_HINT_SCHEMA_VERSIONS:
        return (
            f"{path}.schema_version must be one of "
            f"{sorted(ALLOWED_RUNTIME_SHAPE_HINT_SCHEMA_VERSIONS)}"
        )
    src = _source(hint)
    low = src.lower()
    if "expert" in low or str(hint.get("derived_from") or "").lower() == "expert":
        return f"{path}.source/derived_from must not reference expert labels"
    return f"{path}.source is not whitelisted: {src!r}"
