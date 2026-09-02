"""Resolved, persistent per-core search policies.

Configuration expresses intent.  A policy snapshot records the exact result
after CPU-generation limits and topology evidence have been applied, so resume
never silently changes behavior after a kernel/sysfs or application update.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corecycler.engine.topology import CPUTopology

    from .config import TunerConfig

POLICY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CorePolicy:
    max_offset: int
    coarse_step: int
    confirm_multiplier: float = 1.0
    core_class: str = "standard"
    source: str = "global"


@dataclass(slots=True)
class PolicySnapshot:
    policies: dict[int, CorePolicy]
    topology: dict[str, object]
    x3d: dict[str, object]
    warnings: list[str]
    positive_acknowledged: bool = False
    schema_version: int = POLICY_SCHEMA_VERSION

    def to_json(self) -> str:
        data = {
            "schema_version": self.schema_version,
            "topology": self.topology,
            "x3d": self.x3d,
            "policies": {str(core): asdict(policy) for core, policy in sorted(self.policies.items())},
            "warnings": self.warnings,
            "positive_acknowledged": self.positive_acknowledged,
        }
        return json.dumps(data, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> PolicySnapshot | None:
        """Strictly load a snapshot. Empty JSON denotes a legacy session."""
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed policy_json") from exc
        if data == {}:
            return None
        if not isinstance(data, dict) or data.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported policy snapshot schema")
        try:
            raw_policies = data["policies"]
            topology = data["topology"]
            x3d = data["x3d"]
            warnings = data["warnings"]
            acknowledged = data["positive_acknowledged"]
        except KeyError as exc:
            raise ValueError(f"policy snapshot missing {exc.args[0]}") from exc
        if not isinstance(raw_policies, dict) or not isinstance(topology, dict) or not isinstance(x3d, dict):
            raise ValueError("invalid policy snapshot objects")
        if not isinstance(warnings, list) or not all(isinstance(v, str) for v in warnings):
            raise ValueError("invalid policy warnings")
        if not isinstance(acknowledged, bool):
            raise ValueError("invalid positive acknowledgement")
        policies: dict[int, CorePolicy] = {}
        try:
            for key, value in raw_policies.items():
                if not isinstance(key, str) or not key.isdigit() or not isinstance(value, dict):
                    raise ValueError("invalid per-core policy")
                if set(value) != {"max_offset", "coarse_step", "confirm_multiplier", "core_class", "source"}:
                    raise ValueError("invalid per-core policy fields")
                policy = CorePolicy(**value)
                if (
                    not isinstance(policy.max_offset, int)
                    or isinstance(policy.max_offset, bool)
                    or not isinstance(policy.coarse_step, int)
                    or isinstance(policy.coarse_step, bool)
                    or policy.coarse_step < 1
                    or not isinstance(policy.confirm_multiplier, (int, float))
                    or isinstance(policy.confirm_multiplier, bool)
                    or policy.confirm_multiplier <= 0
                    or policy.core_class not in ("vcache", "standard")
                    or policy.source not in ("global", "automatic", "forced", "override")
                ):
                    raise ValueError("invalid per-core policy values")
                policies[int(key)] = policy
        except TypeError as exc:
            raise ValueError("invalid per-core policy values") from exc
        return cls(policies, topology, x3d, warnings, acknowledged)

    def validate_topology(self, current: CPUTopology) -> list[str]:
        """Return hard topology mismatches; cache evidence may disappear safely."""
        expected = self.topology.get("core_ccds")
        actual = {str(core): info.ccd for core, info in sorted(current.cores.items())}
        errors: list[str] = []
        if expected != actual:
            errors.append("physical core/CCD topology changed since the session was created")
        identity = self.topology.get("cpu_identity", {})
        if isinstance(identity, dict):
            for key, raw_value in (("family", current.family), ("model", current.model)):
                value = raw_value if isinstance(raw_value, int) and not isinstance(raw_value, bool) else 0
                if identity.get(key) != value:
                    errors.append(f"CPU {key} changed since the session was created")
        return errors


def resolve_policy(
    config: TunerConfig,
    topology: CPUTopology,
    co_range: tuple[int, int],
    *,
    positive_acknowledged: bool = False,
) -> PolicySnapshot:
    """Resolve generation, global, X3D, then explicit per-core policy."""
    global_max = max(config.max_offset, co_range[0]) if config.direction < 0 else min(config.max_offset, co_range[1])
    detection = getattr(topology, "x3d_detection", "none")
    if detection not in ("none", "model_only", "cache_verified", "cache_only", "ambiguous"):
        detection = "none"
    raw_vcache = getattr(topology, "vcache_ccds", frozenset())
    detected_vcache = (
        frozenset(value for value in raw_vcache if isinstance(value, int) and not isinstance(value, bool))
        if isinstance(raw_vcache, (set, frozenset, list, tuple))
        else frozenset()
    )
    warnings: list[str] = []
    available_cores = set(topology.cores)
    selected_cores = set(config.cores_to_test) if config.cores_to_test is not None else available_cores
    missing_cores = selected_cores - available_cores
    overridden_cores = {int(core) for core in config.core_policy_overrides if core.isdigit()}
    missing_overrides = overridden_cores - selected_cores
    if missing_cores:
        raise ValueError(f"configured cores do not exist: {sorted(missing_cores)}")
    if missing_overrides:
        raise ValueError(f"policy overrides target untested cores: {sorted(missing_overrides)}")
    if config.x3d_mode == "off":
        effective = frozenset()
        mapping_source = "off"
    elif config.x3d_mode == "force":
        available = {info.ccd for info in topology.cores.values() if info.ccd is not None}
        requested = frozenset(config.x3d_force_vcache_ccds)
        missing = requested - available
        if missing:
            raise ValueError(f"forced V-Cache CCDs do not exist: {sorted(missing)}")
        effective = requested
        mapping_source = "forced"
        if detected_vcache and detected_vcache != effective:
            warnings.append("forced V-Cache CCD mapping contradicts detected cache evidence")
    elif detection == "ambiguous":
        effective = frozenset()
        mapping_source = "ambiguous-fallback"
        warnings.append("X3D topology is ambiguous; using uniform global settings without guessing a CCD")
    else:
        effective = detected_vcache
        mapping_source = "automatic"

    if config.direction > 0 and effective and not positive_acknowledged:
        raise ValueError("positive X3D tuning requires explicit acknowledgement")

    policies: dict[int, CorePolicy] = {}
    for core, info in sorted(topology.cores.items()):
        if core not in selected_cores:
            continue
        is_vcache = info.ccd in effective
        max_offset = global_max
        coarse_step = config.coarse_step
        multiplier = 1.0
        source = "global"
        if is_vcache:
            if config.direction < 0:
                max_offset = max(global_max, config.x3d_vcache_negative_floor, co_range[0])
            coarse_step = config.x3d_vcache_coarse_step
            multiplier = config.x3d_vcache_confirm_multiplier
            source = "forced" if config.x3d_mode == "force" else "automatic"
        override = config.core_policy_overrides.get(str(core), {})
        if override:
            if "max_offset" in override:
                raw_max = int(override["max_offset"])
                max_offset = max(raw_max, co_range[0]) if config.direction < 0 else min(raw_max, co_range[1])
            if "coarse_step" in override:
                coarse_step = int(override["coarse_step"])
            if "confirm_multiplier" in override:
                multiplier = float(override["confirm_multiplier"])
            source = "override"
        policies[core] = CorePolicy(
            max_offset=max_offset,
            coarse_step=coarse_step,
            confirm_multiplier=multiplier,
            core_class="vcache" if is_vcache else "standard",
            source=source,
        )

    l3_sizes = getattr(topology, "ccd_l3_sizes_kib", {})
    if not isinstance(l3_sizes, dict):
        l3_sizes = {}

    def _integer(name: str) -> int:
        value = getattr(topology, name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    topology_data: dict[str, object] = {
        "cpu_identity": {
            "model_name": topology.model_name,
            "vendor": topology.vendor if isinstance(topology.vendor, str) else "",
            "family": _integer("family"),
            "model": _integer("model"),
            "stepping": _integer("stepping"),
        },
        "core_ccds": {str(core): info.ccd for core, info in sorted(topology.cores.items())},
        "ccd_l3_sizes_kib": {
            str(ccd): size
            for ccd, size in sorted(l3_sizes.items())
            if isinstance(ccd, int) and isinstance(size, int)
        },
    }
    x3d_data: dict[str, object] = {
        "mode": config.x3d_mode,
        "detection": detection,
        "detected_vcache_ccds": sorted(detected_vcache),
        "effective_vcache_ccds": sorted(effective),
        "mapping_source": mapping_source,
    }
    return PolicySnapshot(policies, topology_data, x3d_data, warnings, positive_acknowledged)


def legacy_policy(config: TunerConfig, cores: list[int]) -> dict[int, CorePolicy]:
    """Uniform behavior for sessions created before policy snapshots existed."""
    return {core: CorePolicy(config.max_offset, config.coarse_step) for core in cores}
