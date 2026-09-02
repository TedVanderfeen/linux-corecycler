"""X3D and generic per-core policy resolution/persistence contracts."""

from __future__ import annotations

import json

import pytest

from corecycler.engine.topology import CPUTopology, PhysicalCore
from corecycler.tuner.config import TunerConfig
from corecycler.tuner.policy import PolicySnapshot, resolve_policy


def _topology(*, detection: str = "cache_verified", vcache=frozenset({0})) -> CPUTopology:
    return CPUTopology(
        model_name="AMD Ryzen 9 9950X3D",
        vendor="AuthenticAMD",
        family=26,
        model=68,
        stepping=2,
        ccds=2,
        is_x3d=True,
        vcache_ccd=next(iter(vcache)) if len(vcache) == 1 else None,
        vcache_ccds=vcache,
        ccd_l3_sizes_kib={0: 98304, 1: 32768},
        x3d_detection=detection,
        cores={
            0: PhysicalCore(0, 0, None, (0, 16), 0 in vcache),
            1: PhysicalCore(1, 1, None, (1, 17), 1 in vcache),
        },
    )


def test_moderate_vcache_policy_and_standard_global_policy():
    snap = resolve_policy(TunerConfig(max_offset=-50, coarse_step=8), _topology(), (-60, 10))
    assert (snap.policies[0].max_offset, snap.policies[0].coarse_step) == (-25, 3)
    assert snap.policies[0].confirm_multiplier == 1.5
    assert snap.policies[0].core_class == "vcache"
    assert (snap.policies[1].max_offset, snap.policies[1].coarse_step) == (-50, 8)


def test_less_aggressive_global_floor_wins_for_vcache():
    snap = resolve_policy(TunerConfig(max_offset=-20), _topology(), (-60, 10))
    assert snap.policies[0].max_offset == -20


def test_generation_clamp_precedes_override():
    cfg = TunerConfig(
        core_policy_overrides={"1": {"max_offset": -55, "coarse_step": 2, "confirm_multiplier": 2.0}}
    )
    snap = resolve_policy(cfg, _topology(), (-30, 30))
    assert snap.policies[1].max_offset == -30
    assert snap.policies[1].coarse_step == 2
    assert snap.policies[1].confirm_multiplier == 2.0
    assert snap.policies[1].source == "override"


def test_ambiguous_auto_falls_back_uniform_with_warning():
    snap = resolve_policy(TunerConfig(max_offset=-40), _topology(detection="ambiguous", vcache=frozenset()), (-60, 10))
    assert {p.max_offset for p in snap.policies.values()} == {-40}
    assert snap.x3d["mapping_source"] == "ambiguous-fallback"
    assert snap.warnings


def test_force_mode_selects_multiple_ccds_and_detects_contradiction():
    cfg = TunerConfig(x3d_mode="force", x3d_force_vcache_ccds=[1])
    snap = resolve_policy(cfg, _topology(), (-60, 10))
    assert snap.policies[1].core_class == "vcache"
    assert snap.policies[0].core_class == "standard"
    assert snap.warnings


def test_force_rejects_nonexistent_ccd():
    cfg = TunerConfig(x3d_mode="force", x3d_force_vcache_ccds=[2])
    with pytest.raises(ValueError, match="do not exist"):
        resolve_policy(cfg, _topology(), (-60, 10))


def test_positive_x3d_requires_ack_but_retains_step_and_multiplier():
    cfg = TunerConfig(direction=1, max_offset=8)
    with pytest.raises(ValueError, match="acknowledgement"):
        resolve_policy(cfg, _topology(), (-60, 10))
    snap = resolve_policy(cfg, _topology(), (-60, 10), positive_acknowledged=True)
    assert snap.policies[0].max_offset == 8
    assert snap.policies[0].coarse_step == 3
    assert snap.policies[0].confirm_multiplier == 1.5
    assert snap.positive_acknowledged is True


def test_snapshot_roundtrip_and_topology_mismatch():
    snap = resolve_policy(TunerConfig(), _topology(), (-60, 10))
    restored = PolicySnapshot.from_json(snap.to_json())
    assert restored is not None
    assert restored.policies == snap.policies
    changed = _topology()
    changed.cores[1] = PhysicalCore(1, 0, None, (1, 17), True)
    assert restored.validate_topology(changed)
    identity_changed = _topology()
    identity_changed.model += 1
    assert any("model changed" in error for error in restored.validate_topology(identity_changed))


@pytest.mark.parametrize("raw", ["{", "[]", '{"schema_version":99}', '{"schema_version":1}'])
def test_malformed_policy_rejected(raw):
    with pytest.raises(ValueError):
        PolicySnapshot.from_json(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(policies=[]),
        lambda d: d.update(warnings="bad"),
        lambda d: d.update(positive_acknowledged=1),
        lambda d: d["policies"].update({"bad": {}}),
        lambda d: d["policies"]["0"].update(extra=1),
        lambda d: d["policies"]["0"].update(coarse_step=0),
        lambda d: d["policies"]["0"].update(max_offset=None),
    ],
)
def test_structurally_invalid_snapshots_are_rejected(mutate):
    data = json.loads(resolve_policy(TunerConfig(), _topology(), (-60, 10)).to_json())
    mutate(data)
    with pytest.raises(ValueError):
        PolicySnapshot.from_json(json.dumps(data))


def test_bad_policy_constructor_shape_is_rejected():
    data = json.loads(resolve_policy(TunerConfig(), _topology(), (-60, 10)).to_json())
    data["policies"]["0"] = {
        "max_offset": -25,
        "coarse_step": 3,
        "confirm_multiplier": 1.5,
        "core_class": "vcache",
        "source": "automatic",
        "unexpected": 1,
    }
    with pytest.raises(ValueError):
        PolicySnapshot.from_json(json.dumps(data))


def test_policy_constructor_type_error_is_normalized(monkeypatch):
    data = resolve_policy(TunerConfig(), _topology(), (-60, 10)).to_json()
    monkeypatch.setattr("corecycler.tuner.policy.CorePolicy", lambda **_kwargs: (_ for _ in ()).throw(TypeError()))
    with pytest.raises(ValueError, match="invalid per-core policy values"):
        PolicySnapshot.from_json(data)


def test_empty_snapshot_is_legacy_marker():
    assert PolicySnapshot.from_json("{}") is None


def test_untested_override_and_missing_core_are_refused():
    with pytest.raises(ValueError, match="untested"):
        resolve_policy(
            TunerConfig(cores_to_test=[0], core_policy_overrides={"1": {"max_offset": -20}}),
            _topology(),
            (-60, 10),
        )
    with pytest.raises(ValueError, match="do not exist"):
        resolve_policy(TunerConfig(cores_to_test=[9]), _topology(), (-60, 10))


def test_snapshot_json_contains_evidence_and_exact_per_core_policy():
    data = json.loads(resolve_policy(TunerConfig(), _topology(), (-60, 10)).to_json())
    assert data["schema_version"] == 1
    assert data["topology"]["ccd_l3_sizes_kib"] == {"0": 98304, "1": 32768}
    assert data["x3d"]["effective_vcache_ccds"] == [0]
    assert data["policies"]["0"]["max_offset"] == -25


def test_off_mode_and_positive_standard_policy():
    off = resolve_policy(TunerConfig(x3d_mode="off"), _topology(), (-60, 10))
    assert off.x3d["mapping_source"] == "off"
    assert all(policy.core_class == "standard" for policy in off.policies.values())
    positive = resolve_policy(TunerConfig(direction=1, max_offset=30), _topology(vcache=frozenset()), (-60, 10))
    assert all(policy.max_offset == 10 for policy in positive.policies.values())


def test_non_dataclass_like_topology_falls_back_to_safe_primitives():
    from unittest.mock import MagicMock

    topo = MagicMock()
    topo.cores = {0: PhysicalCore(0, 0, None, (0,))}
    topo.model_name = "fixture"
    snap = resolve_policy(TunerConfig(), topo, (-60, 10))
    assert snap.topology["cpu_identity"]["family"] == 0
    assert snap.x3d["detection"] == "none"
