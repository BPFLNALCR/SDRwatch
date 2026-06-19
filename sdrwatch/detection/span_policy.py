"""Profile-governed signal span policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


def _get_attr(source: Any, attr: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(attr, default)
    return getattr(source, attr, default)


def _clean_positive_float(value: Any, *, invalid: list[str], field_name: str) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        invalid.append(field_name)
        return None
    if parsed <= 0.0:
        if parsed < 0.0:
            invalid.append(field_name)
        return None
    return parsed


def _clean_text(value: Any, default: str) -> str:
    if value in (None, ""):
        return default
    text = str(value).strip()
    return text or default


def _clean_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


@dataclass(frozen=True)
class SignalSpanPolicy:
    profile_name: Optional[str] = None
    min_identity_bandwidth_hz: Optional[float] = None
    min_persist_bandwidth_hz: Optional[float] = None
    max_persist_bandwidth_hz: Optional[float] = None
    min_match_bandwidth_hz: Optional[float] = None
    min_display_bandwidth_hz: Optional[float] = None
    allow_revisit_to_shrink_identity: bool = True
    allow_revisit_to_move_center: bool = True
    min_revisit_bandwidth_for_identity_update_hz: Optional[float] = None
    max_revisit_center_delta_for_identity_update_hz: Optional[float] = None
    fragmented_revisit_policy: str = "confirmation_only"
    raw_fragment_interpretation: str = "threshold_fragment"
    center_smoothing_enabled: bool = False
    invalid_fields: Tuple[str, ...] = field(default_factory=tuple)

    def to_effective_parameters(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "min_identity_bandwidth_hz": self.min_identity_bandwidth_hz,
            "min_persist_bandwidth_hz": self.min_persist_bandwidth_hz,
            "max_persist_bandwidth_hz": self.max_persist_bandwidth_hz,
            "min_match_bandwidth_hz": self.min_match_bandwidth_hz,
            "min_display_bandwidth_hz": self.min_display_bandwidth_hz,
            "allow_revisit_to_shrink_identity": self.allow_revisit_to_shrink_identity,
            "allow_revisit_to_move_center": self.allow_revisit_to_move_center,
            "min_revisit_bandwidth_for_identity_update_hz": (
                self.min_revisit_bandwidth_for_identity_update_hz
            ),
            "max_revisit_center_delta_for_identity_update_hz": (
                self.max_revisit_center_delta_for_identity_update_hz
            ),
            "fragmented_revisit_policy": self.fragmented_revisit_policy,
            "raw_fragment_interpretation": self.raw_fragment_interpretation,
            "center_smoothing_enabled": self.center_smoothing_enabled,
            "invalid_fields": list(self.invalid_fields),
        }


def resolve_signal_span_policy(args: Any) -> SignalSpanPolicy:
    invalid: list[str] = []
    profile_name = _get_attr(args, "_applied_profile", None) or _get_attr(args, "profile", None)
    min_match = _clean_positive_float(
        _get_attr(args, "min_match_bandwidth_hz", _get_attr(args, "min_emit_bandwidth_hz")),
        invalid=invalid,
        field_name="min_match_bandwidth_hz",
    )
    min_display = _clean_positive_float(
        _get_attr(args, "min_display_bandwidth_hz", _get_attr(args, "min_emit_bandwidth_hz")),
        invalid=invalid,
        field_name="min_display_bandwidth_hz",
    )
    min_identity = _clean_positive_float(
        _get_attr(args, "min_identity_bandwidth_hz"),
        invalid=invalid,
        field_name="min_identity_bandwidth_hz",
    )
    if min_identity is None and min_match is not None:
        min_identity = min_match
    elif min_identity is not None and min_match is not None:
        min_identity = max(min_identity, min_match)
    min_persist = _clean_positive_float(
        _get_attr(args, "min_persist_bandwidth_hz"),
        invalid=invalid,
        field_name="min_persist_bandwidth_hz",
    )
    if min_persist is None:
        min_persist = min_match if min_match is not None else min_identity

    max_persist = None
    for field_name in (
        "max_persist_bandwidth_hz",
        "max_persist_width_hz",
        "max_card_width_hz",
        "max_detection_width_hz",
    ):
        max_persist = _clean_positive_float(
            _get_attr(args, field_name),
            invalid=invalid,
            field_name=field_name,
        )
        if max_persist is not None:
            break
    if max_persist is not None and min_persist is not None and max_persist < min_persist:
        invalid.append("max_persist_bandwidth_hz_below_min_persist_bandwidth_hz")
        max_persist = min_persist

    min_revisit = _clean_positive_float(
        _get_attr(args, "min_revisit_bandwidth_for_identity_update_hz"),
        invalid=invalid,
        field_name="min_revisit_bandwidth_for_identity_update_hz",
    )
    if min_revisit is None and min_identity is not None:
        min_revisit = min_identity
    max_revisit_delta = _clean_positive_float(
        _get_attr(args, "max_revisit_center_delta_for_identity_update_hz"),
        invalid=invalid,
        field_name="max_revisit_center_delta_for_identity_update_hz",
    )
    if max_revisit_delta is None:
        max_revisit_delta = _clean_positive_float(
            _get_attr(args, "center_match_hz"),
            invalid=invalid,
            field_name="center_match_hz",
        )

    shrink = _clean_bool(_get_attr(args, "allow_revisit_to_shrink_identity"))
    if shrink is None:
        shrink = min_identity is None
    move_center = _clean_bool(_get_attr(args, "allow_revisit_to_move_center"))
    if move_center is None:
        move_center = True
    center_smoothing = _clean_bool(_get_attr(args, "center_smoothing_enabled"))
    if center_smoothing is None:
        center_smoothing = False

    return SignalSpanPolicy(
        profile_name=str(profile_name) if profile_name not in (None, "") else None,
        min_identity_bandwidth_hz=min_identity,
        min_persist_bandwidth_hz=min_persist,
        max_persist_bandwidth_hz=max_persist,
        min_match_bandwidth_hz=min_match,
        min_display_bandwidth_hz=min_display,
        allow_revisit_to_shrink_identity=bool(shrink),
        allow_revisit_to_move_center=bool(move_center),
        min_revisit_bandwidth_for_identity_update_hz=min_revisit,
        max_revisit_center_delta_for_identity_update_hz=max_revisit_delta,
        fragmented_revisit_policy=_clean_text(_get_attr(args, "fragmented_revisit_policy"), "confirmation_only"),
        raw_fragment_interpretation=_clean_text(_get_attr(args, "raw_fragment_interpretation"), "threshold_fragment"),
        center_smoothing_enabled=bool(center_smoothing),
        invalid_fields=tuple(dict.fromkeys(invalid)),
    )
