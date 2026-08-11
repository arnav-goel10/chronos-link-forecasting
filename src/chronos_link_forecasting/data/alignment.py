from __future__ import annotations

import warnings

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def causal_asof_join(
    target: pd.DataFrame,
    feature: pd.DataFrame,
    *,
    timestamp: str,
    by: str,
    tolerance: str | pd.Timedelta,
) -> pd.DataFrame:
    """Join features available at or before each target timestamp."""
    _require_columns(target, {by, timestamp}, "target")
    _require_columns(feature, {by, timestamp}, "feature")
    _require_datetime_timestamp(target, timestamp, "target")
    _require_datetime_timestamp(feature, timestamp, "feature")

    if target[timestamp].dtype != feature[timestamp].dtype:
        raise ValueError("target and feature timestamps must use the same dtype")
    if target.duplicated(subset=[by, timestamp]).any():
        raise ValueError("duplicate target key/timestamp pairs are not allowed")

    feature_names = [name for name in feature.columns if name not in {by, timestamp}]
    if not feature_names:
        raise ValueError("feature must include at least one value column")

    conflicting_names = set(feature_names).intersection(target.columns)
    audit_names = {f"{name}__observed_at" for name in feature_names}
    conflicting_names.update(audit_names.intersection(target.columns))
    conflicting_names.update(audit_names.intersection(feature.columns))
    if conflicting_names:
        conflicts = ", ".join(sorted(str(name) for name in conflicting_names))
        raise ValueError(
            f"feature output columns conflict with input schema: {conflicts}"
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The 'generic' unit for NumPy timedelta is deprecated",
                category=DeprecationWarning,
            )
            parsed_tolerance = pd.Timedelta(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be a positive duration") from error
    if parsed_tolerance <= pd.Timedelta(0):
        raise ValueError("tolerance must be a positive duration")

    target_copy = target.copy(deep=True).sort_values([timestamp, by], kind="mergesort")
    feature_copy = feature.copy(deep=True)
    for feature_name in feature_names:
        feature_copy[f"{feature_name}__observed_at"] = feature_copy[timestamp]
    feature_copy = feature_copy.sort_values([timestamp, by], kind="mergesort")

    joined = pd.merge_asof(
        target_copy,
        feature_copy,
        on=timestamp,
        by=by,
        direction="backward",
        allow_exact_matches=True,
        tolerance=parsed_tolerance,
    ).reset_index(drop=True)

    for feature_name in feature_names:
        audit_name = f"{feature_name}__observed_at"
        observed = joined[audit_name].notna()
        if (joined.loc[observed, audit_name] > joined.loc[observed, timestamp]).any():
            raise ValueError("causal alignment produced a future feature timestamp")

    return joined


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{frame_name} must include columns: {columns}")


def _require_datetime_timestamp(
    frame: pd.DataFrame, timestamp: str, frame_name: str
) -> None:
    if not is_datetime64_any_dtype(frame[timestamp].dtype):
        raise ValueError(f"{frame_name} timestamp must be datetime-like")
    if frame[timestamp].isna().any():
        raise ValueError(f"{frame_name} timestamp contains missing values")
