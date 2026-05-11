"""
Phase 1 — Validation module for speaker-independent dataset split.
Validates cluster counts, speaker counts, ratio checks, and speaker overlap.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Set, Tuple

import pandas as pd

# Cluster definitions: age range (inclusive) -> cluster id
# C4 = 15-17, C5 = 18+ (binary: age < 18 -> minor, age >= 18 -> adult)
CLUSTER_AGE_RANGES = {
    "C1": (0, 5),
    "C2": (6, 10),
    "C3": (11, 14),
    "C4": (15, 17),
    "C5": (18, 120),
}

# Minimum required samples and speakers per cluster (pre-split)
MIN_SAMPLES = {"C1": 800, "C2": 800, "C3": 1000, "C4": 1200, "C5": 5000}
MIN_SPEAKERS = {"C1": 80, "C2": 80, "C3": 100, "C4": 120, "C5": 500}

MINOR_CLUSTERS = {"C1", "C2", "C3", "C4"}
ADULT_CLUSTER = "C5"


def age_to_cluster(age: int) -> str:
    """Map age to cluster. C1 0-5, C2 6-10, C3 11-14, C4 15-17, C5 18+."""
    for cluster, (lo, hi) in CLUSTER_AGE_RANGES.items():
        if lo <= age <= hi:
            return cluster
    return "C5"  # default adult


def age_to_binary(age: int) -> int:
    """Binary label for training: minor=1, adult=0. Age < 18 -> minor, age >= 18 -> adult."""
    return 1 if age < 18 else 0


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure manifest has required columns. Exit with error if not."""
    required = {"segment_id", "speaker_id", "age", "path"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Manifest missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)


def add_cluster_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'cluster' column from 'age'. Modifies copy."""
    df = df.copy()
    df["cluster"] = df["age"].astype(int).map(age_to_cluster)
    return df


def validate_cluster_counts(df: pd.DataFrame) -> None:
    """Validate minimum sample and speaker counts per cluster. Exit on failure."""
    validate_required_columns(df)
    if "cluster" not in df.columns:
        df = add_cluster_column(df)

    errors: List[str] = []
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        sub = df[df["cluster"] == cluster]
        n_samples = len(sub)
        n_speakers = sub["speaker_id"].nunique()
        if n_samples < MIN_SAMPLES[cluster]:
            errors.append(
                f"Cluster {cluster}: sample count {n_samples} < required {MIN_SAMPLES[cluster]}"
            )
        if n_speakers < MIN_SPEAKERS[cluster]:
            errors.append(
                f"Cluster {cluster}: speaker count {n_speakers} < required {MIN_SPEAKERS[cluster]}"
            )

    if errors:
        print("ERROR: Pre-split cluster validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def get_minor_adult_counts(df: pd.DataFrame) -> Tuple[int, int]:
    """Return (minor_count, adult_count). Minor = C1-C4, adult = C5."""
    if "cluster" not in df.columns:
        df = add_cluster_column(df)
    minor = df[df["cluster"].isin(MINOR_CLUSTERS)]
    adult = df[df["cluster"] == ADULT_CLUSTER]
    return len(minor), len(adult)


def validate_speaker_disjoint(
    speakers_a: Set[str],
    speakers_b: Set[str],
    speakers_c: Set[str],
    speakers_d: Set[str] | None,
) -> None:
    """Ensure no speaker appears in more than one dataset. Exit on failure."""
    sets_to_check = [speakers_a, speakers_b, speakers_c]
    names = ["A", "B", "C"]
    if speakers_d is not None:
        sets_to_check.append(speakers_d)
        names.append("D")

    for i in range(len(sets_to_check)):
        for j in range(i + 1, len(sets_to_check)):
            overlap = sets_to_check[i] & sets_to_check[j]
            if overlap:
                print(
                    f"ERROR: Speaker overlap between {names[i]} and {names[j]}: "
                    f"{len(overlap)} speakers",
                    file=sys.stderr,
                )
                sys.exit(1)


def validate_post_split_a(minor: int, adult: int) -> None:
    """A: adult:minor ratio <= 5:1."""
    if minor == 0:
        print("ERROR: Dataset A has no minor samples.", file=sys.stderr)
        sys.exit(1)
    ratio = adult / minor
    if ratio > 5.0:
        print(
            f"ERROR: Dataset A adult:minor ratio {ratio:.2f} > 5:1 (adult={adult}, minor={minor})",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_post_split_b(minor: int, adult: int) -> None:
    """B: minor >= 1500, adult >= 1500."""
    if minor < 1500:
        print(
            f"ERROR: Dataset B minor count {minor} < 1500",
            file=sys.stderr,
        )
        sys.exit(1)
    if adult < 1500:
        print(
            f"ERROR: Dataset B adult count {adult} < 1500",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_post_split_c(minor: int, adult: int) -> None:
    """C: adult >= 2000, minor >= 1000, adult:minor <= 4:1."""
    if adult < 2000:
        print(
            f"ERROR: Dataset C adult count {adult} < 2000",
            file=sys.stderr,
        )
        sys.exit(1)
    if minor < 1000:
        print(
            f"ERROR: Dataset C minor count {minor} < 1000",
            file=sys.stderr,
        )
        sys.exit(1)
    if minor == 0:
        print("ERROR: Dataset C has no minor samples.", file=sys.stderr)
        sys.exit(1)
    ratio = adult / minor
    if ratio > 4.0:
        print(
            f"ERROR: Dataset C adult:minor ratio {ratio:.2f} > 4:1 (adult={adult}, minor={minor})",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_no_segment_overlap(
    seg_a: Set[str],
    seg_b: Set[str],
    seg_c: Set[str],
    seg_d: Set[str] | None,
) -> None:
    """Ensure no segment_id appears in more than one dataset. Exit on failure."""
    all_segs = list(seg_a) + list(seg_b) + list(seg_c)
    if seg_d is not None:
        all_segs += list(seg_d)
    if len(all_segs) != len(set(all_segs)):
        print(
            "ERROR: At least one segment_id appears in more than one dataset.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_cluster_stats(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Return per-cluster sample and speaker counts."""
    if "cluster" not in df.columns:
        df = add_cluster_column(df)
    out: Dict[str, Dict[str, int]] = {}
    for c in ["C1", "C2", "C3", "C4", "C5"]:
        sub = df[df["cluster"] == c]
        out[c] = {"samples": len(sub), "speakers": sub["speaker_id"].nunique()}
    return out
