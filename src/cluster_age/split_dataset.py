"""
Phase 1 — Speaker-independent dataset split for Cluster-Interpretable Binary Age Verification.

Splits manifest into:
  A (Train), B (Calibration), C (Threshold selection), D (Evaluation, optional).

Usage:
  python -m src.cluster_age.split_dataset --manifest path/to/master_manifest.csv --output-dir path/to/output
  python -m src.cluster_age.split_dataset --manifest path/to/master_manifest.csv --output-dir path/to/output --enable-d

Input manifest must have columns: segment_id, speaker_id, age, path.
Cluster is derived from age (C1 0-5, C2 6-10, C3 11-14, C4 15-17, C5 18+).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import numpy as np

import pandas as pd

from src.cluster_age.split_validation import (
    ADULT_CLUSTER,
    MINOR_CLUSTERS,
    add_cluster_column,
    get_cluster_stats,
    get_minor_adult_counts,
    validate_cluster_counts,
    validate_no_segment_overlap,
    validate_post_split_a,
    validate_post_split_b,
    validate_post_split_c,
    validate_speaker_disjoint,
)

# Reproducibility
SPLIT_SEED = 42

# Speaker assignment ratios (of unique speakers)
# Order: D (if enabled), then C, then B, rest -> A
RATIO_D = 0.10
RATIO_C = 0.15
RATIO_B = 0.15
# A gets the remainder (e.g. 0.60 or 0.70 if D disabled)

SCRIPT_VERSION = "1.0.0"


def load_manifest(path: Path) -> pd.DataFrame:
    """Load CSV manifest. Required columns: segment_id, speaker_id, age, path."""
    df = pd.read_csv(path)
    df["segment_id"] = df["segment_id"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(-1).astype(int)
    if (df["age"] < 0).any():
        print("ERROR: Manifest contains invalid (negative or non-numeric) age.", file=sys.stderr)
        sys.exit(1)
    return df


def assign_speakers(
    speaker_ids: List[str],
    enable_d: bool,
    seed: int,
) -> tuple[Set[str], Set[str], Set[str], Set[str] | None]:
    """
    Assign speakers to D, C, B, A in that order (shuffle with seed, then slice).
    Returns (speakers_a, speakers_b, speakers_c, speakers_d or None).
    """
    rng = np.random.default_rng(seed)
    shuffled = list(speaker_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    idx = 0

    if enable_d:
        n_d = max(1, int(n * RATIO_D))
        speakers_d = set(shuffled[idx : idx + n_d])
        idx += n_d
    else:
        speakers_d = None

    n_c = max(1, int(n * RATIO_C))
    speakers_c = set(shuffled[idx : idx + n_c])
    idx += n_c

    n_b = max(1, int(n * RATIO_B))
    speakers_b = set(shuffled[idx : idx + n_b])
    idx += n_b

    speakers_a = set(shuffled[idx:])
    return speakers_a, speakers_b, speakers_c, speakers_d


def segments_by_speakers(df: pd.DataFrame, speaker_ids: Set[str]) -> pd.DataFrame:
    """Return rows of df where speaker_id is in speaker_ids."""
    return df[df["speaker_id"].isin(speaker_ids)].copy()


def subsample_to_ratio(
    df: pd.DataFrame,
    max_adult_to_minor_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    If adult/minor > max_adult_to_minor_ratio, subsample adult segments so ratio <= max.
    Uses fixed seed for reproducibility.
    """
    if "cluster" not in df.columns:
        df = add_cluster_column(df)
    minor_df = df[df["cluster"].isin(MINOR_CLUSTERS)]
    adult_df = df[df["cluster"] == ADULT_CLUSTER]
    n_minor = len(minor_df)
    n_adult = len(adult_df)
    if n_minor == 0:
        return df
    current_ratio = n_adult / n_minor
    if current_ratio <= max_adult_to_minor_ratio:
        return df
    target_adult = int(n_minor * max_adult_to_minor_ratio)
    if target_adult >= n_adult:
        return df
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(adult_df.index, size=target_adult, replace=False)
    drop_adult_idx = adult_df.index.difference(keep_idx)
    return df.drop(index=drop_adult_idx)


def subsample_c_for_ratio(df_c: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Ensure C has adult:minor <= 4:1 by subsampling adults if needed."""
    return subsample_to_ratio(df_c, max_adult_to_minor_ratio=4.0, seed=seed + 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Speaker-independent split for cluster-age verification (A/B/C/D)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to master manifest CSV (columns: segment_id, speaker_id, age, path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write train/calibration/threshold/eval manifests and split_metadata.json.",
    )
    parser.add_argument(
        "--enable-d",
        action="store_true",
        help="Include dataset D (evaluation); otherwise only A, B, C.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help=f"Random seed for speaker assignment (default: {SPLIT_SEED}).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and add cluster
    df = load_manifest(args.manifest)
    df = add_cluster_column(df)

    # Pre-split validation (cluster minimums)
    validate_cluster_counts(df)

    # Unique speakers (deterministic order for assignment)
    speaker_ids = df["speaker_id"].unique().tolist()
    n_speakers = len(speaker_ids)

    # Assign speakers to D, C, B, A
    speakers_a, speakers_b, speakers_c, speakers_d = assign_speakers(
        speaker_ids, args.enable_d, args.seed
    )

    # Segment DataFrames per dataset
    df_a = segments_by_speakers(df, speakers_a)
    df_b = segments_by_speakers(df, speakers_b)
    df_c = segments_by_speakers(df, speakers_c)
    if speakers_d is not None:
        df_d = segments_by_speakers(df, speakers_d)
    else:
        df_d = None

    # Enforce ratios by subsampling
    df_a = subsample_to_ratio(df_a, max_adult_to_minor_ratio=5.0, seed=args.seed + 10)
    df_c = subsample_c_for_ratio(df_c, args.seed + 20)

    # Post-split validation
    minor_a, adult_a = get_minor_adult_counts(df_a)
    validate_post_split_a(minor_a, adult_a)

    minor_b, adult_b = get_minor_adult_counts(df_b)
    validate_post_split_b(minor_b, adult_b)

    minor_c, adult_c = get_minor_adult_counts(df_c)
    validate_post_split_c(minor_c, adult_c)

    validate_speaker_disjoint(
        speakers_a, speakers_b, speakers_c, speakers_d
    )
    validate_no_segment_overlap(
        set(df_a["segment_id"]),
        set(df_b["segment_id"]),
        set(df_c["segment_id"]),
        set(df_d["segment_id"]) if df_d is not None else None,
    )

    # Write manifests
    df_a.to_csv(args.output_dir / "train_manifest.csv", index=False)
    df_b.to_csv(args.output_dir / "calibration_manifest.csv", index=False)
    df_c.to_csv(args.output_dir / "threshold_manifest.csv", index=False)
    if df_d is not None:
        df_d.to_csv(args.output_dir / "eval_manifest.csv", index=False)

    # Build split_metadata.json
    metadata = {
        "script_version": SCRIPT_VERSION,
        "random_seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "enable_d": args.enable_d,
        "total_speakers": n_speakers,
        "total_segments": len(df),
        "datasets": {
            "A": {
                "speaker_count": len(speakers_a),
                "segment_count": len(df_a),
                "cluster_stats": get_cluster_stats(df_a),
                "minor_count": minor_a,
                "adult_count": adult_a,
                "adult_minor_ratio": round(adult_a / minor_a, 4) if minor_a else None,
            },
            "B": {
                "speaker_count": len(speakers_b),
                "segment_count": len(df_b),
                "cluster_stats": get_cluster_stats(df_b),
                "minor_count": minor_b,
                "adult_count": adult_b,
            },
            "C": {
                "speaker_count": len(speakers_c),
                "segment_count": len(df_c),
                "cluster_stats": get_cluster_stats(df_c),
                "minor_count": minor_c,
                "adult_count": adult_c,
                "adult_minor_ratio": round(adult_c / minor_c, 4) if minor_c else None,
            },
        },
        "output_files": [
            "train_manifest.csv",
            "calibration_manifest.csv",
            "threshold_manifest.csv",
        ],
    }
    if args.enable_d and df_d is not None and speakers_d is not None:
        minor_d, adult_d = get_minor_adult_counts(df_d)
        metadata["datasets"]["D"] = {
            "speaker_count": len(speakers_d),
            "segment_count": len(df_d),
            "cluster_stats": get_cluster_stats(df_d),
            "minor_count": minor_d,
            "adult_count": adult_d,
        }
        metadata["output_files"].append("eval_manifest.csv")

    with open(args.output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Phase 1 split complete.")
    print(f"  A: {len(df_a)} segments, {len(speakers_a)} speakers")
    print(f"  B: {len(df_b)} segments, {len(speakers_b)} speakers")
    print(f"  C: {len(df_c)} segments, {len(speakers_c)} speakers")
    if df_d is not None:
        print(f"  D: {len(df_d)} segments, {len(speakers_d)} speakers")
    print(f"  Metadata: {args.output_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
