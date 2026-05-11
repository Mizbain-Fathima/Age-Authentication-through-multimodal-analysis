"""
Build data/master_manifest.csv from audio-data/*.csv (Common Voice format).
Columns: segment_id, speaker_id, age, path (path relative to project root).
Age from categorical (teens, twenties, ...) mapped to numeric. No client_id in CSV -> speaker_id="unknown".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DATA = PROJECT_ROOT / "audio-data"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "master_manifest.csv"

AUDIO_AGE_MAP = {
    "teens": 15,
    "twenties": 24,
    "thirties": 34,
    "fourties": 44,
    "fifties": 54,
    "sixties": 64,
    "seventies": 74,
    "eighties": 84,
    "nineties": 92,
}


def resolve_path(filename: str) -> Path:
    """CSV filename is e.g. cv-valid-train/sample-000000.mp3; actual path is audio-data/cv-valid-train/cv-valid-train/sample-000000.mp3."""
    parts = filename.replace("\\", "/").split("/")
    if len(parts) == 2:
        split, fname = parts
        return AUDIO_DATA / split / split / fname
    return AUDIO_DATA / filename


def main() -> int:
    if not AUDIO_DATA.exists():
        print("ERROR: audio-data/ not found", file=sys.stderr)
        return 1

    rows = []
    for csv_path in sorted(AUDIO_DATA.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "age" not in df.columns or "filename" not in df.columns:
            continue
        df = df[df["age"].notna() & (df["age"].astype(str).str.strip() != "")].copy()
        df["age_str"] = df["age"].str.strip().str.lower()
        df = df[df["age_str"].isin(AUDIO_AGE_MAP)]
        df["numeric_age"] = df["age_str"].map(AUDIO_AGE_MAP)
        for _, r in df.iterrows():
            path = resolve_path(r["filename"])
            if path.exists():
                rows.append(
                    {
                        "segment_id": "",
                        "speaker_id": "unknown",
                        "age": int(r["numeric_age"]),
                        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    }
                )

    if not rows:
        print("ERROR: No rows with valid age and existing file", file=sys.stderr)
        return 1

    df_out = pd.DataFrame(rows)
    df_out["segment_id"] = [f"seg_{i:08d}" for i in range(len(df_out))]
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"Wrote {len(df_out)} rows to {OUTPUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
