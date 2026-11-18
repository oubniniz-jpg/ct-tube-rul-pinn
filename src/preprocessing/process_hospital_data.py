"""
Preprocess CHU Ibn Rochd hospital failure data.
Computes derived features: tube lifespan, time between failures,
failure mode encoding, and operational insights.
"""

import pandas as pd
import numpy as np


def load_and_process_failures(raw_path: str = "data/raw/chu_ibn_rochd_tube_failures.csv") -> pd.DataFrame:
    """Load raw failure data and compute derived features."""
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Tube lifespan (days between consecutive replacements)
    df["days_since_previous"] = df["date"].diff().dt.days
    df["tube_lifespan_days"] = df["days_since_previous"]

    # Assign tube ID (each row = one tube's end of life)
    df["tube_id"] = range(len(df))

    # Estimated scans (assuming ~80 scans/day based on high workload)
    scans_per_day = 80
    df["estimated_total_scans"] = df["tube_lifespan_days"] * scans_per_day

    # Failure mode encoding
    failure_map = {
        "filament_rupture": 0,
        "cracked_glass": 1,
        "overheated_track": 2,
        "anode_stem_rupture": 3,
    }
    df["failure_code"] = df["failure_cause_en"].map(failure_map)

    # Year and season
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the failure dataset."""
    valid = df.dropna(subset=["tube_lifespan_days"])
    stats = {
        "total_replacements": len(df),
        "observation_period_days": (df["date"].max() - df["date"].min()).days,
        "mean_lifespan_days": valid["tube_lifespan_days"].mean(),
        "std_lifespan_days": valid["tube_lifespan_days"].std(),
        "min_lifespan_days": valid["tube_lifespan_days"].min(),
        "max_lifespan_days": valid["tube_lifespan_days"].max(),
        "median_lifespan_days": valid["tube_lifespan_days"].median(),
        "failure_distribution": df["failure_cause_en"].value_counts().to_dict(),
    }
    return stats


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/hospital_failures_processed.csv"):
    """Save processed data."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    df = load_and_process_failures()
    stats = compute_statistics(df)

    print("=" * 50)
    print("CHU Ibn Rochd - Tube Failure Analysis")
    print("=" * 50)
    print(f"Total replacements: {stats['total_replacements']}")
    print(f"Observation period: {stats['observation_period_days']} days ({stats['observation_period_days']/365:.1f} years)")
    print(f"Mean tube lifespan: {stats['mean_lifespan_days']:.0f} days ({stats['mean_lifespan_days']/365:.1f} years)")
    print(f"Std deviation: {stats['std_lifespan_days']:.0f} days")
    print(f"Range: {stats['min_lifespan_days']:.0f} - {stats['max_lifespan_days']:.0f} days")
    print(f"\nFailure distribution:")
    for cause, count in stats["failure_distribution"].items():
        print(f"  {cause}: {count}")

    save_processed_data(df)
