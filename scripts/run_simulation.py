"""
Dataset Generation Script (Windows-safe)
==========================================
Generates the 500-tube synthetic dataset and saves it to data/synthetic/.

Fixes the previous exit-code-4 issue:
  - Uses ASCII-only print statements (no Unicode characters)
  - Explicit UTF-8 encoding for all file I/O
  - Proper sys.stdout.reconfigure() for Windows console

Usage:
    cd "C:\\Users\\admin\\Desktop\\Master 2027"
    python scripts/run_simulation.py
    python scripts/run_simulation.py --tubes 50   # quick test run
"""

import sys
import os
import argparse
import yaml

# Fix Windows console encoding before any print
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import DegradationEngine


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CT-tube degradation dataset")
    parser.add_argument("--tubes", type=int, default=None,
                        help="Number of tubes (default: from config)")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config file path")
    parser.add_argument("--out", default="data/synthetic/degradation_data.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("CT-Scanner X-Ray Tube Simulation")
    print("Physics-Informed RUL Dataset Generator")
    print("=" * 60)

    config = load_config(args.config)
    n_tubes = args.tubes or config["simulation"]["num_tubes"]

    print(f"\nConfig  : {args.config}")
    print(f"Tubes   : {n_tubes}")
    print(f"Output  : {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    engine = DegradationEngine(config)

    print("\nRunning simulation...")
    dataset = engine.generate_dataset(num_tubes=n_tubes, save_path=args.out)

    print(f"\nDataset shape  : {dataset.shape}")
    print(f"Tubes          : {dataset['tube_id'].nunique()}")

    # Lifespan stats
    lifespans = (
        dataset.groupby("tube_id")["scan_cycle"]
        .max()
        .reset_index()
    )
    lifespans["lifespan_days"] = lifespans["scan_cycle"] / 300  # CHU Ibn Rochd regime
    print(f"\nLifespan stats (at 300 scans/day):")
    print(f"  Mean   : {lifespans['lifespan_days'].mean():.1f} days")
    print(f"  Median : {lifespans['lifespan_days'].median():.1f} days")
    print(f"  Std    : {lifespans['lifespan_days'].std():.1f} days")
    print(f"  Min    : {lifespans['lifespan_days'].min():.1f} days")
    print(f"  Max    : {lifespans['lifespan_days'].max():.1f} days")
    print(f"  Target : 395 days (CHU Ibn Rochd, Zakaria PFE 2023)")

    failure_modes = (
        dataset.drop_duplicates("tube_id")["failure_mode"]
        .value_counts()
    )
    print(f"\nFailure mode distribution:")
    for mode, count in failure_modes.items():
        pct = 100 * count / n_tubes
        print(f"  {mode:<25}: {count:>3} ({pct:.1f}%)")

    print(f"\nDataset saved to: {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
