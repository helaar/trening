# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "pandas>=2.0.0",
#   "matplotlib>=3.7.0",
# ]
# ///
"""
Plot HRV, Pulse, and Sleep Hours from TrainingPeaks custom metrics exports.

Usage:
    python scripts/plot_metrics.py [--metrics-dir PATH]

Shows two views:
  1. Full timeline across all years
  2. Year-over-year comparison (day-of-year overlay)
"""

import argparse
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path(__file__).parent.parent.parent / "ENV" / "metrics"
METRICS_OF_INTEREST = ["HRV", "Pulse", "Sleep Hours"]
ROLLING_WINDOW = 7  # days for smoothing


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    files = sorted(metrics_dir.glob("metrics*.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics CSV files found in {metrics_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, quotechar='"')
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = raw.columns.str.strip()
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    raw["date"] = raw["Timestamp"].dt.date

    # Keep only metrics we care about, with simple numeric values
    raw = raw[raw["Type"].isin(METRICS_OF_INTEREST)].copy()
    raw["value"] = pd.to_numeric(raw["Value"], errors="coerce")
    raw = raw.dropna(subset=["value"])

    # Pivot so each metric is a column
    pivot = raw.pivot_table(index="date", columns="Type", values="value", aggfunc="mean")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot


def add_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rolled = df.copy()
    for col in df.columns:
        rolled[f"{col}_roll"] = df[col].rolling(window, center=True, min_periods=3).mean()
    return rolled


def plot_timeline(ax: plt.Axes, df: pd.DataFrame, metric: str, color: str) -> None:
    roll_col = f"{metric}_roll"
    ax.scatter(df.index, df[metric], alpha=0.2, s=8, color=color, zorder=2)
    if roll_col in df.columns:
        ax.plot(df.index, df[roll_col], color=color, linewidth=1.8, zorder=3,
                label=f"{metric} ({ROLLING_WINDOW}d avg)")
    ax.set_ylabel(metric, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_year_overlay(ax: plt.Axes, df: pd.DataFrame, metric: str, years: list[int]) -> None:
    colors = plt.cm.tab10.colors
    for i, year in enumerate(years):
        yearly = df[df.index.year == year][metric].dropna()
        if yearly.empty:
            continue
        doy = yearly.index.day_of_year
        rolled = yearly.rolling(ROLLING_WINDOW, center=True, min_periods=3).mean()
        ax.scatter(doy, yearly.values, alpha=0.15, s=6, color=colors[i % len(colors)])
        ax.plot(doy, rolled.values, color=colors[i % len(colors)], linewidth=1.8, label=str(year))

    ax.set_ylabel(metric, fontsize=9)
    ax.set_xlabel("Day of year", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, title="Year")

    # Month labels on x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TrainingPeaks custom metrics")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR,
                        help="Directory containing metrics CSV files")
    args = parser.parse_args()

    df = load_metrics(args.metrics_dir)
    df = add_rolling(df, ROLLING_WINDOW)

    available = [m for m in METRICS_OF_INTEREST if m in df.columns]
    if not available:
        print(f"None of {METRICS_OF_INTEREST} found in data. Available: {list(df.columns)}")
        return

    years = sorted(df.index.year.unique().tolist())
    n = len(available)

    fig = plt.figure(figsize=(14, 5 * n))
    fig.suptitle("Training Metrics Overview", fontsize=13, fontweight="bold", y=0.98)

    colors = ["steelblue", "tomato", "seagreen"]

    # --- Left column: full timeline ---
    for i, (metric, color) in enumerate(zip(available, colors)):
        ax = fig.add_subplot(n, 2, i * 2 + 1)
        plot_timeline(ax, df, metric, color)
        if i == 0:
            ax.set_title("Full timeline", fontsize=10, fontweight="bold")

    # --- Right column: year-over-year ---
    for i, metric in enumerate(available):
        ax = fig.add_subplot(n, 2, i * 2 + 2)
        plot_year_overlay(ax, df, metric, years)
        if i == 0:
            ax.set_title("Year-over-year comparison", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output = args.metrics_dir / "metrics.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")
    plt.show()


if __name__ == "__main__":
    main()
