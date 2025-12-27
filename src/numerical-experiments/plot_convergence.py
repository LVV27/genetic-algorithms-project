import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def read_optimization_csv(csv_file: str) -> pd.DataFrame:
    """
    Read and parse optimization results from CSV file.

    Args:
        csv_file: Path to the CSV file containing optimization data

    Returns:
        DataFrame with parsed numeric columns

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or malformed
    """
    try:
        # Read CSV skipping comment lines
        df = pd.read_csv(csv_file, comment="#", header=None, usecols=[0, 1, 2, 3, 4])

        # Assign proper column names
        df.columns = ["Iteration", "Elapsed time", "Mean value", "Best value", "Cycle"]

        # Ensure numeric columns are properly typed
        numeric_cols = ["Iteration", "Elapsed time", "Mean value", "Best value"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop any rows that couldn't be converted to numeric
        df = df.dropna(subset=numeric_cols)

        if df.empty:
            raise ValueError(f"No valid data found in {csv_file}")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_file}: {str(e)}")


import matplotlib.pyplot as plt

def plot_convergence(csv_file: str):
    df = read_optimization_csv(csv_file)

    fig, ax_left = plt.subplots(figsize=(12, 7))

    # Left axis: Mean fitness (blue)
    ax_left.plot(
        df["Iteration"],
        df["Mean value"],
        label="Mean Fitness",
        color="#3498db",
        linewidth=2,
        alpha=0.8,
    )
    ax_left.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax_left.set_ylabel("Mean Fitness (Tour Length)", fontsize=13, fontweight="bold")
    ax_left.grid(True, alpha=0.25, linestyle="--")

    # Right axis: Best fitness (red)
    ax_right = ax_left.twinx()
    ax_right.plot(
        df["Iteration"],
        df["Best value"],
        label="Best Fitness",
        color="#e74c3c",
        linewidth=2.5,
    )
    ax_right.set_ylabel("Best Fitness (Tour Length)", fontsize=13, fontweight="bold")

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        fontsize=11,
        framealpha=0.95,
        loc="best",
    )

    plt.tight_layout()
    plt.show()



def analyze_multiple_runs(csv_files: List[str]):
    """
    Analyze results from multiple optimization runs and display statistics.

    Args:
        csv_files: List of paths to CSV files from different runs
    """
    final_means = []
    final_bests = []

    # Extract final values from each run
    for file in csv_files:
        try:
            df = read_optimization_csv(file)
            final_means.append(df["Mean value"].iloc[-1])
            final_bests.append(df["Best value"].iloc[-1])
        except Exception as e:
            print(f"⚠️  Warning: Skipping {file} - {str(e)}")
            continue

    if not final_means:
        print("❌ Error: No valid data found in any CSV files")
        return

    # Create histogram plots with modern styling
    fig = plt.figure(figsize=(14, 6))

    # Mean fitness histogram
    ax1 = plt.subplot(1, 2, 1)
    n, bins, patches = ax1.hist(
        final_means,
        bins=100,
        color="#3498db",
        edgecolor="#2c3e50",
        alpha=0.8,
        linewidth=1.5,
    )

    # Color gradient for bars
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Blues(0.4 + 0.6 * i / len(patches)))

    ax1.axvline(
        np.mean(final_means),
        color="#e74c3c",
        linestyle="--",
        linewidth=2.5,
        label=f"Mean: {np.mean(final_means):.2f}",
    )
    # ax1.set_title(
    #     "Final Mean Fitness Distribution", fontsize=13, fontweight="bold", pad=15
    # )
    ax1.set_xlabel("Mean Fitness", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25, axis="y", linestyle="--")

    # Best fitness histogram
    ax2 = plt.subplot(1, 2, 2)
    n, bins, patches = ax2.hist(
        final_bests,
        bins=100,
        color="#2ecc71",
        edgecolor="#27ae60",
        alpha=0.8,
        linewidth=1.5,
    )

    # Color gradient for bars
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Greens(0.4 + 0.6 * i / len(patches)))

    ax2.axvline(
        np.mean(final_bests),
        color="#e74c3c",
        linestyle="--",
        linewidth=2.5,
        label=f"Mean: {np.mean(final_bests):.2f}",
    )
    # ax2.set_title(
    #     "Final Best Fitness Distribution", fontsize=13, fontweight="bold", pad=15
    # )
    ax2.set_xlabel("Best Fitness", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25, axis="y", linestyle="--")

    plt.tight_layout()
    plt.show()

    # Adjusted box width and column widths
    box_width = 74  # wider to accommodate decimals
    col1_width = 30
    col2_width = 20
    col3_width = 20

    print("\n┌" + "─" * box_width + "─┐")
    print("│" + " STATISTICAL SUMMARY".center(box_width) + " │")
    print("├" + "─" * box_width + "─┤")
    print(f"│  {'Metric':<{col1_width}} {'Mean Fitness':>{col2_width}} {'Best Fitness':>{col3_width}} │")
    print("├" + "─" * box_width + "─┤")

    metrics = [
        ("Runs", len(final_means), len(final_bests)),
        ("Mean", np.mean(final_means), np.mean(final_bests)),
        ("Std Dev", np.std(final_means), np.std(final_bests)),
        ("Min", np.min(final_means), np.min(final_bests)),
        ("Max", np.max(final_means), np.max(final_bests)),
        ("Median", np.median(final_means), np.median(final_bests)),
    ]

    for metric_name, mean_val, best_val in metrics:
        if metric_name == "Runs":
            print(f"│  {metric_name:<{col1_width}} {mean_val:>{col2_width}} {best_val:>{col3_width}} │")
        else:
            print(f"│  {metric_name:<{col1_width}} {mean_val:>{col2_width}.2f} {best_val:>{col3_width}.2f} │")

    print("└" + "─" * box_width + "─┘\n")

    # Additional insights
    improvement = (
        (np.mean(final_means) - np.mean(final_bests)) / np.mean(final_means)
    ) * 100
    best_run = np.argmin(final_bests)

    print("📊 Key Insights:")
    print(
        f"   • Best solution found in run #{best_run + 1}: {final_bests[best_run]:.2f}"
    )
    print(f"   • Average improvement (mean→best): {improvement:.2f}%")
    print(
        f"   • Coefficient of variation: {(np.std(final_bests) / np.mean(final_bests) * 100):.2f}%"
    )
    print()

import numpy as np

def extract_run_metrics(csv_file: str, eps: float = 1e-9) -> None:
    df = read_optimization_csv(csv_file)

    it = df["Iteration"].to_numpy()
    t  = df["Elapsed time"].to_numpy()
    mean = df["Mean value"].to_numpy()
    best = df["Best value"].to_numpy()

    # Best-so-far (monotone non-increasing)
    best_so_far = np.minimum.accumulate(best)

    final_best = float(best_so_far[-1])
    final_mean = float(mean[-1])

    # Gap mean vs best (final)
    gap = final_mean - final_best

    # Improvement amount (from first logged best to final best)
    initial_best = float(best_so_far[0])
    improvement = initial_best - final_best

    # Iteration/time of best solution (first time reaching final best)
    plateau_idx = int(np.argmax(best_so_far <= final_best + eps))  # first hit of final best
    plateau_iter = int(it[plateau_idx])
    plateau_time = float(t[plateau_idx])

    # Time to best solution (same as reaching final best in this logging setup)
    time_to_best = plateau_time

    # Iteration of last improvement (last strict decrease in best-so-far)
    drops = np.where(np.diff(best_so_far) < -eps)[0]
    if drops.size == 0:
        last_impr_iter = int(it[0])
        last_impr_time = float(t[0])
    else:
        last_idx = int(drops[-1] + 1)  # +1 because diff refers to transition into this index
        last_impr_iter = int(it[last_idx])
        last_impr_time = float(t[last_idx])

    # Print only what you asked for
    print(f"iteration of last improvement: {last_impr_iter}")
    print(f"time of last improvement (s): {last_impr_time:.2f}")
    print(f"time to best solution (s): {time_to_best:.2f}")
    print(f"time to reach plateau (s): {plateau_time:.2f}  (gen {plateau_iter})")
    print(f"gap mean - best (final): {gap:.2f}")
    print(f"final best value: {final_best:.2f}")
    print(f"improvement amount: {improvement:.2f}")
