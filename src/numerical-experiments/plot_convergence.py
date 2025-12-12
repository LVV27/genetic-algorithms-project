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


def plot_convergence(csv_file: str):
    """
    Plot convergence curves for mean and best fitness values.

    Args:
        csv_file: Path to the CSV file containing optimization data
    """
    df = read_optimization_csv(csv_file)

    # Create convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["Iteration"],
        df["Mean value"],
        label="Mean value",
        color="blue",
        linewidth=1.5,
    )
    plt.plot(
        df["Iteration"],
        df["Best value"],
        label="Best value",
        color="red",
        linewidth=1.5,
    )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Objective Value", fontsize=12)
    plt.title("GA Convergence Over Iterations", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
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
            print(f"Warning: Skipping {file} due to error: {str(e)}")
            continue

    if not final_means:
        print("Error: No valid data found in any CSV files")
        return

    # Create histogram plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mean fitness histogram
    ax1.hist(final_means, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_title("Final Mean Fitness over Runs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mean Fitness", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # Best fitness histogram
    ax2.hist(final_bests, bins=10, color="salmon", edgecolor="black", alpha=0.7)
    ax2.set_title("Final Best Fitness over Runs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Best Fitness", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    # Print statistical summary
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    print(f"Number of successful runs: {len(final_means)}")
    print("\nFinal Mean Fitness:")
    print(f"  Mean: {np.mean(final_means):.4f}")
    print(f"  Std:  {np.std(final_means):.4f}")
    print(f"  Min:  {np.min(final_means):.4f}")
    print(f"  Max:  {np.max(final_means):.4f}")
    print("\nFinal Best Fitness:")
    print(f"  Mean: {np.mean(final_bests):.4f}")
    print(f"  Std:  {np.std(final_bests):.4f}")
    print(f"  Min:  {np.min(final_bests):.4f}")
    print(f"  Max:  {np.max(final_bests):.4f}")
    print("=" * 50 + "\n")
