import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import r0843621
from plot_convergence import plot_convergence, analyze_multiple_runs, extract_run_metrics

# Configuration
BENCHMARK_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "benchmark", "tour50.csv")
)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/"))

def setup_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_experiment_header(run_num: int, total_runs: int):
    """Print a clean experiment header."""
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT {run_num}/{total_runs}")
    print(f"{'═' * 70}")


def run_single_experiment():
    """
    Run a single optimization experiment and plot convergence.

    Saves results to 'output/r0123456.csv' and displays convergence plot.
    """
    print("\n" + "═" * 70)
    print("  SINGLE EXPERIMENT MODE")
    print("═" * 70)

    print(OUTPUT_DIR)
    #solver = r0843621.r0843621()
    csv_file = os.path.join(OUTPUT_DIR, "r0843621_1000.csv")

    # Configure reporter and run optimization
    #solver.reporter.filename = csv_file
    #solver.optimize(BENCHMARK_FILE)

    extract_run_metrics(csv_file)
    # Visualize results
    plot_convergence(csv_file)


def run_multiple_experiments(runs=10):
    """
    Run multiple optimization experiments and analyze variability.

    Args:
        runs: Number of independent runs to execute (default: 10)

    Saves individual run results and displays statistical analysis.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print(
        f"║  MULTIPLE EXPERIMENTS MODE - {runs} RUNS{' ' * (68 - 35 - len(str(runs)))}║"
    )
    print("╚" + "═" * 68 + "╝")

    csv_files = []

    for i in range(runs):
        print_experiment_header(i + 1, runs)

        # solver = r0843621.r0843621()

        # Configure output for this run
        run_csv = os.path.join(OUTPUT_DIR, f"r0123456_run_{i}")
        # solver.reporter = r0843621.Reporter.Reporter(run_csv)

        # Run optimization
        # solver.optimize(BENCHMARK_FILE)

        # Store path for analysis (Reporter adds .csv extension)
        csv_files.append(run_csv + ".csv")

    # Final analysis
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  STATISTICAL ANALYSIS" + " " * 46 + "║")
    print("╚" + "═" * 68 + "╝\n")

    analyze_multiple_runs(csv_files)


def main():
    """Main entry point for experiment execution."""
    setup_output_directory()

    # Choose experiment type
    run_single_experiment()
    # run_multiple_experiments(runs=500)


if __name__ == "__main__":
    main()
