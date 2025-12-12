import os
from src import r0123456
from plot_convergence import plot_convergence, analyze_multiple_runs

# Paths
benchmark_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'benchmark', 'tour50.csv')
)
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def run_single_experiment():
    solver = r0123456.r0123456()
    csv_file = os.path.join(output_dir, "r0123456.csv")
    solver.reporter.filename = csv_file  # set filename first
    solver.optimize(benchmark_file)     # now optimize will save there
    plot_convergence(csv_file)

def run_multiple_experiments(runs=10):
    csv_files = []
    for i in range(runs):
        solver = r0123456.r0123456()
        # Save each run CSV in output folder (without .csv extension, Reporter adds it)
        run_csv = os.path.join(output_dir, f"r0123456_run_{i}")
        solver.reporter = r0123456.Reporter.Reporter(run_csv)
        solver.optimize(benchmark_file)
        csv_files.append(run_csv + ".csv")

    analyze_multiple_runs(csv_files)

if __name__ == "__main__":
    # Run a single experiment and plot convergence
    # run_single_experiment()

    # Run multiple experiments and analyze variability
    run_multiple_experiments(runs=100)
