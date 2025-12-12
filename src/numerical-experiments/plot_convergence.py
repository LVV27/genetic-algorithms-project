import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(csv_file):
    # Read CSV skipping comment lines
    df = pd.read_csv(
        csv_file,
        comment='#',
        header=None,
        usecols=[0, 1, 2, 3, 4]
    )

    # Assign proper column names
    df.columns = ['Iteration', 'Elapsed time', 'Mean value', 'Best value', 'Cycle']

    # Ensure numeric columns
    numeric_cols = ['Iteration', 'Elapsed time', 'Mean value', 'Best value']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that couldn't be converted to numeric
    df = df.dropna(subset=numeric_cols)

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['Mean value'], label='Mean value', color='blue')
    plt.plot(df['Iteration'], df['Best value'], label='Best value', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('GA Convergence Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_multiple_runs(csv_files):
    final_means, final_bests = [], []

    for file in csv_files:
        df = pd.read_csv(file, comment='#', header=None, usecols=[0, 1, 2, 3, 4])
        df.columns = ['Iteration', 'Elapsed time', 'Mean value', 'Best value', 'Cycle']
        numeric_cols = ['Mean value', 'Best value']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)
        final_means.append(df['Mean value'].iloc[-1])
        final_bests.append(df['Best value'].iloc[-1])

    # Histogram plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(final_means, bins=10, color='skyblue', edgecolor='black')
    plt.title('Final Mean Fitness over Runs')
    plt.xlabel('Mean Fitness')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(final_bests, bins=10, color='salmon', edgecolor='black')
    plt.title('Final Best Fitness over Runs')
    plt.xlabel('Best Fitness')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print("Mean of final means:", np.mean(final_means))
    print("Std of final means:", np.std(final_means))
    print("Mean of final bests:", np.mean(final_bests))
    print("Std of final bests:", np.std(final_bests))
