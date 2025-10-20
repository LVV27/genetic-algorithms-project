# Genetic Algorithms Project

This project implements a Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP) using Python.

## Current Implementation

### Initialization
- Each individual in the population is represented as a random permutation of cities.
- Population size is controlled via `GA_PARAMS["POPULATION_SIZE"]`.

### Selection
- **k-Tournament Selection**: Randomly select `k` individuals and choose the best one based on fitness (shorter tour distance is better).
- Other options:
  - Roulette Wheel Selection
  - Rank-based Selection
  - Stochastic Universal Sampling

### Mutation
- **Swap Mutation**: Two random cities in a tour are swapped.
- Probability of mutation is determined by a self-adaptive `alpha` parameter.
- Other options:
  - Inversion Mutation
  - Scramble Mutation
  - Insert Mutation

### Recombination (Crossover)
- **Partially Mapped Crossover (PMX)**: Ensures offspring are valid tours.
- Other options:
  - Order Crossover (OX)
  - Cycle Crossover (CX)
  - Edge Recombination

### Elimination
- **(μ + λ) Elimination**: Combine parents and offspring, sort by fitness, and keep the best individuals.
- Other options:
  - (μ, λ) Elimination
  - Steady-State Replacement
  - Age-Based Replacement

---

## Parameters
- `POPULATION_SIZE`: Number of individuals in the population
- `OFFSPRING_SIZE`: Number of children generated each generation
- `GENERATIONS`: Maximum number of generations to run
- `TOURNAMENT_K`: Tournament size for selection
- `MUTATION_ALPHA_MIN` / `MUTATION_ALPHA_MAX`: Range for mutation probability
- `CROSSOVER_PROB`: Probability of performing crossover
