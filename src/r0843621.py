import Reporter
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numba


# ==============================================================
# LOGGING CONFIGURATION
# ==============================================================


class ProfessionalFormatter(logging.Formatter):
    """Custom log formatter with severity-specific styling."""

    FORMATS = {
        logging.INFO: "%(message)s",
        logging.WARNING: "⚠️  %(message)s",
        logging.ERROR: "❌ %(message)s",
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.FORMATS.get(record.levelno, "%(message)s")
        return logging.Formatter(fmt).format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

_handler = logging.StreamHandler()
_handler.setFormatter(ProfessionalFormatter())
logger.addHandler(_handler)
logger.propagate = False


# ==============================================================
# ALGORITHM PARAMETERS
# ==============================================================


@dataclass(frozen=True)
class GAParams:
    """Centralized configuration for the Genetic Algorithm."""

    # Population Parameters
    POPULATION_SIZE: int = 100  # λ
    OFFSPRING_SIZE: int = 100  # μ
    GENERATIONS: int = 10_000_000

    # Selection Parameters
    TOURNAMENT_K: int = 5  # Tournament size (larger = more selective)

    # Mutation Parameters
    MUTATION_ALPHA_MIN: float = 0.08
    MUTATION_ALPHA_MAX: float = 0.20

    # Crossover Parameters
    CROSSOVER_PROB: float = 0.8

    # Initialization Parameters
    GREEDY_FRACTION: float = 0.20  # Fraction of population initialized with greedy
    GREEDY_ATTEMPTS_MULTIPLIER: int = 20  # Try this many greedy starts per target slot

    # Logging Parameters
    LOG_INTERVAL: int = 100

    # Penalty for missing edges (Inf)
    USE_PENALTY_FOR_INF: bool = True
    INF_PENALTY_FACTOR: float = 10

    # Candidate list (0 to disable)
    CANDIDATE_LIST_SIZE: int = 0


GA = GAParams()


# ==============================================================
# CONSOLE OUTPUT HELPERS
# ==============================================================


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    logger.info("")
    logger.info("─" * width)
    logger.info(f" {title}")
    logger.info("─" * width)


def print_stats_table(stats: dict) -> None:
    """Print statistics with aligned columns."""
    for k, v in stats.items():
        if isinstance(v, float):
            logger.info(f"  {k:<30} {v:>12.2f}")
        else:
            logger.info(f"  {k:<30} {v:>12}")


# ==============================================================
# PROBLEM DEFINITION
# ==============================================================


class TravelingSalesmanProblem:
    """
    TSP instance wrapper with distance matrix and optional edge penalties.
    """

    HEURISTIC_VALUES = {
        "tour50.csv": 15665,
        "tour250.csv": 87874,
        "tour500.csv": 119458,
        "tour750.csv": 140149,
        "tour1000.csv": 70468,
    }

    def __init__(self, distance_matrix: np.ndarray, filename: Optional[str] = None):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.filename = filename

        # For sparse graphs: track which edges are valid
        self.feasible = np.isfinite(distance_matrix)

        # Penalized matrix: replace Inf edges with large penalty
        self.penalized_matrix = self._build_penalized_matrix(distance_matrix)

    def _build_penalized_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Build distance matrix with penalties for missing edges."""
        penalized = distance_matrix.copy()

        if not GA.USE_PENALTY_FOR_INF or not np.isinf(distance_matrix).any():
            return penalized

        # Calculate penalty value
        finite_mask = np.isfinite(distance_matrix)
        max_finite = (
            float(np.max(distance_matrix[finite_mask])) if finite_mask.any() else 1.0
        )
        penalty = (max_finite + 1.0) * GA.INF_PENALTY_FACTOR

        # Apply candidate list if configured
        if GA.CANDIDATE_LIST_SIZE > 0:
            n = self.num_cities
            candidate_mask = np.zeros((n, n), dtype=bool)

            for i in range(n):
                candidate_mask[i, i] = True
                row = distance_matrix[i]
                finite_indices = np.where(np.isfinite(row))[0]
                finite_indices = finite_indices[finite_indices != i]

                if GA.CANDIDATE_LIST_SIZE < len(finite_indices):
                    partition_indices = np.argpartition(
                        row[finite_indices], GA.CANDIDATE_LIST_SIZE
                    )[: GA.CANDIDATE_LIST_SIZE]
                    nearest = finite_indices[partition_indices]
                else:
                    nearest = finite_indices

                candidate_mask[i, nearest] = True

            penalized[~candidate_mask] = penalty
            penalized[~np.isfinite(penalized)] = penalty
        else:
            # No candidate list: penalize all missing edges
            penalized[~finite_mask] = penalty

        return penalized

    def get_distance(self, city_a: int, city_b: int) -> float:
        """Return distance between two cities."""
        return self.distance_matrix[city_a, city_b]

    def print_info(self) -> None:
        """Print instance metadata."""
        print_section("PROBLEM INSTANCE")

        stats = {
            "Instance": self.filename or "Unknown",
            "Cities": self.num_cities,
        }

        heuristic_value = self.HEURISTIC_VALUES.get(self.filename)
        if heuristic_value is not None:
            stats["Known Heuristic"] = float(heuristic_value)

        print_stats_table(stats)


# ==============================================================
# INDIVIDUAL (Solution Representation)
# ==============================================================


class Individual:
    """
    A candidate TSP solution.

    Attributes:
        tour: Permutation of city indices
        mutation_rate: Self-adaptive parameter
        fitness: Total tour length (lower is better)
    """

    def __init__(
        self,
        problem: Optional[TravelingSalesmanProblem] = None,
        tour: Optional[np.ndarray] = None,
        mutation_rate: Optional[float] = None,
    ):
        if tour is not None:
            self.tour = np.asarray(tour, dtype=int)
        elif problem is not None:
            self.tour = np.random.permutation(problem.num_cities)
        else:
            raise ValueError("Either `problem` or `tour` must be provided.")

        if mutation_rate is None:
            self.mutation_rate = random.uniform(
                GA.MUTATION_ALPHA_MIN, GA.MUTATION_ALPHA_MAX
            )
        else:
            self.mutation_rate = float(mutation_rate)

        self.fitness: Optional[float] = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        """Evaluate tour fitness using penalized distance matrix."""
        self.fitness = evaluate_tour_numba(self.tour, problem.penalized_matrix)
        return self.fitness


@numba.njit(cache=True)
def evaluate_tour_numba(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute total tour length efficiently using Numba.
    Returns np.inf if any edge is invalid.
    """
    total = 0.0
    n = tour.shape[0]

    for i in range(n):
        city_a = tour[i]
        city_b = tour[(i + 1) % n]
        distance = distance_matrix[city_a, city_b]

        if np.isinf(distance):
            return np.inf

        total += distance

    return total


# ==============================================================
# DIVERSITY MEASUREMENT
# ==============================================================


def edge_diversity(population: List[Individual]) -> float:
    if not population:
        return 0.0

    tours = [ind.tour for ind in population]
    pop_size = len(tours)
    n = len(tours[0])

    edges = set()

    for t in tours:
        for i in range(n):
            edges.add((t[i], t[(i + 1) % n]))

    denom = min(pop_size * n, n * (n - 1))
    return len(edges) / denom


# ==============================================================
# POPULATION INITIALIZATION
# ==============================================================


def nearest_neighbor_greedy(
    problem: TravelingSalesmanProblem,
    start_city: int = 0,
) -> Optional[np.ndarray]:
    """
    Nearest-neighbor heuristic (Numba-accelerated).
    Returns None if construction fails (sparse graphs).
    """
    tour = nearest_neighbor_greedy_numba(
        problem.distance_matrix,
        start_city,
    )

    if tour[0] == -1:
        return None

    return tour


@numba.njit(cache=True)
def nearest_neighbor_greedy_numba(
    distance_matrix: np.ndarray,
    start_city: int,
) -> np.ndarray:
    """
    Numba-accelerated nearest neighbor construction.

    Returns:
        Tour as array, or array filled with -1 if construction fails
    """
    num_cities = distance_matrix.shape[0]
    tour = np.empty(num_cities, dtype=np.int32)
    tour[0] = start_city

    visited = np.zeros(num_cities, dtype=np.bool_)
    visited[start_city] = True

    current_city = start_city

    for step in range(1, num_cities):
        best_distance = np.inf
        best_next_city = -1

        for candidate in range(num_cities):
            if visited[candidate]:
                continue

            distance = distance_matrix[current_city, candidate]
            if distance < best_distance:
                best_distance = distance
                best_next_city = candidate

        # Construction failed
        if best_next_city == -1 or np.isinf(best_distance):
            tour[:] = -1
            return tour

        tour[step] = best_next_city
        visited[best_next_city] = True
        current_city = best_next_city

    return tour


def initialize_population_mixed(
    problem: TravelingSalesmanProblem,
    population_size: int,
) -> List[Individual]:
    """
    Initialize population with mix of greedy and random solutions.

    Strategy:
        - GA.GREEDY_FRACTION greedy nearest-neighbor tours
        - Remainder random permutations
    """
    population: List[Individual] = []

    num_greedy_target = int(population_size * GA.GREEDY_FRACTION)
    num_random_target = population_size - num_greedy_target

    logger.info(f"Generating {num_greedy_target} greedy solutions...")

    max_greedy_attempts = num_greedy_target * GA.GREEDY_ATTEMPTS_MULTIPLIER

    start_cities = random.sample(
        range(problem.num_cities),
        min(problem.num_cities, max_greedy_attempts),
    )

    greedy_candidates: List[Individual] = []

    for start_city in start_cities:
        tour = nearest_neighbor_greedy(problem, start_city)
        if tour is None:
            continue

        ind = Individual(tour=tour)
        ind.evaluate(problem)

        if np.isfinite(ind.fitness):
            greedy_candidates.append(ind)

    greedy_candidates.sort(key=lambda ind: ind.fitness)
    greedy_selected = greedy_candidates[:num_greedy_target]

    population.extend(greedy_selected)
    greedy_count = len(greedy_selected)

    logger.info(f"Generating {num_random_target} random solutions...")
    random_count = 0

    while len(population) < population_size:
        ind = Individual(problem=problem)
        ind.evaluate(problem)

        if np.isfinite(ind.fitness):
            population.append(ind)
            random_count += 1

    logger.info(
        f"Initialized {len(population)} individuals "
        f"({greedy_count} greedy, {random_count} random)"
    )

    return population


# ==============================================================
# SELECTION
# ==============================================================


def tournament_selection(
    population: List[Individual],
    tournament_size: int,
) -> Individual:
    """
    Tournament selection: sample K individuals and return the best.
    Larger tournament size increases selection pressure.
    """
    competitors = random.sample(population, tournament_size)
    return min(competitors, key=lambda ind: ind.fitness)


# ==============================================================
# MUTATION OPERATORS
# ==============================================================


def mutation_swap(tour: np.ndarray) -> None:
    """Swap two random cities in-place."""
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]


def mutation_inversion(tour: np.ndarray) -> None:
    """Reverse a random contiguous segment in-place."""
    start, end = sorted(random.sample(range(len(tour)), 2))
    tour[start : end + 1] = tour[start : end + 1][::-1]


def mutation_insertion(individual: Individual) -> None:
    """Remove one city and insert it at another position."""
    tour = individual.tour
    n = len(tour)

    source, target = random.sample(range(n), 2)
    city = tour[source]

    if source < target:
        individual.tour = np.concatenate(
            [tour[:source], tour[source + 1 : target + 1], [city], tour[target + 1 :]]
        )
    else:
        individual.tour = np.concatenate(
            [tour[:target], [city], tour[target:source], tour[source + 1 :]]
        )


def mutation(individual: Individual) -> None:
    """
    Apply mutation with probability = individual.mutation_rate.
    Operator selection weighted toward inversion (good for TSP).
    """
    if random.random() >= individual.mutation_rate:
        return

    operators = ["swap", "inversion", "insertion"]
    weights = [0.25, 0.55, 0.20]

    choice = random.choices(operators, weights=weights, k=1)[0]

    if choice == "swap":
        mutation_swap(individual.tour)
    elif choice == "inversion":
        mutation_inversion(individual.tour)
    else:
        mutation_insertion(individual)


# ==============================================================
# CROSSOVER OPERATORS
# ==============================================================


def order_crossover(
    problem: TravelingSalesmanProblem,
    parent_a: Individual,
    parent_b: Individual,
) -> Individual:
    """
    Order Crossover (OX): preserve relative city order from parents.

    Strategy:
        1. Copy contiguous segment from parent_a
        2. Fill remaining positions using parent_b's city order
    """
    num_cities = problem.num_cities
    start, end = sorted(random.sample(range(num_cities), 2))

    child_tour = np.full(num_cities, -1, dtype=int)
    child_tour[start : end + 1] = parent_a.tour[start : end + 1]
    used_cities = set(child_tour[start : end + 1])

    remaining_cities = [city for city in parent_b.tour if city not in used_cities]

    fill_index = 0
    for i in range(num_cities):
        if child_tour[i] == -1:
            child_tour[i] = remaining_cities[fill_index]
            fill_index += 1

    return Individual(tour=child_tour, mutation_rate=parent_a.mutation_rate)


# ==============================================================
# SURVIVOR SELECTION (Deterministic Crowding)
# ==============================================================


# TODO understand this


def elimination_with_crowding(
    population: List[Individual],
    offspring: List[Individual],
    population_size: int,
) -> List[Individual]:
    """
    Deterministic crowding using directed edge overlap.
    Each child competes with its most similar parent.
    """
    if not offspring:
        population.sort(key=lambda ind: ind.fitness)
        return population[:population_size]

    survivors = list(population)
    survivors.sort(key=lambda ind: ind.fitness)
    survivors = survivors[:population_size]

    # Build successor matrix for fast overlap computation
    pop_succs = np.empty((len(survivors), survivors[0].tour.shape[0]), dtype=np.int32)
    pop_fits = np.empty(len(survivors), dtype=np.float64)

    for i, ind in enumerate(survivors):
        pop_succs[i, :] = build_successor(ind.tour)
        pop_fits[i] = float(ind.fitness)

    # Process each offspring
    for child in offspring:
        if child.fitness is None:
            continue

        child_succ = build_successor(child.tour)
        child_fit = float(child.fitness)

        rival_idx = best_rival_index_by_edge_overlap(
            child_succ, pop_succs, child_fit, pop_fits
        )

        if child_fit < pop_fits[rival_idx]:
            survivors[rival_idx] = child
            pop_succs[rival_idx, :] = child_succ
            pop_fits[rival_idx] = child_fit

    survivors.sort(key=lambda ind: ind.fitness)
    return survivors[:population_size]


@numba.njit(cache=True)
def build_successor(tour: np.ndarray) -> np.ndarray:
    """
    Build successor representation: succ[city] = next_city in tour.
    Used for fast edge overlap computation.
    """
    n = tour.shape[0]
    succ = np.empty(n, dtype=np.int32)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        succ[a] = b
    return succ


@numba.njit(cache=True)
def best_rival_index_by_edge_overlap(
    child_succ: np.ndarray,
    pop_succs: np.ndarray,
    child_fit: float,
    pop_fits: np.ndarray,
) -> int:
    """
    Find population member with maximum directed edge overlap with child.
    Tie-breaker: closest fitness.
    """
    pop_size = pop_succs.shape[0]
    n = child_succ.shape[0]

    best_idx = 0
    best_overlap = -1
    best_fit_diff = 1e300

    for i in range(pop_size):
        overlap = 0
        for city in range(n):
            if child_succ[city] == pop_succs[i, city]:
                overlap += 1

        fit_diff = abs(child_fit - pop_fits[i])

        if (overlap > best_overlap) or (
            overlap == best_overlap and fit_diff < best_fit_diff
        ):
            best_overlap = overlap
            best_fit_diff = fit_diff
            best_idx = i

    return best_idx


# ==============================================================
# MAIN SOLVER
# ==============================================================


class r0843621:
    """
    Genetic Algorithm solver for TSP.

    Features:
        - Mixed initialization (greedy + random)
        - Tournament selection
        - Order crossover (OX)
        - Weighted mutation operators
        - Deterministic crowding survivor selection
    """

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename: str) -> int:
        """Main optimization routine."""
        # Load problem
        problem = TravelingSalesmanProblem(
            self._read_distance_matrix(filename), os.path.basename(filename)
        )
        problem.print_info()

        # Initialize population
        print_section("INITIALIZATION")
        population = initialize_population_mixed(problem, GA.POPULATION_SIZE)
        self._log_population_stats(population, "Initial Population")

        # Track best solution
        best_overall = min(population, key=lambda x: x.fitness)
        best_overall_fitness = best_overall.fitness
        stall_gens = 0

        # Evolution loop
        print_section("EVOLUTION")
        start_time = time.perf_counter()
        checkpoint = start_time

        for generation in range(1, GA.GENERATIONS + 1):
            # Generate offspring
            offspring = self._evolve_one_generation(population, problem)

            # Survivor selection
            population = elimination_with_crowding(
                population, offspring, GA.POPULATION_SIZE
            )

            # Track best solution
            gen_best = min(population, key=lambda x: x.fitness)
            gen_mean = float(np.mean([ind.fitness for ind in population]))

            if gen_best.fitness < best_overall_fitness - 1e-9:
                best_overall = gen_best
                best_overall_fitness = gen_best.fitness
                stall_gens = 0
            else:
                stall_gens += 1

            # Periodic logging
            if generation % GA.LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - checkpoint
                checkpoint = time.perf_counter()
                diversity = edge_diversity(population)

                logger.info(
                    f"  Gen {generation:4d} │ Mean: {gen_mean:12.2f} │ "
                    f"Best: {gen_best.fitness:12.2f} │ Div: {diversity:8.2%} │ "
                    f"Δt: {elapsed:7.2f}s │ NoImp: {stall_gens:4d}"
                )

            # Check time limit
            if self.reporter.report(gen_mean, gen_best.fitness, gen_best.tour) < 0:
                logger.info("\nTime limit reached")
                break

        # Final statistics
        total_time = time.perf_counter() - start_time
        print_section("RESULTS")
        print_stats_table(
            {
                "Best Fitness": best_overall.fitness,
                "Generations": generation,
                "Total Time (s)": total_time,
                "Avg Time/Gen (s)": total_time / generation,
                "Final Diversity": edge_diversity(population),
            }
        )
        logger.info("")

        return 0

    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        """Load distance matrix from CSV file."""
        with open(filename, "r") as f:
            return np.loadtxt(f, delimiter=",")

    def _evolve_one_generation(
        self,
        population: List[Individual],
        problem: TravelingSalesmanProblem,
    ) -> List[Individual]:
        """Generate offspring for one generation."""
        offspring: List[Individual] = []
        target = GA.OFFSPRING_SIZE

        while len(offspring) < target:
            # Select parents
            parent1 = tournament_selection(population, GA.TOURNAMENT_K)
            parent2 = tournament_selection(population, GA.TOURNAMENT_K)

            # Crossover or clone
            if random.random() < GA.CROSSOVER_PROB:
                child = order_crossover(problem, parent1, parent2)
            else:
                child = Individual(
                    tour=np.copy(parent1.tour), mutation_rate=parent1.mutation_rate
                )

            mutation(child)

            # Evaluate
            child.evaluate(problem)

            # Only keep valid offspring
            if np.isfinite(child.fitness):
                offspring.append(child)

        return offspring

    def _log_population_stats(self, population: List[Individual], label: str) -> None:
        """Log population fitness statistics."""
        fits = [ind.fitness for ind in population]
        print_stats_table(
            {
                "Mean": float(np.mean(fits)),
                "Best": float(np.min(fits)),
                "Worst": float(np.max(fits)),
            }
        )


# ==============================================================
# 2-OPT LOCAL SEARCH (kept for future use)
# ==============================================================


# def two_opt_local_search(
#     individual: Individual,
#     problem: TravelingSalesmanProblem,
#     max_iters: int = 5,
# ) -> Individual:
#     """
#     Apply 2-opt local search: remove two edges and reconnect.
#     Continues until no improving move found or max iterations reached.

#     Currently not integrated into main optimization loop.
#     """
#     tour = individual.tour
#     distance_matrix = problem.distance_matrix
#     feasible = problem.feasible

#     if individual.fitness is None:
#         individual.evaluate(problem)

#     iteration = 0

#     while iteration < max_iters:
#         iteration += 1

#         i, j = find_first_2opt_improvement(tour, distance_matrix, feasible)

#         if i == -1:
#             break  # Local optimum reached

#         tour[i : j + 1] = tour[i : j + 1][::-1]

#     individual.evaluate(problem)
#     return individual


# @numba.njit(cache=True)
# def find_first_2opt_improvement(
#     tour: np.ndarray,
#     distance_matrix: np.ndarray,
#     feasible: np.ndarray,
# ) -> Tuple[int, int]:
#     """
#     Find first improving 2-opt move.
#     Returns (i, j) for improvement, or (-1, -1) if none found.
#     """
#     n = tour.shape[0]

#     for i in range(1, n - 1):
#         a = tour[i - 1]
#         b = tour[i]

#         for j in range(i + 1, n):
#             next_j = (j + 1) % n
#             c = tour[j]
#             d = tour[next_j]

#             if not feasible[a, c] or not feasible[b, d]:
#                 continue

#             cost_removed = distance_matrix[a, b] + distance_matrix[c, d]
#             cost_added = distance_matrix[a, c] + distance_matrix[b, d]

#             if np.isinf(cost_removed):
#                 return i, j

#             if cost_added < cost_removed - 1e-9:
#                 return i, j

#     return -1, -1
