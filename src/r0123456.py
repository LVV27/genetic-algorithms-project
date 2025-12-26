import Reporter
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import copy
import numpy as np
import numba


# ==============================================================
# LOGGING CONFIGURATION
# ==============================================================


class ProfessionalFormatter(logging.Formatter):
    """
    Custom log formatter with severity-specific styling.

    Provides clean, readable output with emoji indicators for warnings/errors.
    """

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
logger.handlers = []  # Reset handlers when re-running in notebooks/IDEs

_handler = logging.StreamHandler()
_handler.setFormatter(ProfessionalFormatter())
logger.addHandler(_handler)
logger.propagate = False


# ==============================================================
# ALGORITHM PARAMETERS (Single Source of Truth)
# ==============================================================


@dataclass(frozen=True)
class GAParams:
    """
    Centralized configuration for the Genetic Algorithm.

    All algorithm behavior is controlled through these parameters,
    making it easy to tune and document the approach for defense.
    """

    # ===== Population Parameters =====
    POPULATION_SIZE: int = 100  # λ (lambda) - population size
    OFFSPRING_SIZE: int = 100  # μ (mu) - offspring per generation
    GENERATIONS: int = (
        10_000_000  # Maximum generations (usually terminated by time limit)
    )

    # ===== Selection Parameters =====
    TOURNAMENT_K: int = (
        5  # Tournament size for parent selection (larger = more selective)
    )

    # ===== Mutation Parameters =====
    # Self-adaptive mutation rate bounds
    MUTATION_ALPHA_MIN: float = 0.08  # Minimum mutation rate (exploitation)
    MUTATION_ALPHA_MAX: float = 0.20  # Maximum mutation rate (exploration)
    DIVERSITY_CHECK_INTERVAL: int = (
        5  # How often to measure diversity for adaptive mutation
    )

    # ===== Crossover Parameters =====
    CROSSOVER_PROB: float = 0.8  # Probability of crossover vs. cloning
    ERX_PROB_SPARSE: float = 0.4  # Probability of using ERX in sparse graphs

    # ===== Initialization Parameters =====
    GREEDY_SEED_COUNT: int = 20  # Number of best greedy solutions to seed population
    GREEDY_RESTARTS: int = (
        1000  # Number of different start cities for greedy construction
    )

    # ===== Local Search Parameters =====
    LOCAL_SEARCH_MAX_ITERS: int = 5  # Max iterations per 2-opt application

    # When to apply 2-opt (selective to save computation)
    LSO_APPLY_IF_BEATS_ANY_PARENT: bool = False  # Apply if offspring beats a parent
    LSO_NEAR_BEST_FRAC: float = 0.01  # Apply if within 1% of best
    LSO_ALWAYS_IMPROVE_TOP_K: int = 3  # Always apply to top K offspring
    LSO_LOG_COUNTS: bool = False  # Log how many get local search

    # ===== Sparse Graph Strategy =====
    SPARSITY_THRESHOLD: float = 1.1  # Fraction of inf edges to trigger sparse mode
    SPARSE_OFFSPRING_TARGET_MULTIPLIER: int = (
        3  # Target * this = max attempts in sparse graphs
    )
    SPARSE_MIN_OFFSPRING_FACTOR: int = 3  # Minimum offspring = max(3, pop_size / this)

    # ===== Perturbation Parameters =====
    # Used during initialization to diversify from greedy seeds
    PERTURB_LIGHT_SWAPS: int = 200  # Divisor: n // this = number of swaps for light
    PERTURB_MEDIUM_SWAPS: int = 100  # Divisor: n // this = number of swaps for medium
    PERTURB_HEAVY_SWAPS: int = 50  # Divisor: n // this = number of swaps for heavy
    PERTURB_HEAVY_ITERATIONS: int = 2  # Number of perturbations for heavy mode

    # ===== Repair Parameters =====
    MAX_REPAIR_ATTEMPTS: int = 100  # Maximum attempts to fix invalid tour
    PERTURBATION_MAX_ATTEMPTS_MULTIPLIER: int = 50  # Max attempts = needed * this

    # ===== Large Neighborhood Search (LNS) Parameters =====
    LNS_STALL_THRESHOLD: int = 200  # Apply LNS after this many stagnant generations
    LNS_CHECK_INTERVAL: int = 10  # Check every N generations when stalled
    LNS_NUM_RESTARTS: int = 50  # Number of LNS restart attempts
    LNS_2OPT_ITERS: int = 10  # 2-opt iterations after LNS reconstruction
    LNS_DESTROY_FRACTION: float = 0.4  # Fraction of tour to destroy/rebuild
    LNS_ACCEPTANCE_UPHILL_DELTA: int = (
        1000  # Accept uphill moves up to this delta when stalled
    )
    LNS_ACCEPTANCE_STALL_THRESHOLD: int = 200  # Only accept uphill if stalled this long

    # ===== Logging Parameters =====
    LOG_INTERVAL: int = 100  # Print statistics every N generations
    LOG_SPARSE_OFFSPRING_STATS_INTERVAL: int = 20  # Log offspring generation stats

    # ===== Missing Edge Penalty =====
    # Whether to replace missing (Inf) edges with a large penalty cost.
    USE_PENALTY_FOR_INF: bool = True
    # Penalty multiplier: penalty = (max_finite_edge + 1) * INF_PENALTY_FACTOR
    INF_PENALTY_FACTOR: float = 10

    # ===== Candidate List Strategy =====
    # How many nearest neighbours of each city should keep their original cost.
    # Set to 0 to disable; otherwise edges outside these k neighbours get penalised.
    CANDIDATE_LIST_SIZE: int = 0


GA = GAParams()


@dataclass(frozen=True)
class GAFeatures:
    USE_SPARSE_STRATEGY: bool = False
    USE_DIVERSITY_INJECTION: bool = True
    USE_STAGNATION_MUTATION_BOOST: bool = False
    USE_ADAPTIVE_MUTATION_RATE: bool = False
    USE_MUTATION_PERTURB: bool = False
    USE_MUTATION_CLASSIC: bool = True
    USE_LOCAL_SEARCH: bool = False
    USE_LNS: bool = False


FEATURES = GAFeatures()


# ==============================================================
# CONSOLE OUTPUT HELPERS
# ==============================================================


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header for console output."""
    logger.info("")
    logger.info("─" * width)
    logger.info(f" {title}")
    logger.info("─" * width)


def print_stats_table(stats: dict) -> None:
    """Print a dictionary of statistics with aligned columns."""
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
    TSP instance wrapper.

    Stores the distance matrix and provides convenient access methods.
    Includes optional benchmark values for validation.
    """

    # Known heuristic values for standard benchmarks (for validation)
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

        # feasibility matrix for sparse-aware operators
        self.feasible = np.isfinite(distance_matrix)

        # Start with a copy for penalised distances
        self.penalized_matrix = distance_matrix.copy()
        self.penalty_distance = None

        # Only apply the scheme if enabled and the matrix has Inf entries
        if GA.USE_PENALTY_FOR_INF and np.isinf(distance_matrix).any():
            finite_mask = np.isfinite(distance_matrix)
            max_finite = (
                float(np.max(distance_matrix[finite_mask]))
                if finite_mask.any()
                else 1.0
            )
            self.penalty_distance = (max_finite + 1.0) * GA.INF_PENALTY_FACTOR

            # Candidate list handling: keep k nearest neighbours at original cost
            if GA.CANDIDATE_LIST_SIZE and GA.CANDIDATE_LIST_SIZE > 0:
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

                penalised = distance_matrix.copy()
                penalised[~candidate_mask] = self.penalty_distance
                penalised[~np.isfinite(penalised)] = self.penalty_distance
                self.penalized_matrix = penalised
            else:
                # No candidate list: penalise all missing edges
                penalised = distance_matrix.copy()
                penalised[~finite_mask] = self.penalty_distance
                self.penalized_matrix = penalised
        else:
            # no missing edges or penalty disabled
            self.penalized_matrix = distance_matrix

    def get_distance(self, city_a: int, city_b: int) -> float:
        """Return the distance between two cities."""
        return self.distance_matrix[city_a, city_b]

    def print_info(self) -> None:
        """Print instance metadata and known benchmark if available."""
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
        tour: Permutation of city indices representing the route
        mutation_rate: Self-adaptive parameter controlling mutation strength
        fitness: Total tour length (lower is better, None if not evaluated)
    """

    def __init__(
        self,
        problem: Optional[TravelingSalesmanProblem] = None,
        tour: Optional[np.ndarray] = None,
        mutation_rate: Optional[float] = None,
    ):
        # Initialize tour
        if tour is not None:
            self.tour = np.asarray(tour, dtype=int)
        elif problem is not None:
            self.tour = np.random.permutation(problem.num_cities)
        else:
            raise ValueError(
                "Either `problem` (for random init) or `tour` must be provided."
            )

        # Initialize mutation rate within allowed bounds
        if mutation_rate is None:
            self.mutation_rate = random.uniform(
                GA.MUTATION_ALPHA_MIN,
                GA.MUTATION_ALPHA_MAX,
            )
        else:
            self.mutation_rate = float(mutation_rate)

        self.fitness: Optional[float] = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        """
        Evaluate tour fitness on the penalised distance matrix.
        Tours with missing edges accrue a large penalty cost.
        """
        self.fitness = evaluate_tour_numba(self.tour, problem.penalized_matrix)
        return self.fitness


@numba.njit(cache=True)
def evaluate_tour_numba(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute total tour length efficiently using Numba JIT compilation.

    Returns np.inf if any edge is invalid (infinite distance).
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


@numba.njit(cache=True)
def build_successor(tour: np.ndarray) -> np.ndarray:
    """
    Build successor representation: succ[city] = next_city in the tour.
    Directed edges are (city -> succ[city]).
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
    Return index of rival in population with maximum directed edge overlap.
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


Edge = Tuple[int, int]


def directed_edge_set(tour: np.ndarray) -> Set[Edge]:
    """Directed edges (a->b) along the cyclic tour."""
    n = tour.shape[0]
    return {(int(tour[i]), int(tour[(i + 1) % n])) for i in range(n)}


def edge_diversity(population: List[Individual]) -> float:
    if not population:
        return 0.0
    n = len(population[0].tour)
    edges = set()
    for ind in population:
        t = ind.tour
        for i in range(n):
            edges.add((int(t[i]), int(t[(i + 1) % n])))
    denom = min(len(population) * n, n * (n - 1))
    return len(edges) / denom


def find_most_similar_by_edge_overlap(
    child: Individual,
    population: List[Individual],
) -> Optional[Individual]:
    """
    Find the individual in the population that shares the most directed edges
    with the child (deterministic crowding niche matching).
    """
    if not population:
        return None

    child_edges = directed_edge_set(child.tour)

    best_rival = None
    best_overlap = -1
    best_fit_diff = float("inf")  # tie-breaker

    # If child's fitness is None (shouldn't happen), treat as 0 for tie-breaking
    child_fit = child.fitness if child.fitness is not None else 0.0

    for rival in population:
        rival_edges = directed_edge_set(rival.tour)
        overlap = len(child_edges & rival_edges)

        rival_fit = rival.fitness if rival.fitness is not None else 0.0
        fit_diff = abs(child_fit - rival_fit)

        if overlap > best_overlap or (
            overlap == best_overlap and fit_diff < best_fit_diff
        ):
            best_overlap = overlap
            best_fit_diff = fit_diff
            best_rival = rival

    return best_rival


# ==============================================================
# POPULATION INITIALIZATION
# ==============================================================


def perturb_double_bridge(tour: np.ndarray) -> np.ndarray:
    n = len(tour)
    if n < 8:
        return tour.copy()

    p1, p2, p3 = sorted(random.sample(range(1, n), 3))
    A = tour[:p1]
    B = tour[p1:p2]
    C = tour[p2:p3]
    D = tour[p3:]

    # Randomly choose a reconnection pattern
    patterns = [
        np.concatenate([A, C, B, D]),
        np.concatenate([A, D, C, B]),
        np.concatenate([A, C, D, B]),
        np.concatenate([A, B, D, C]),
    ]
    return random.choice(patterns)


def perturb_multi_swap(tour: np.ndarray, k: int) -> np.ndarray:
    """Apply k random city swaps to the tour."""
    tour = tour.copy()
    n = len(tour)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def perturb_segment_reverse(tour: np.ndarray) -> np.ndarray:
    """Reverse a random contiguous segment of the tour."""
    tour = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i : j + 1] = tour[i : j + 1][::-1]
    return tour


def apply_perturbation(tour: np.ndarray, strength: str) -> np.ndarray:
    """
    Apply perturbation of specified strength to diversify the tour.

    Args:
        tour: Original tour
        strength: 'light', 'medium', or 'heavy'

    Returns:
        Perturbed tour
    """
    n = len(tour)

    if strength == "light":
        k = max(1, n // GA.PERTURB_LIGHT_SWAPS)
        return perturb_multi_swap(tour, k)

    if strength == "medium":
        k = max(2, n // GA.PERTURB_MEDIUM_SWAPS)
        return random.choice(
            [
                lambda t: perturb_double_bridge(t),
                lambda t: perturb_multi_swap(t, k),
                lambda t: perturb_segment_reverse(t),
            ]
        )(tour)

    # Heavy perturbation: apply multiple operations
    result = tour.copy()
    k = max(3, n // GA.PERTURB_HEAVY_SWAPS)
    for _ in range(GA.PERTURB_HEAVY_ITERATIONS):
        result = random.choice(
            [
                perturb_double_bridge,
                lambda x: perturb_multi_swap(x, k),
                perturb_segment_reverse,
            ]
        )(result)
    return result


def repair_tour(
    problem: TravelingSalesmanProblem,
    tour: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Attempt to repair a tour containing invalid (infinite) edges.

    Strategy: For each invalid edge, try swapping with later cities
    until a valid configuration is found.

    Returns:
        Repaired tour if successful, None if repair fails
    """
    tour = tour.copy()
    n = len(tour)

    for _ in range(GA.MAX_REPAIR_ATTEMPTS):
        repaired_any = False

        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]

            # Check if current edge is invalid
            if not np.isfinite(problem.get_distance(a, b)):
                # Try swapping b with a later city c
                for j in range(i + 2, n):
                    c = tour[j]

                    # Check if new edges would be valid
                    if np.isfinite(problem.get_distance(a, c)) and np.isfinite(
                        problem.get_distance(c, b)
                    ):
                        tour[(i + 1) % n], tour[j] = tour[j], tour[(i + 1) % n]
                        repaired_any = True
                        break

                if not repaired_any:
                    return None  # Unrecoverable edge

        if not repaired_any:
            return tour  # Fully repaired

    return None  # Max attempts exceeded


def fill_with_perturbations(
    problem: TravelingSalesmanProblem,
    base_population: List[Individual],
    target_size: int,
) -> int:
    """
    Fill population by creating perturbed copies of existing good solutions.

    Uses top 50% of population as templates and applies various perturbations.
    Ensures all generated individuals are valid (finite fitness).

    Returns:
        Number of individuals added
    """
    added = 0
    attempts = 0
    needed = target_size - len(base_population)
    max_attempts = needed * GA.PERTURBATION_MAX_ATTEMPTS_MULTIPLIER

    # Use top half as templates
    templates = sorted(base_population, key=lambda ind: ind.fitness)
    templates = templates[: max(1, len(templates) // 2)]

    while len(base_population) < target_size and attempts < max_attempts:
        attempts += 1
        parent = random.choice(templates)

        # Choose perturbation strength with decreasing probability
        strength = random.choices(
            ["light", "medium", "heavy"],
            weights=[0.4, 0.4, 0.2],
            k=1,
        )[0]

        # is this not just overriding the params

        tour = apply_perturbation(parent.tour, strength)
        ind = Individual(tour=tour, mutation_rate=parent.mutation_rate)
        ind.evaluate(problem)

        # Attempt repair if invalid
        if not np.isfinite(ind.fitness):
            repaired = repair_tour(problem, tour)
            if repaired is None:
                continue
            ind = Individual(tour=repaired, mutation_rate=parent.mutation_rate)
            ind.evaluate(problem)

        if np.isfinite(ind.fitness):
            base_population.append(ind)
            added += 1

    return added


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

        # Vectorized search for nearest unvisited city
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


def nearest_neighbor_greedy(
    problem: TravelingSalesmanProblem,
    start_city: int = 0,
) -> Optional[np.ndarray]:
    """
    Construct a tour using the nearest-neighbor heuristic.

    Starting from start_city, repeatedly visit the closest unvisited city.

    Returns:
        Complete tour, or None if construction fails (sparse graph)
    """
    tour = nearest_neighbor_greedy_numba(problem.distance_matrix, start_city)

    # Check if construction failed
    if tour[0] == -1:
        return None

    return tour


def generate_greedy_candidates(
    problem: TravelingSalesmanProblem,
    restart_budget: int,
) -> List[Individual]:
    """
    Generate multiple greedy tours from different starting cities.

    Only valid (finite fitness) tours are returned.
    """
    n = problem.num_cities
    starts = random.sample(range(n), k=min(n, restart_budget))

    candidates: List[Individual] = []

    for start in starts:
        tour = nearest_neighbor_greedy(problem, start)
        if tour is None:
            continue

        ind = Individual(tour=tour)
        ind.evaluate(problem)

        if np.isfinite(ind.fitness):
            candidates.append(ind)

    return candidates


def select_best_unique_seeds(
    candidates: List[Individual],
    seed_target: int,
) -> Tuple[List[Individual], Set]:
    """
    Select the best unique tours (no duplicates).

    Returns:
        Tuple of (selected individuals, set of tour keys for deduplication)
    """
    seeds: List[Individual] = []
    seen_keys: Set = set()

    for ind in sorted(candidates, key=lambda x: x.fitness):
        key = tuple(ind.tour.tolist())

        if key in seen_keys:
            continue

        seen_keys.add(key)
        seeds.append(ind)

        if len(seeds) >= seed_target:
            break

    return seeds, seen_keys


def clone_to_size(
    population: List[Individual],
    target_size: int,
) -> int:
    """
    Clone existing individuals (with deep copy) until target size is reached.

    Used as last resort when other initialization methods don't produce enough individuals.

    Returns:
        Number of clones added
    """
    if not population:
        return 0

    added = 0
    base = len(population)
    i = 0

    while len(population) < target_size:
        population.append(copy.deepcopy(population[i % base]))
        i += 1
        added += 1

    return added


def initialize_population_greedy_sparse_aware(
    problem: TravelingSalesmanProblem,
    population_size: int,
) -> List[Individual]:
    """
    Initialize population using a multi-stage strategy:

    1. Generate many greedy nearest-neighbor tours from different start cities
    2. Select the best unique tours as seeds
    3. Create diversity by perturbing these seeds
    4. Clone only if necessary to reach target size

    This approach works well for both dense and sparse graphs.
    """
    # Stage 1: Generate greedy seeds
    greedy_candidates = generate_greedy_candidates(
        problem,
        restart_budget=max(GA.GREEDY_RESTARTS, population_size * 20),
    )

    population, _ = select_best_unique_seeds(
        greedy_candidates,
        GA.GREEDY_SEED_COUNT,
    )

    greedy_count = len(population)

    if greedy_count == 0:
        raise RuntimeError("No valid greedy tours found - graph may be disconnected")

    # Stage 2: Fill with perturbations
    perturb_count = fill_with_perturbations(
        problem,
        population,
        population_size,
    )

    # Stage 3: Clone only if absolutely necessary
    clone_count = 0
    if len(population) < population_size:
        clone_count = clone_to_size(population, population_size)

    # Log initialization summary
    logger.info(
        f"Initialized {len(population)} individuals "
        f"({greedy_count} greedy, "
        f"{perturb_count} perturbations, "
        f"{clone_count} clones)"
    )

    return population


def initialize_population_random(
    problem: TravelingSalesmanProblem,
    population_size: int,
) -> List[Individual]:
    population: List[Individual] = []

    while len(population) < population_size:
        ind = Individual(problem=problem)
        ind.evaluate(problem)

        if np.isfinite(ind.fitness):
            population.append(ind)

    logger.info(f"Initialized {len(population)} individuals (random)")
    return population



# ==============================================================
# SELECTION
# ==============================================================


def tournament_selection(
    population: List[Individual],
    tournament_size: int,
) -> Individual:
    """
    Select one individual using tournament selection.

    Randomly sample tournament_size individuals and return the best.
    Larger tournament size increases selection pressure.
    """
    competitors = random.sample(population, tournament_size)
    winner = min(competitors, key=lambda ind: ind.fitness)
    return winner


# ==============================================================
# MUTATION OPERATORS
# ==============================================================


def mutation_perturb(individual: Individual) -> None:
    if random.random() >= individual.mutation_rate:
        return
    strength = random.choices(
        ["light", "medium", "heavy"],
        weights=[0.70, 0.25, 0.05],  # tune
        k=1,
    )[0]
    individual.tour = apply_perturbation(individual.tour, strength)


def mutation_swap(tour: np.ndarray) -> None:
    """Swap two random cities in-place."""
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]


def mutation_inversion(tour: np.ndarray) -> None:
    """Reverse a random contiguous segment in-place."""
    start, end = sorted(random.sample(range(len(tour)), 2))
    tour[start : end + 1] = tour[start : end + 1][::-1]


def mutation_insertion(individual: Individual) -> None:
    """
    Remove one city and insert it at another position.

    This is a more disruptive mutation than swap or inversion.
    """
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
    Apply mutation to an individual using weighted operator selection.

    Mutation occurs with probability = individual.mutation_rate.
    Operator weights favor inversion (good for TSP) over swap and insertion.
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


def mutation_sparse_aware(
    individual: Individual,
    problem: TravelingSalesmanProblem,
) -> None:
    """
    Apply mutation while ensuring validity for sparse graphs.

    In sparse graphs, random mutations often create invalid tours.
    This function tries multiple mutations and keeps the first valid one.
    Falls back to original tour if all attempts fail.
    """
    if random.random() >= individual.mutation_rate:
        return

    original_tour = individual.tour.copy()
    original_fitness = individual.fitness

    for _ in range(GA.MAX_REPAIR_ATTEMPTS):
        operator = random.choice(["swap", "double_swap", "inversion"])

        if operator == "swap":
            mutation_swap(individual.tour)

        elif operator == "double_swap":
            # Two swaps can escape local optima better
            mutation_swap(individual.tour)
            mutation_swap(individual.tour)

        else:  # Inversion with size constraint
            start, end = sorted(random.sample(range(len(individual.tour)), 2))
            # Avoid very large inversions which are more likely to be invalid
            if end - start > len(individual.tour) // 3:
                continue
            individual.tour[start : end + 1] = individual.tour[start : end + 1][::-1]

        # Keep mutation if it produces a valid tour
        if np.isfinite(individual.evaluate(problem)):
            return

        # Otherwise revert and try again
        individual.tour = original_tour.copy()
        individual.fitness = original_fitness


def adaptive_mutation_rate(
    individual: Individual,
    diversity: float,
) -> None:
    """
    Adapt individual's mutation rate based on population diversity.

    Strategy:
        - Low diversity → increase mutation rate (explore more)
        - High diversity → decrease mutation rate (exploit more)

    Mutation rate is always clamped to configured bounds.
    """
    min_rate = GA.MUTATION_ALPHA_MIN
    max_rate = GA.MUTATION_ALPHA_MAX

    base_rate = (min_rate + max_rate) / 2.0

    # Diversity adjustment: lower diversity increases mutation
    adjustment = (1.0 - diversity) * 0.5

    new_rate = base_rate + adjustment
    individual.mutation_rate = float(np.clip(new_rate, min_rate, max_rate))


def adaptive_mutation_stagnation(
    population: List[Individual],
    stall_gens: int,
) -> None:
    """
    Increase mutation rates when stagnating to force exploration.
    """
    if stall_gens < 50:
        return  # Not stalled yet

    # Calculate mutation boost based on stall duration
    if stall_gens < 200:
        boost = 1.2
    elif stall_gens < 500:
        boost = 1.5
    else:
        boost = 2.0

    for ind in population:
        new_rate = min(GA.MUTATION_ALPHA_MAX, ind.mutation_rate * boost)
        ind.mutation_rate = new_rate


# ==============================================================
# CROSSOVER OPERATORS
# ==============================================================


def order_crossover(
    problem: TravelingSalesmanProblem,
    parent_a: Individual,
    parent_b: Individual,
) -> Individual:
    """
    Order Crossover (OX) for permutation representation.

    Strategy:
        1. Copy a contiguous segment from parent_a
        2. Fill remaining positions using parent_b's city order

    This preserves relative city order from parents while creating new tours.
    """
    num_cities = problem.num_cities
    start, end = sorted(random.sample(range(num_cities), 2))

    child_tour = np.full(num_cities, -1, dtype=int)

    # Copy segment from first parent
    child_tour[start : end + 1] = parent_a.tour[start : end + 1]
    used_cities = set(child_tour[start : end + 1])

    # Fill remaining positions from second parent
    remaining_cities = [city for city in parent_b.tour if city not in used_cities]

    fill_index = 0
    for i in range(num_cities):
        if child_tour[i] == -1:
            child_tour[i] = remaining_cities[fill_index]
            fill_index += 1

    return Individual(
        tour=child_tour,
        mutation_rate=parent_a.mutation_rate,
    )


def pmx_crossover(
    problem: TravelingSalesmanProblem,
    parent_a: Individual,
    parent_b: Individual,
) -> Individual:
    """
    Partially Mapped Crossover (PMX) - better for escaping local optima.

    Creates more diverse offspring than OX by using position-based mapping.
    """
    n = problem.num_cities
    child_tour = np.copy(parent_a.tour)

    # Select random crossover segment
    start, end = sorted(random.sample(range(n), 2))

    # Build mapping from segment
    mapping = {}
    for i in range(start, end + 1):
        city_a = parent_a.tour[i]
        city_b = parent_b.tour[i]
        mapping[city_a] = city_b
        child_tour[i] = city_b

    # Fix conflicts outside the segment
    for i in list(range(0, start)) + list(range(end + 1, n)):
        city = child_tour[i]

        # If this city was swapped into the segment, follow mapping chain
        while city in mapping.values():
            # Find which key maps to this city
            for key, val in mapping.items():
                if val == city:
                    city = key
                    break

        child_tour[i] = city

    return Individual(tour=child_tour, mutation_rate=parent_a.mutation_rate)


# ==============================================================
# EDGE RECOMBINATION CROSSOVER (for sparse graphs)
# ==============================================================


def _cyclic_neighbors(tour: np.ndarray) -> Dict[int, Set[int]]:
    """
    Build adjacency table from a tour.

    Each city maps to its immediate predecessor and successor (cyclic).
    """
    n = len(tour)
    adjacency: Dict[int, Set[int]] = {int(c): set() for c in tour}

    for i in range(n):
        city = int(tour[i])
        left = int(tour[(i - 1) % n])
        right = int(tour[(i + 1) % n])
        adjacency[city].add(left)
        adjacency[city].add(right)

    return adjacency


def _merge_edge_tables(tour1: np.ndarray, tour2: np.ndarray) -> Dict[int, Set[int]]:
    """Merge adjacency tables from both parents."""
    adj1 = _cyclic_neighbors(tour1)
    adj2 = _cyclic_neighbors(tour2)
    merged = {city: set(adj1[city]) | set(adj2[city]) for city in adj1.keys()}
    return merged


def _remove_city_from_all(edge_table: Dict[int, Set[int]], city: int) -> None:
    """Remove a city from all adjacency lists (it's been visited)."""
    for neighbors in edge_table.values():
        neighbors.discard(city)


def _feasible_neighbors(
    problem: TravelingSalesmanProblem,
    current: int,
    candidates: Set[int],
) -> List[int]:
    """Filter candidates to those with valid (finite) edges from current city."""
    return [c for c in candidates if np.isfinite(problem.get_distance(current, c))]


def _choose_next_city_erx(
    problem: TravelingSalesmanProblem,
    edge_table: Dict[int, Set[int]],
    current: int,
    remaining: Set[int],
) -> int:
    """
    Select next city using Edge Recombination heuristic.

    Strategy (in priority order):
        1. Prefer cities that were neighbors in parents (edge preservation)
        2. Among valid neighbors, choose one with smallest degree (avoid dead ends)
        3. If no parent edges work, choose any feasible city
        4. Last resort: random remaining city (may be invalid)
    """
    # Try parent-neighbor edges first
    parent_neighbors = edge_table.get(current, set())
    options = _feasible_neighbors(problem, current, parent_neighbors & remaining)

    if options:
        # Prefer cities with smaller degree (classic ERX heuristic)
        min_degree = min(len(edge_table[c]) for c in options)
        best = [c for c in options if len(edge_table[c]) == min_degree]
        return random.choice(best)

    # Fallback: any remaining city with valid edge
    feasible_any = [
        c for c in remaining if np.isfinite(problem.get_distance(current, c))
    ]
    if feasible_any:
        return random.choice(feasible_any)

    # Last resort: may create invalid edge
    return random.choice(list(remaining))


def edge_recombination_crossover_sparse_aware(
    problem: TravelingSalesmanProblem,
    parent1: Individual,
    parent2: Individual,
    start_city: Optional[int] = None,
) -> Individual:
    """
    Edge Recombination Crossover adapted for sparse graphs.

    ERX preserves edges from parents, which increases the chance of
    producing valid offspring in sparse graphs where many edges are missing.

    Tries multiple start cities to find a valid tour. If all attempts fail,
    clones the better parent.
    """
    n = problem.num_cities

    # Try multiple start cities
    max_restarts = 2
    starts: List[int] = []
    if start_city is not None:
        starts.append(int(start_city))
    starts.extend(random.sample(range(n), k=min(n, max_restarts)))

    # Deduplicate while preserving order
    seen = set()
    starts = [s for s in starts if not (s in seen or seen.add(s))]

    for start in starts:
        edge_table = _merge_edge_tables(parent1.tour, parent2.tour)
        remaining = set(range(n))

        current = int(start)
        child_tour: List[int] = [current]
        remaining.remove(current)
        _remove_city_from_all(edge_table, current)

        # Build tour greedily using edge table
        while remaining:
            next_city = _choose_next_city_erx(problem, edge_table, current, remaining)
            child_tour.append(next_city)
            remaining.remove(next_city)
            _remove_city_from_all(edge_table, next_city)
            current = next_city

        # Check if tour is valid
        child = Individual(
            tour=np.asarray(child_tour, dtype=int), mutation_rate=parent1.mutation_rate
        )
        child.evaluate(problem)

        if np.isfinite(child.fitness):
            return child

    # Fallback: clone better parent
    better = parent1 if parent1.fitness < parent2.fitness else parent2
    clone = Individual(tour=np.copy(better.tour), mutation_rate=better.mutation_rate)
    clone.fitness = better.fitness
    return clone


# ==============================================================
# LOCAL SEARCH (2-opt)
# ==============================================================


def two_opt_local_search(
    individual: Individual,
    problem: TravelingSalesmanProblem,
    max_iters: int = None,
) -> Individual:
    """
    Apply 2-opt local search to improve a tour.

    2-opt removes two edges and reconnects the tour in the only other way possible,
    accepting the change if it improves fitness.

    Continues until no improving move is found or max iterations reached.
    """
    if max_iters is None:
        max_iters = GA.LOCAL_SEARCH_MAX_ITERS

    tour = individual.tour
    distance_matrix = problem.distance_matrix
    feasible = problem.feasible

    if individual.fitness is None:
        individual.evaluate(problem)

    iteration = 0

    while iteration < max_iters:
        iteration += 1

        # Find first improving 2-opt move
        i, j = find_first_2opt_improvement(tour, distance_matrix, feasible)

        if i == -1:
            break  # Local optimum reached

        # Apply 2-opt swap (reverse segment between i and j)
        tour[i : j + 1] = tour[i : j + 1][::-1]

    # Re-evaluate after modifications
    individual.evaluate(problem)
    return individual


@numba.njit(cache=True)
def find_first_2opt_improvement(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    feasible: np.ndarray,
) -> Tuple[int, int]:
    """
    Scan for first improving 2-opt move using Numba acceleration.

    Returns:
        (i, j) indices for improvement, or (-1, -1) if none found
    """
    n = tour.shape[0]

    for i in range(1, n - 1):
        a = tour[i - 1]
        b = tour[i]

        for j in range(i + 1, n):
            next_j = (j + 1) % n
            c = tour[j]
            d = tour[next_j]

            # Check if new edges would be valid
            if not feasible[a, c] or not feasible[b, d]:
                continue

            cost_removed = distance_matrix[a, b] + distance_matrix[c, d]
            cost_added = distance_matrix[a, c] + distance_matrix[b, d]

            # Repair case: current tour has invalid edge
            if np.isinf(cost_removed):
                return i, j

            # Improvement case
            if cost_added < cost_removed - 1e-9:
                return i, j

    return -1, -1


# ==============================================================
# LARGE NEIGHBORHOOD SEARCH (LNS)
# ==============================================================


def lns_destroy_repair(
    problem: TravelingSalesmanProblem,
    tour: np.ndarray,
    destroy_fraction: float = None,
) -> Optional[np.ndarray]:
    """
    Large Neighborhood Search: destroy part of tour and rebuild greedily.

    Strategy:
        1. Remove a fraction of cities from the tour
        2. Greedily reinsert them at positions that minimize cost

    This can escape local optima by making larger moves than standard mutation.

    Args:
        problem: TSP instance
        tour: Current tour
        destroy_fraction: Fraction of cities to remove and reinsert

    Returns:
        Reconstructed tour, or None if reconstruction fails
    """
    if destroy_fraction is None:
        destroy_fraction = GA.LNS_DESTROY_FRACTION

    n = len(tour)
    num_to_remove = int(n * destroy_fraction)

    # Remove random cities
    removed_indices = random.sample(range(n), num_to_remove)
    removed_cities = [tour[i] for i in removed_indices]
    random.shuffle(removed_cities)

    remaining = [c for c in tour if c not in removed_cities]

    # Greedily reinsert each removed city
    for city in removed_cities:
        best_position = None
        best_cost = float("inf")

        # Try inserting at each position
        for i in range(len(remaining)):
            prev_city = remaining[i - 1]
            next_city = remaining[i]

            # Check if insertion would be valid
            if not np.isfinite(problem.get_distance(prev_city, city)):
                continue
            if not np.isfinite(problem.get_distance(city, next_city)):
                continue

            # Calculate insertion cost (with small randomization to break ties)
            cost = (
                problem.get_distance(prev_city, city)
                + problem.get_distance(city, next_city)
                - problem.get_distance(prev_city, next_city)
            )

            # Small random perturbation helps escape local optima
            cost *= 1.0 + random.uniform(-0.02, 0.02)

            if cost < best_cost:
                best_cost = cost
                best_position = i

        if best_position is None:
            return None  # Couldn't reinsert city

        remaining.insert(best_position, city)

    return np.array(remaining, dtype=int)


# ==============================================================
# HYBRID LOCAL SEARCH APPLICATION
# ==============================================================


def _compute_near_best_threshold(
    pop_best: Optional[float],
    overall_best: Optional[float],
) -> Optional[float]:
    """
    Compute fitness threshold for "near best" classification.

    Returns:
        Fitness value representing "close to best" (within configured fraction)
    """
    thresholds: List[float] = []

    if pop_best is not None and np.isfinite(pop_best):
        thresholds.append(pop_best * (1.0 + GA.LSO_NEAR_BEST_FRAC))

    if overall_best is not None and np.isfinite(overall_best):
        thresholds.append(overall_best * (1.0 + GA.LSO_NEAR_BEST_FRAC))

    return min(thresholds) if thresholds else None


def apply_local_search_hybrid(
    offspring: List[Individual],
    problem: TravelingSalesmanProblem,
    pop_best_fitness: Optional[float] = None,
    best_overall_fitness: Optional[float] = None,
) -> None:
    """
    Apply 2-opt local search selectively to promising offspring.

    Strategy:
        - Always apply to top K offspring (most promising)
        - Apply to offspring near best fitness (within configured threshold)
        - Apply to offspring that beat at least one parent

    This balances solution quality with computational efficiency.
    """
    if not offspring:
        return

    # Ensure all fitness values are computed and sort
    offspring.sort(key=lambda ind: ind.fitness)

    # Always improve top K
    top_k = GA.LSO_ALWAYS_IMPROVE_TOP_K
    selected_ids = {id(ind) for ind in offspring[:top_k]}

    # Compute "near best" threshold
    near_best_threshold = _compute_near_best_threshold(
        pop_best=pop_best_fitness,
        overall_best=best_overall_fitness,
    )

    # Select additional offspring based on criteria
    for ind in offspring[top_k:]:
        if id(ind) in selected_ids:
            continue

        # Check if near best
        near_best = (
            near_best_threshold is not None and ind.fitness <= near_best_threshold
        )

        # Check if beats parent (if configured)
        beats_parent = False
        if GA.LSO_APPLY_IF_BEATS_ANY_PARENT:
            p1_fit = getattr(ind, "_p1_fitness", None)
            p2_fit = getattr(ind, "_p2_fitness", None)
            if p1_fit is not None and p2_fit is not None:
                beats_parent = (ind.fitness + 1e-9) < max(p1_fit, p2_fit)

        if near_best or beats_parent:
            selected_ids.add(id(ind))

    if GA.LSO_LOG_COUNTS:
        logger.info(f"  Applying 2-opt to {len(selected_ids)} offspring")

    # Apply 2-opt to selected individuals
    for ind in offspring:
        if id(ind) in selected_ids:
            two_opt_local_search(ind, problem)


# ==============================================================
# SURVIVOR SELECTION
# ==============================================================


def elimination_with_crowding(
    population: List[Individual],
    offspring: List[Individual],
    population_size: int,
) -> List[Individual]:
    """
    Deterministic crowding using directed edge overlap (Numba-accelerated).

    Rival chosen = survivor with max overlap in successor representation.
    Child replaces rival if child is fitter.
    """
    if not offspring:
        population.sort(key=lambda ind: ind.fitness)
        return population[:population_size]

    # Start with survivors list
    survivors = list(population)

    # Ensure we have exactly population_size survivors to compare against
    survivors.sort(key=lambda ind: ind.fitness)
    survivors = survivors[:population_size]

    # Build successor matrix + fitness array for survivors
    # pop_succs[i, city] = next city after 'city' in survivor i
    pop_succs = np.empty((len(survivors), survivors[0].tour.shape[0]), dtype=np.int32)
    pop_fits = np.empty(len(survivors), dtype=np.float64)

    for i, ind in enumerate(survivors):
        pop_succs[i, :] = build_successor(ind.tour)
        pop_fits[i] = float(ind.fitness)

    # Process offspring
    for child in offspring:
        if child.fitness is None:
            # should not happen, but be safe
            continue

        child_succ = build_successor(child.tour)
        child_fit = float(child.fitness)

        rival_idx = best_rival_index_by_edge_overlap(
            child_succ, pop_succs, child_fit, pop_fits
        )

        if child_fit < pop_fits[rival_idx]:
            # Replace rival in-place (no list remove, keeps arrays aligned)
            survivors[rival_idx] = child
            pop_succs[rival_idx, :] = child_succ
            pop_fits[rival_idx] = child_fit

    survivors.sort(key=lambda ind: ind.fitness)
    return survivors[:population_size]


# ==============================================================
# SPARSITY DETECTION
# ==============================================================


def is_sparse_matrix(distance_matrix: np.ndarray) -> bool:
    """
    Detect if distance matrix represents a sparse graph.

    Sparse = many off-diagonal infinite edges (missing connections).

    Returns:
        True if sparsity exceeds configured threshold
    """
    n = distance_matrix.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)
    inf_count = int(np.isinf(distance_matrix[off_diagonal]).sum())
    total = n * (n - 1)
    sparsity = inf_count / total if total else 0.0

    logger.info(f"  Matrix sparsity: {sparsity:.1%} of edges are infinite")

    return sparsity > GA.SPARSITY_THRESHOLD


# ==============================================================
# MAIN SOLVER
# ==============================================================


class r0123456:
    """
    Genetic Algorithm solver for TSP with adaptive strategies.

    Features:
        - Sparse-aware crossover and mutation
        - Hybrid local search (2-opt)
        - Large Neighborhood Search for escaping local optima
        - Self-adaptive mutation rates
        - Diversity preservation
    """

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.is_sparse = False

    def optimize(self, filename: str) -> int:
        """
        Main optimization routine.

        Args:
            filename: Path to CSV file containing distance matrix

        Returns:
            0 on successful completion
        """
        # Load problem instance
        problem = TravelingSalesmanProblem(
            self._read_distance_matrix(filename), os.path.basename(filename)
        )
        problem.print_info()

        # Initialize population
        print_section("INITIALIZATION")
        population = initialize_population_greedy_sparse_aware(
            problem, GA.POPULATION_SIZE
        )
        self._log_population_stats(population, "Initial Population")

        # Track best solution
        best_overall = min(population, key=lambda x: x.fitness)
        best_overall_fitness = best_overall.fitness
        last_improve_gen = 0
        stall_gens = 0

        # Evolution loop
        print_section("EVOLUTION")
        start_time = time.perf_counter()
        checkpoint = start_time

        for generation in range(1, GA.GENERATIONS + 1):
            # Generate offspring
            offspring = self._evolve_one_generation(
                population, problem, generation, best_overall_fitness
            )

            # Survivor selection
            population = elimination_with_crowding(
                population, offspring, GA.POPULATION_SIZE
            )

            # 🔥 Diversity injection (crank it up if collapsing)
            if (
                FEATURES.USE_DIVERSITY_INJECTION
                and generation % GA.DIVERSITY_CHECK_INTERVAL == 0
            ):
                injected = crank_diversity_injection(
                    problem,
                    population,
                    target_diversity=0.1,  # tune
                    replace_frac=0.6,
                    accept_avg_overlap_max=0.5,
                )

            # Track best solution
            gen_best = min(population, key=lambda x: x.fitness)
            gen_mean = float(np.mean([ind.fitness for ind in population]))

            # Check for improvement
            if gen_best.fitness < best_overall_fitness - 1e-9:
                best_overall = gen_best
                best_overall_fitness = gen_best.fitness
                last_improve_gen = generation
                stall_gens = 0
            else:
                stall_gens += 1

                # Boost mutation when stalled
                if FEATURES.USE_STAGNATION_MUTATION_BOOST and stall_gens % 50 == 0:
                    adaptive_mutation_stagnation(population, stall_gens)

            # Periodic logging
            if generation % GA.LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - checkpoint
                checkpoint = time.perf_counter()
                diversity = edge_diversity(population)

                logger.info(
                    f"  Gen {generation:4d} │ Mean: {gen_mean:12.2f} │ "
                    f"Best: {gen_best.fitness:12.2f} │ Div: {diversity:8.2%} │ "
                    f"Δt: {elapsed:7.2f}s │ NoImp: {stall_gens:4d} (last@{last_improve_gen:4d})"
                )

            # Apply LNS when stalled
            if (
                FEATURES.USE_LNS
                and stall_gens > GA.LNS_STALL_THRESHOLD
                and generation % GA.LNS_CHECK_INTERVAL == 0
            ):
                improved = self._apply_large_neighborhood_search(
                    problem, best_overall, best_overall_fitness, stall_gens, population
                )

                if improved is not None:
                    best_overall = improved
                    best_overall_fitness = improved.fitness
                    stall_gens = 0

            # Check time limit
            if self.reporter.report(gen_mean, gen_best.fitness, gen_best.tour) < 0:
                logger.info("\n  ⏱  Time limit reached")
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
        generation: int,
        best_overall_fitness: float,
    ) -> List[Individual]:
        """
        Generate offspring for one generation.

        Strategy adapts based on graph sparsity:
            - Dense graphs: standard OX crossover with adaptive mutation
            - Sparse graphs: ERX crossover with sparse-aware mutation
        """
        # Detect sparsity on first generation
        if generation == 1:
            detected_sparse = is_sparse_matrix(problem.distance_matrix)
            self.is_sparse = bool(detected_sparse and FEATURES.USE_SPARSE_STRATEGY)
            self._log_strategy()

        offspring: List[Individual] = []
        pop_best = min(population, key=lambda x: x.fitness).fitness

        # Check diversity periodically
        diversity = None
        if generation % GA.DIVERSITY_CHECK_INTERVAL == 0:
            diversity = edge_diversity(population)

        # Configure offspring generation based on sparsity
        if self.is_sparse:
            target = GA.OFFSPRING_SIZE
            max_attempts = target * GA.SPARSE_OFFSPRING_TARGET_MULTIPLIER
        else:
            target = GA.OFFSPRING_SIZE
            max_attempts = target * GA.PERTURBATION_MAX_ATTEMPTS_MULTIPLIER

        # Generate offspring
        attempts = 0
        while len(offspring) < target and attempts < max_attempts:
            attempts += 1

            # Select parents
            parent1 = tournament_selection(population, GA.TOURNAMENT_K)
            parent2 = tournament_selection(population, GA.TOURNAMENT_K)

            # Create child based on graph type
            child = self._create_offspring_dense(problem, parent1, parent2, diversity)

            # Store parent fitness for local search decisions
            child._p1_fitness = parent1.fitness
            child._p2_fitness = parent2.fitness

            # Only keep valid offspring
            if np.isfinite(child.fitness):
                offspring.append(child)

        # Clone if needed to maintain population stability
        cloned = self._ensure_minimum_offspring(offspring, population)

        # Log offspring generation statistics
        if generation % GA.LOG_SPARSE_OFFSPRING_STATS_INTERVAL == 0 and (
            len(offspring) < target or cloned
        ):
            logger.info(
                f"  Gen {generation}: Generated {len(offspring) - cloned}/{target} "
                f"offspring, cloned {cloned}"
            )

        # Apply local search to promising offspring
        if offspring and FEATURES.USE_LOCAL_SEARCH:
            apply_local_search_hybrid(
                offspring,
                problem,
                pop_best_fitness=pop_best,
                best_overall_fitness=best_overall_fitness,
            )

        return offspring

    def _create_offspring_sparse(
        self,
        problem: TravelingSalesmanProblem,
        parent1: Individual,
        parent2: Individual,
    ) -> Individual:
        """
        Create offspring for sparse graphs using edge-preserving operators.
        """
        # Use ERX occasionally, otherwise clone better parent
        if random.random() < GA.ERX_PROB_SPARSE:
            child = edge_recombination_crossover_sparse_aware(problem, parent1, parent2)
        else:
            better = parent1 if parent1.fitness < parent2.fitness else parent2
            child = Individual(
                tour=np.copy(better.tour), mutation_rate=better.mutation_rate
            )
            child.fitness = better.fitness

        # Apply sparse-aware mutation
        mutation_sparse_aware(child, problem)

        return child

    def _create_offspring_dense(
        self,
        problem: TravelingSalesmanProblem,
        parent1: Individual,
        parent2: Individual,
        diversity: Optional[float],
    ) -> Individual:
        """
        Create offspring for dense graphs using standard operators.
        """
        # Apply crossover or clone
        if random.random() < GA.CROSSOVER_PROB:
            child = order_crossover(problem, parent1, parent2)
            # or pmx, generation take way longer!! research others
        else:
            child = Individual(
                tour=np.copy(parent1.tour), mutation_rate=parent1.mutation_rate
            )

        # Adapt mutation rate if diversity was measured
        if FEATURES.USE_ADAPTIVE_MUTATION_RATE and diversity is not None:
            adaptive_mutation_rate(child, diversity)

        # Apply standard mutation
        if FEATURES.USE_MUTATION_PERTURB:
            mutation_perturb(child)
        elif FEATURES.USE_MUTATION_CLASSIC:
            mutation(child)

        child.evaluate(problem)

        return child

    def _ensure_minimum_offspring(
        self,
        offspring: List[Individual],
        population: List[Individual],
    ) -> int:
        """
        Ensure minimum offspring count by cloning if necessary.

        Returns:
            Number of clones added
        """
        min_offspring = max(GA.SPARSE_MIN_OFFSPRING_FACTOR, len(population))
        cloned = 0

        while len(offspring) < min_offspring and population:
            parent = random.choice(population)
            clone = Individual(
                tour=np.copy(parent.tour), mutation_rate=parent.mutation_rate
            )
            clone.fitness = parent.fitness
            clone._p1_fitness = clone._p2_fitness = parent.fitness
            offspring.append(clone)
            cloned += 1

        return cloned

    def _apply_large_neighborhood_search(
        self,
        problem: TravelingSalesmanProblem,
        best_individual: Individual,
        best_fitness: float,
        stall_gens: int,
        population: List[Individual],
    ) -> Optional[Individual]:
        """
        Apply Large Neighborhood Search to escape local optima.

        Destroys and rebuilds portions of the best tour, then applies
        local search to the reconstruction.

        Returns:
            Improved individual if found, None otherwise
        """
        logger.info(f"  Applying LNS (stalled for {stall_gens} generations)")

        best_lns = None
        best_lns_fitness = best_fitness

        # Try multiple LNS restarts
        for _ in range(GA.LNS_NUM_RESTARTS):
            candidate_tour = lns_destroy_repair(
                problem, best_individual.tour, destroy_fraction=GA.LNS_DESTROY_FRACTION
            )

            if candidate_tour is None:
                continue

            candidate = Individual(tour=candidate_tour)
            candidate.evaluate(problem)

            # Apply intensive local search to LNS result
            two_opt_local_search(candidate, problem, max_iters=GA.LNS_2OPT_ITERS)

            if candidate.fitness < best_lns_fitness:
                best_lns = candidate
                best_lns_fitness = candidate.fitness

        # Accept improvement or small uphill move when deeply stalled
        if best_lns is not None:
            delta = best_lns.fitness - best_fitness
            accept_uphill = (
                stall_gens > GA.LNS_ACCEPTANCE_STALL_THRESHOLD
                and delta < GA.LNS_ACCEPTANCE_UPHILL_DELTA
            )

            if delta < 0 or accept_uphill:
                population[0] = best_lns  # Replace worst in population
                logger.info(
                    f"    LNS {'improved' if delta < 0 else 'accepted'}: "
                    f"Δ = {delta:.2f}"
                )
                return best_lns

        return None

    def _log_strategy(self) -> None:
        """Log the selected strategy based on graph sparsity."""
        if self.is_sparse:
            logger.info("  🔍 Sparse matrix detected - using adapted strategy")
            logger.info(
                "  Strategy: Crowding + sparse-aware mutation + "
                "ERX crossover + hybrid local search"
            )
        else:
            logger.info("  Strategy: Standard GA + hybrid local search + LNS")

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


def crank_diversity_injection(
    problem: TravelingSalesmanProblem,
    population: List[Individual],
    target_diversity: float = 0.015,  # tune: your edge_diversity is tiny for big n
    replace_frac: float = 0.20,  # replace worst 20% (attempted)
    sample_k: int = 12,  # overlap check sample size
    accept_avg_overlap_max: float = 0.35,  # accept if child shares <=35% edges on avg
    max_attempts_per_slot: int = 50,  # tries to find a diverse valid tour
) -> int:
    """
    If population diversity is low, replace some WORST individuals with
    injected tours that have low directed-edge overlap with population.

    Returns: number of replacements actually made.
    """
    if not population:
        return 0

    cur_div = edge_diversity(population)
    if cur_div >= target_diversity:
        return 0

    # Sort best->worst (we replace from the end)
    population.sort(key=lambda ind: ind.fitness)

    pop_size = len(population)
    n = population[0].tour.shape[0]

    # Build successor matrix for fast overlap checks
    pop_succs = np.empty((pop_size, n), dtype=np.int32)
    for i, ind in enumerate(population):
        pop_succs[i, :] = build_successor(ind.tour)

    # Decide how many to try to replace (from worst upwards)
    num_slots = max(1, int(pop_size * replace_frac))
    replaced = 0

    # Indices we will consider "victims": worst individuals
    victim_indices = list(range(pop_size - num_slots, pop_size))

    for victim_idx in victim_indices:
        # Try multiple attempts to generate a diverse *valid* tour
        for _ in range(max_attempts_per_slot):
            # Prefer injecting from GOOD solutions but with STRONG perturbation
            parent = random.choice(population[: max(5, pop_size // 3)])  # top third
            new_tour = apply_perturbation(parent.tour, "heavy")

            # Repair if needed (sparse)
            if np.isinf(evaluate_tour_numba(new_tour, problem.penalized_matrix)):
                repaired = repair_tour(problem, new_tour)
                if repaired is None:
                    continue
                new_tour = repaired

            # Evaluate candidate
            cand = Individual(tour=new_tour, mutation_rate=parent.mutation_rate)
            cand.evaluate(problem)
            if not np.isfinite(cand.fitness):
                continue

            # Overlap check against a random sample of population
            k = min(sample_k, pop_size)
            sample_idxs = np.array(random.sample(range(pop_size), k), dtype=np.int32)
            cand_succ = build_successor(cand.tour)
            avg_ov = float(
                avg_overlap_ratio_with_sample(cand_succ, pop_succs, sample_idxs)
            )

            if avg_ov <= accept_avg_overlap_max:
                # Accept: replace the victim
                population[victim_idx] = cand
                pop_succs[victim_idx, :] = cand_succ
                replaced += 1
                break

    # Keep population sorted for downstream selection logic
    population.sort(key=lambda ind: ind.fitness)
    return replaced


@numba.njit(cache=True)
def directed_edge_overlap_count(succ_a: np.ndarray, succ_b: np.ndarray) -> int:
    """Count shared directed edges between two tours in successor form."""
    n = succ_a.shape[0]
    c = 0
    for i in range(n):
        if succ_a[i] == succ_b[i]:
            c += 1
    return c


@numba.njit(cache=True)
def avg_overlap_ratio_with_sample(
    child_succ: np.ndarray,
    pop_succs: np.ndarray,
    sample_idxs: np.ndarray,
) -> float:
    """
    Average overlap ratio (0..1) between child and a sample of population.
    """
    n = child_succ.shape[0]
    s = 0.0
    for k in range(sample_idxs.shape[0]):
        idx = sample_idxs[k]
        s += directed_edge_overlap_count(child_succ, pop_succs[idx]) / n
    return s / sample_idxs.shape[0]


# ==============================================================
# MAIN ENTRY POINT
# ==============================================================


def evaluate_tour_from_csv_string(
    distance_csv_path: str,
    tour_csv_string: str,
) -> float:
    """
    Evaluate tour length where tour is provided as a CSV string.

    Example tour_csv_string: "0,5,2,7,1,3,4,6"
    """
    distance_matrix = np.loadtxt(distance_csv_path, delimiter=",")
    tour = np.fromstring(tour_csv_string, sep=",", dtype=int)

    problem = TravelingSalesmanProblem(distance_matrix)
    return evaluate_tour_numba(tour, problem.distance_matrix)


if __name__ == "__main__":
    dist_csv = "src/benchmark/tour750.csv"
    tour_csv = "92,446,37,254,396,229,323,472,296,362,480,649,586,90,426,720,143,67,674,514,265,691,234,193,646,504,245,463,579,351,214,604,132,46,484,661,680,40,736,518,675,225,43,567,454,131,177,590,304,62,172,176,605,298,117,50,212,714,505,194,699,507,73,55,443,520,220,34,31,113,237,57,119,636,491,373,692,133,153,739,89,20,91,727,24,553,407,663,191,599,3,623,530,218,164,738,215,497,703,83,717,558,243,244,232,408,106,552,197,64,32,203,283,189,27,702,21,523,316,495,345,412,537,474,12,540,460,68,201,724,689,693,103,647,568,75,458,301,705,314,654,365,291,151,15,77,332,591,743,668,701,181,733,485,109,248,584,148,145,667,595,233,696,343,105,451,528,239,594,746,144,626,300,198,449,625,442,141,587,7,134,394,9,585,716,184,427,421,327,742,487,348,452,51,563,608,157,683,430,659,434,517,321,310,317,609,108,288,428,66,159,534,289,231,393,258,457,488,747,251,359,531,559,278,741,629,719,439,49,69,635,502,554,640,149,331,58,36,477,459,82,190,468,435,200,270,284,271,209,45,180,380,735,631,299,473,573,501,260,54,600,175,121,128,61,236,606,387,549,39,41,481,207,498,557,135,222,614,146,102,130,644,170,1,542,409,202,110,6,728,564,311,492,361,281,476,676,124,493,276,383,581,519,732,87,71,539,432,238,541,178,16,571,65,455,388,688,282,308,302,211,536,76,114,274,402,360,379,598,42,628,358,619,545,2,417,166,96,438,706,565,127,666,651,292,85,23,471,242,722,319,633,744,35,47,224,371,526,167,489,38,208,346,610,147,216,726,320,335,700,386,704,592,374,513,324,697,368,257,617,397,363,494,686,677,18,414,561,400,154,580,328,17,441,401,369,384,86,470,285,466,548,188,333,186,652,120,411,112,521,690,235,344,376,749,169,707,370,624,653,615,506,576,419,510,632,246,694,59,382,195,249,79,662,729,570,252,30,630,129,596,543,656,511,469,219,535,404,496,179,698,60,329,48,483,679,657,582,734,424,440,464,713,84,395,643,4,685,538,204,410,74,399,650,524,533,500,28,627,721,660,221,486,5,588,453,711,347,355,745,115,601,616,93,572,682,253,22,338,378,94,279,259,315,574,391,342,611,621,444,171,695,708,294,475,160,352,290,456,357,478,168,142,174,162,658,318,593,330,490,509,390,99,227,275,583,44,155,671,437,566,267,673,665,366,429,562,78,230,158,715,392,138,196,577,681,687,433,367,269,748,569,206,26,508,415,532,205,479,403,389,642,312,262,163,462,306,405,418,192,337,710,98,560,313,637,56,136,118,293,295,527,709,353,217,655,684,126,272,546,95,123,280,372,0,544,634,622,63,712,731,223,450,104,261,551,669,529,25,381,467,336,645,213,522,613,137,307,247,597,122,325,445,639,406,210,718,125,638,737,165,515,139,375,150,161,241,341,555,81,648,297,8,385,482,33,11,730,664,339,356,603,461,413,70,255,266,575,286,448,334,422,183,512,423,52,612,303,268,322,287,350,111,100,273,420,602,13,101,152,182,187,672,326,589,226,72,107,305,88,97,185,53,354,556,264,173,525,29,516,607,425,670,465,240,618,256,398,723,19,263,277,740,377,364,620,578,228,80,349,199,550,499,340,309,431,156,250,10,547,447,416,14,725,678,116,436,503,140,641"

    length = evaluate_tour_from_csv_string(dist_csv, tour_csv)
    print(f"Tour length: {length:.2f}")
