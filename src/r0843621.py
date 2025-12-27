import Reporter
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    # ==========================================================
    # EXECUTION / BACKEND
    # ==========================================================
    USE_NUMBA: bool = True

    # ==========================================================
    # POPULATION & GENERATIONAL MODEL (μ, λ)
    # ==========================================================
    POPULATION_SIZE: int = 100  # λ
    OFFSPRING_SIZE: int = 200  # μ
    GENERATIONS: int = 10_000_000

    # ==========================================================
    # INITIALIZATION OPERATOR (Greedy + Random)
    # ==========================================================
    GREEDY_FRACTION: float = 0.01  # Fraction of population initialized with greedy
    GREEDY_ATTEMPTS_MULTIPLIER: int = 200  # Try this many greedy starts per target slot

    # ==========================================================
    # SELECTION OPERATOR
    # ==========================================================
    TOURNAMENT_K: int = 2  # Tournament size (larger = more selective)

    # ==========================================================
    # CROSSOVER OPERATOR
    # ==========================================================
    CROSSOVER_PROB: float = 1

    # ==========================================================
    # MUTATION OPERATOR (Self-adaptive mutation rate bounds)
    # ==========================================================
    MUTATION_ALPHA_MIN: float = 0.08
    MUTATION_ALPHA_MAX: float = 0.20

    # ==========================================================
    # SURVIVOR SELECTION / REPLACEMENT OPERATOR
    # ==========================================================
    STALL_LIMIT: int = 5000

    # ==========================================================
    # FITNESS EVALUATION / CONSTRAINT HANDLING
    # ==========================================================
    USE_PENALTY_FOR_INF: bool = True
    INF_PENALTY_FACTOR: float = 10

    # ==========================================================
    # LOCAL SEARCH OPERATORS (Or-opt)
    # ==========================================================
    USE_OR_OPT: bool = True  # Enable / disable Or-opt local search
    OR_OPT_MAX_ITERS: int = 10  # Max iterations for Or-opt
    OR_OPT_MAX_CHAIN_LEN: int = 7  # Maximum chain length (1-3)

    # Local search scheduling (GA-driven)
    LS_INTERVAL: int = 100  # apply LS every N generations
    LS_TOP_K: int = 1  # apply LS to top K individuals
    LS_MODE: int = 1  # 0=first improvement, 1=best improvement for 2-opt

    # ==========================================================
    # DIVERSITY CONTROL / MONITORING
    # ==========================================================
    DIVERSITY_MIN: float = 0.08  # 8%
    DIVERSITY_COOLDOWN: int = 200  # minimum gens between injections

    # ==========================================================
    # BENCHMARK / REFERENCE TARGETS
    # ==========================================================
    HEURISTIC = 72418.75  # todo automatic please
    HEUR_WITHIN_PCT = 0.15

    # ==========================================================
    # LOGGING / REPORTING
    # ==========================================================
    LOG_INTERVAL: int = 10
    SPARSITY_THRESHOLD: float = 0.10


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

        # Detect sparsity
        self.is_sparse = self._is_sparse_matrix(distance_matrix)

        # Penalized matrix: replace Inf edges with large penalty
        self.penalized_matrix = self._build_penalized_matrix(distance_matrix)

    def _is_sparse_matrix(self, distance_matrix: np.ndarray) -> bool:
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

        return sparsity > GA.SPARSITY_THRESHOLD

    def _build_penalized_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Build distance matrix with penalties for missing edges (Inf)."""
        penalized = distance_matrix.copy()

        if not GA.USE_PENALTY_FOR_INF or not np.isinf(distance_matrix).any():
            return penalized

        finite_mask = np.isfinite(distance_matrix)
        max_finite = (
            float(np.max(distance_matrix[finite_mask])) if finite_mask.any() else 1.0
        )
        penalty = (max_finite + 1.0) * GA.INF_PENALTY_FACTOR

        # Penalize all missing edges
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
            "Sparse Graph": "Yes" if self.is_sparse else "No",
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
        self.fitness = evaluate_tour(self.tour, problem.penalized_matrix)
        return self.fitness


def evaluate_tour(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    return (
        evaluate_tour_numba(tour, distance_matrix)
        if GA.USE_NUMBA
        else evaluate_tour_py(tour, distance_matrix)
    )


def evaluate_tour_py(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute total tour length efficiently using Python/NumPy.
    Returns np.inf if any edge is invalid.
    """
    total = 0.0
    n = tour.shape[0]

    for i in range(n):
        city_a = int(tour[i])
        city_b = int(tour[(i + 1) % n])
        distance = float(distance_matrix[city_a, city_b])

        if np.isinf(distance):
            return np.inf

        total += distance

    return total


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
# POPULATION INITIALIZATION
# ==============================================================


def nearest_neighbor_greedy(
    distance_matrix: np.ndarray,
    start_city: int,
) -> Optional[np.ndarray]:
    tour = (
        nearest_neighbor_greedy_numba(distance_matrix, start_city)
        if GA.USE_NUMBA
        else nearest_neighbor_greedy_py(distance_matrix, start_city)
    )

    if tour[0] == -1:
        return None

    return tour


def nearest_neighbor_greedy_py(
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
        tour = nearest_neighbor_greedy(problem.distance_matrix, start_city)
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

    pop_succs = np.empty((len(survivors), survivors[0].tour.shape[0]), dtype=np.int32)
    pop_fits = np.empty(len(survivors), dtype=np.float64)

    for i, ind in enumerate(survivors):
        pop_succs[i, :] = build_successor(ind.tour)
        pop_fits[i] = float(ind.fitness)

    # Process each offspring
    for child in offspring:
        if child.fitness is None or not np.isfinite(child.fitness):
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


def build_successor(tour: np.ndarray) -> np.ndarray:
    return build_successor_numba(tour) if GA.USE_NUMBA else build_successor_py(tour)


def build_successor_py(tour: np.ndarray) -> np.ndarray:
    """
    Build successor representation: succ[city] = next_city in tour.
    Used for fast edge overlap computation.
    """
    n = tour.shape[0]
    succ = np.empty(n, dtype=np.int32)
    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])
        succ[a] = b
    return succ


@numba.njit(cache=True)
def build_successor_numba(tour: np.ndarray) -> np.ndarray:
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


def best_rival_index_by_edge_overlap(
    child_succ: np.ndarray,
    pop_succs: np.ndarray,
    child_fit: float,
    pop_fits: np.ndarray,
) -> int:
    return (
        best_rival_index_by_edge_overlap_numba(
            child_succ, pop_succs, child_fit, pop_fits
        )
        if GA.USE_NUMBA
        else best_rival_index_by_edge_overlap_py(
            child_succ, pop_succs, child_fit, pop_fits
        )
    )


def best_rival_index_by_edge_overlap_py(
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

        fit_diff = abs(child_fit - float(pop_fits[i]))

        if (overlap > best_overlap) or (
            overlap == best_overlap and fit_diff < best_fit_diff
        ):
            best_overlap = overlap
            best_fit_diff = fit_diff
            best_idx = i

    return best_idx


@numba.njit(cache=True)
def best_rival_index_by_edge_overlap_numba(
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
# OR-OPT LOCAL SEARCH
# ==============================================================


def or_opt_local_search(
    individual: Individual,
    problem: TravelingSalesmanProblem,
    max_iters: int = 30,
) -> Tuple[float, int]:
    if individual.fitness is None:
        individual.evaluate(problem)

    before = float(individual.fitness)

    tour = individual.tour
    distance_matrix = problem.distance_matrix
    feasible = problem.feasible

    moves = 0
    iteration = 0
    improved = True

    while iteration < max_iters and improved:
        iteration += 1
        improved = False

        start, length, insert_after_city = or_opt_improvement(
            tour, distance_matrix, feasible, max_chain_len=GA.OR_OPT_MAX_CHAIN_LEN
        )

        if start != -1:
            tour[:] = apply_or_opt_move(tour, start, length, insert_after_city)
            improved = True
            moves += 1

    individual.evaluate(problem)
    after = float(individual.fitness)
    return before - after, moves


def or_opt_improvement(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    feasible: np.ndarray,
    max_chain_len: int = 3,
) -> Tuple[int, int, int]:
    return (
        or_opt_improvement_numba(
            tour, distance_matrix, feasible, max_chain_len=max_chain_len
        )
        if GA.USE_NUMBA
        else or_opt_improvement_py(
            tour, distance_matrix, feasible, max_chain_len=max_chain_len
        )
    )


def or_opt_improvement_py(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    feasible: np.ndarray,
    max_chain_len: int = 3,
) -> Tuple[int, int, int]:
    """
    Returns (start_index, chain_len, insert_after_city)
    where insert_after_city is the CITY id (not an index).
    """
    n = tour.shape[0]
    best_improvement = 0.0
    best_start = -1
    best_length = 0
    best_insert_after_city = -1

    # Try chains of length 1..max_chain_len
    for chain_len in range(1, min(max_chain_len + 1, n - 1)):
        for start in range(n):
            # Build a small removal mask for this chain (wrap-safe)
            remove = np.zeros(n, dtype=np.bool_)
            for k in range(chain_len):
                remove[(start + k) % n] = True

            end = (start + chain_len - 1) % n

            before_chain = int(tour[(start - 1) % n])
            after_chain = int(tour[(end + 1) % n])
            first_in_chain = int(tour[start])
            last_in_chain = int(tour[end])

            # Need to be able to connect before_chain -> after_chain after removal
            if not feasible[before_chain, after_chain]:
                continue

            cost_removed = float(
                distance_matrix[before_chain, first_in_chain]
                + distance_matrix[last_in_chain, after_chain]
            )
            cost_bridge = float(distance_matrix[before_chain, after_chain])

            # Try inserting between (insert_pos, insert_pos+1)
            for insert_pos in range(n):
                # before_insert and after_insert are the edge you break to insert the chain
                if remove[insert_pos] or remove[(insert_pos + 1) % n]:
                    continue  # don't insert "inside" or adjacent to the removed chain

                before_insert = int(tour[insert_pos])
                after_insert = int(tour[(insert_pos + 1) % n])

                if not feasible[before_insert, first_in_chain]:
                    continue
                if not feasible[last_in_chain, after_insert]:
                    continue

                cost_removed_at_insert = float(
                    distance_matrix[before_insert, after_insert]
                )
                cost_added_at_insert = float(
                    distance_matrix[before_insert, first_in_chain]
                    + distance_matrix[last_in_chain, after_insert]
                )

                if np.isinf(cost_removed):
                    improvement = 1e10
                else:
                    improvement = (
                        cost_removed
                        + cost_removed_at_insert
                        - cost_bridge
                        - cost_added_at_insert
                    )

                if improvement > best_improvement + 1e-9:
                    best_improvement = improvement
                    best_start = start
                    best_length = chain_len
                    best_insert_after_city = before_insert  # <-- CITY, not index

    return best_start, best_length, best_insert_after_city


@numba.njit(cache=True)
def or_opt_improvement_numba(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    feasible: np.ndarray,
    max_chain_len: int = 3,
) -> Tuple[int, int, int]:
    """
    Returns (start_index, chain_len, insert_after_city)
    where insert_after_city is the CITY id (not an index).
    """
    n = tour.shape[0]
    best_improvement = 0.0
    best_start = -1
    best_length = 0
    best_insert_after_city = -1

    # Try chains of length 1..max_chain_len
    for chain_len in range(1, min(max_chain_len + 1, n - 1)):
        for start in range(n):
            # Build a small removal mask for this chain (wrap-safe)
            remove = np.zeros(n, dtype=np.bool_)
            for k in range(chain_len):
                remove[(start + k) % n] = True

            end = (start + chain_len - 1) % n

            before_chain = tour[(start - 1) % n]
            after_chain = tour[(end + 1) % n]
            first_in_chain = tour[start]
            last_in_chain = tour[end]

            # Need to be able to connect before_chain -> after_chain after removal
            if not feasible[before_chain, after_chain]:
                continue

            cost_removed = (
                distance_matrix[before_chain, first_in_chain]
                + distance_matrix[last_in_chain, after_chain]
            )
            cost_bridge = distance_matrix[before_chain, after_chain]

            # Try inserting between (insert_pos, insert_pos+1)
            for insert_pos in range(n):
                # before_insert and after_insert are the edge you break to insert the chain
                if remove[insert_pos] or remove[(insert_pos + 1) % n]:
                    continue  # don't insert "inside" or adjacent to the removed chain

                before_insert = tour[insert_pos]
                after_insert = tour[(insert_pos + 1) % n]

                if not feasible[before_insert, first_in_chain]:
                    continue
                if not feasible[last_in_chain, after_insert]:
                    continue

                cost_removed_at_insert = distance_matrix[before_insert, after_insert]
                cost_added_at_insert = (
                    distance_matrix[before_insert, first_in_chain]
                    + distance_matrix[last_in_chain, after_insert]
                )

                if np.isinf(cost_removed):
                    improvement = 1e10
                else:
                    improvement = (
                        cost_removed
                        + cost_removed_at_insert
                        - cost_bridge
                        - cost_added_at_insert
                    )

                if improvement > best_improvement + 1e-9:
                    best_improvement = improvement
                    best_start = start
                    best_length = chain_len
                    best_insert_after_city = before_insert  # <-- CITY, not index

    return best_start, best_length, best_insert_after_city


def apply_or_opt_move(
    tour: np.ndarray, start: int, length: int, insert_after_city: int
) -> np.ndarray:
    return (
        apply_or_opt_move_numba(tour, start, length, insert_after_city)
        if GA.USE_NUMBA
        else apply_or_opt_move_py(tour, start, length, insert_after_city)
    )


def apply_or_opt_move_py(
    tour: np.ndarray, start: int, length: int, insert_after_city: int
) -> np.ndarray:
    n = tour.shape[0]
    dt = tour.dtype  # <-- keep everything consistent

    # Mark which POSITIONS are removed (wrap-safe)
    remove = np.zeros(n, dtype=np.bool_)
    for k in range(length):
        remove[(start + k) % n] = True

    # Extract chain in order (wrap-safe)
    chain = np.empty(length, dtype=dt)
    for k in range(length):
        chain[k] = tour[(start + k) % n]

    # Build remaining tour (preserve order)
    remaining = np.empty(n - length, dtype=dt)
    r = 0
    for i in range(n):
        if not remove[i]:
            remaining[r] = tour[i]
            r += 1

    # Find insertion point in remaining: after the CITY insert_after_city
    ins_idx = -1
    for i in range(remaining.shape[0]):
        if remaining[i] == insert_after_city:
            ins_idx = i
            break

    # Safety fallback (must return same dtype as other branch)
    if ins_idx == -1:
        return tour.copy()

    # Build new tour by inserting chain AFTER ins_idx
    new_tour = np.empty(n, dtype=dt)
    p = 0
    for i in range(remaining.shape[0]):
        new_tour[p] = remaining[i]
        p += 1
        if i == ins_idx:
            for k in range(length):
                new_tour[p] = chain[k]
                p += 1

    return new_tour


# not very computatioanlly heavy!!
@numba.njit(cache=True)
def apply_or_opt_move_numba(
    tour: np.ndarray, start: int, length: int, insert_after_city: int
) -> np.ndarray:
    n = tour.shape[0]
    dt = tour.dtype  # <-- keep everything consistent

    # Mark which POSITIONS are removed (wrap-safe)
    remove = np.zeros(n, dtype=np.bool_)
    for k in range(length):
        remove[(start + k) % n] = True

    # Extract chain in order (wrap-safe)
    chain = np.empty(length, dtype=dt)
    for k in range(length):
        chain[k] = tour[(start + k) % n]

    # Build remaining tour (preserve order)
    remaining = np.empty(n - length, dtype=dt)
    r = 0
    for i in range(n):
        if not remove[i]:
            remaining[r] = tour[i]
            r += 1

    # Find insertion point in remaining: after the CITY insert_after_city
    ins_idx = -1
    for i in range(remaining.shape[0]):
        if remaining[i] == insert_after_city:
            ins_idx = i
            break

    # Safety fallback (must return same dtype as other branch)
    if ins_idx == -1:
        return tour.copy()

    # Build new tour by inserting chain AFTER ins_idx
    new_tour = np.empty(n, dtype=dt)
    p = 0
    for i in range(remaining.shape[0]):
        new_tour[p] = remaining[i]
        p += 1
        if i == ins_idx:
            for k in range(length):
                new_tour[p] = chain[k]
                p += 1

    return new_tour


# ==============================================================
# COMBINED LOCAL SEARCH (just Or-Opt, 2-opt wasn't doing much)
# ==============================================================


def combined_local_search(
    individual: Individual, problem: TravelingSalesmanProblem
) -> float:
    if individual.fitness is None:
        individual.evaluate(problem)

    total_before = float(individual.fitness)

    # Stage 1: Or-opt
    if GA.USE_OR_OPT:
        d, moves = or_opt_local_search(
            individual, problem, max_iters=GA.OR_OPT_MAX_ITERS
        )
        if d > 1e-9:
            logger.info(f"    LS: Or-opt improved by {d:.2f} using {moves} moves")

    total_after = float(individual.fitness)
    return total_before - total_after


# ==============================================================
# DIVERSITY MEASUREMENT (for logging)
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
        - Combined local search (2-opt + Or-opt)
    """

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.last_injection_gen = -(10**9)

    def optimize(self, filename: str) -> int:
        """Main optimization routine."""
        problem = TravelingSalesmanProblem(
            self._read_distance_matrix(filename), os.path.basename(filename)
        )
        problem.print_info()

        # Initialize population
        print_section("INITIALIZATION")
        population = initialize_population_mixed(problem, GA.POPULATION_SIZE)

        self._log_population_stats(population, "Initial Population")
        self.best_overall = min(population, key=lambda x: x.fitness)
        self.stall_gens = 0
        start_time = checkpoint = time.perf_counter()

        # Evolution loop
        print_section("EVOLUTION")
        for generation in range(0, GA.GENERATIONS):
            # Evolve
            offspring = self._evolve_one_generation(population, problem)
            population = elimination_with_crowding(
                population, offspring, GA.POPULATION_SIZE
            )

            # Local search & Track best
            gen_best = min(population, key=lambda x: x.fitness)
            self._process_best_solution(gen_best, problem, generation)

            if GA.LS_INTERVAL > 0 and generation % GA.LS_INTERVAL == 0:
                self._apply_local_search_top_k(population, problem)

            # Logging
            if generation % GA.LOG_INTERVAL == 0:
                div = edge_diversity(population)
                checkpoint = self._log_progress(generation, population, checkpoint, div)
                # self._log_top5_except_best(population)

            # Report and check time limit
            gen_mean = float(np.mean([ind.fitness for ind in population]))
            if (
                self.reporter.report(
                    gen_mean, self.best_overall.fitness, self.best_overall.tour
                )
                < 0
            ):
                logger.info("\nTime limit reached")
                break

            if self.stall_gens >= GA.STALL_LIMIT:
                logger.info(
                    f"\nStopping: no improvement for {GA.STALL_LIMIT} generations."
                )
                break

        self._final_report(generation, population, start_time)
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

    def _process_best_solution(
        self, gen_best: Individual, problem: TravelingSalesmanProblem, generation: int
    ):
        """Update global best and apply local search improvement."""
        # Check if this is a new best
        if gen_best.fitness < self.best_overall.fitness - 1e-9:
            if GA.USE_OR_OPT:
                candidate = Individual(
                    tour=np.copy(gen_best.tour), mutation_rate=gen_best.mutation_rate
                )
                candidate.fitness = gen_best.fitness

                improvement = combined_local_search(candidate, problem)

                if improvement > 1e-9:
                    logger.info(
                        f"Or-opt improved best by {improvement:.2f} (Gen {generation})"
                    )
                    gen_best.tour = np.copy(candidate.tour)
                    gen_best.fitness = candidate.fitness

            self.best_overall = Individual(
                tour=np.copy(gen_best.tour), mutation_rate=gen_best.mutation_rate
            )
            self.best_overall.fitness = gen_best.fitness
            self.stall_gens = 0
        else:
            self.stall_gens += 1

    def _apply_local_search_top_k(
        self, population: List[Individual], problem: TravelingSalesmanProblem
    ) -> None:
        """Apply local search to the top-K individuals (in-place improvements)."""
        if not GA.USE_OR_OPT:
            return

        pop_sorted = sorted(population, key=lambda ind: ind.fitness)
        k = min(GA.LS_TOP_K, len(pop_sorted))

        for idx in range(k):
            ind = pop_sorted[idx]

            # work on a copy so we only accept improvements
            cand = Individual(tour=np.copy(ind.tour), mutation_rate=ind.mutation_rate)
            cand.fitness = ind.fitness

            improvement = combined_local_search(cand, problem)

            if improvement > 1e-9 and cand.fitness < ind.fitness:
                ind.tour = cand.tour.copy()
                ind.fitness = cand.fitness

    def _log_progress(
        self,
        generation: int,
        population: List[Individual],
        checkpoint: float,
        div: float,
    ) -> float:
        """Log generation statistics."""
        elapsed = time.perf_counter() - checkpoint
        gen_best = min(population, key=lambda x: x.fitness)
        gen_mean = float(np.mean([ind.fitness for ind in population]))
        gen_worst = max(population, key=lambda x: x.fitness)

        beat, within = count_vs_heuristic(population, GA.HEURISTIC, GA.HEUR_WITHIN_PCT)

        logger.info(
            f"  Gen {generation:4d} │ Mean: {gen_mean:12.2f} │ "
            f"Best: {gen_best.fitness:12.2f} │ Worst: {gen_worst.fitness:12.2f} │ "
            f"Div: {div:8.2%} │ "
            f"≤H: {beat:3d} │ ≤1.15H: {within:3d} │ "
            f"Δt: {elapsed:7.2f}s │ NoImp: {self.stall_gens:4d}"
        )
        return time.perf_counter()

    def _final_report(
        self, generation: int, population: List[Individual], start_time: float
    ):
        """Print final optimization summary."""
        total_time = time.perf_counter() - start_time
        print_section("RESULTS")
        print_stats_table(
            {
                "Best Fitness": self.best_overall.fitness,
                "Generations": generation,
                "Total Time (s)": total_time,
                "Final Diversity": edge_diversity(population),
            }
        )

    def _log_top5_except_best(self, population: List[Individual]) -> None:
        """Log ranks 2..6 (top 5 excluding the best)."""
        pop_sorted = sorted(population, key=lambda ind: ind.fitness)
        if len(pop_sorted) < 2:
            return

        best = pop_sorted[0].fitness
        upto = min(6, len(pop_sorted))  # ranks 1..6 exist -> print 2..upto

        logger.info("    Top-5 (excluding #1):")
        for rank in range(2, upto + 1):
            f = pop_sorted[rank - 1].fitness
            gap = f - best
            logger.info(f"      #{rank:>2}: {f:12.2f}  (gap +{gap:10.2f})")


def count_vs_heuristic(
    population: List[Individual], heuristic: float, within_pct: float = 0.15
) -> Tuple[int, int]:
    beat = 0
    within = 0
    thresh = (1.0 + within_pct) * heuristic

    for ind in population:
        f = ind.fitness
        if f is None or not np.isfinite(f):
            continue
        if f <= thresh:
            within += 1
            if f <= heuristic:
                beat += 1

    return beat, within
