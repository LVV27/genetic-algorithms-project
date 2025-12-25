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
        7  # Tournament size for parent selection (larger = more selective)
    )

    # ===== Mutation Parameters =====
    # Self-adaptive mutation rate bounds
    MUTATION_ALPHA_MIN: float = 0.04  # Minimum mutation rate (exploitation)
    MUTATION_ALPHA_MAX: float = 0.12  # Maximum mutation rate (exploration)
    DIVERSITY_CHECK_INTERVAL: int = (
        5  # How often to measure diversity for adaptive mutation
    )

    # ===== Crossover Parameters =====
    CROSSOVER_PROB: float = 0.8  # Probability of crossover vs. cloning
    ERX_PROB_SPARSE: float = 0.15  # Probability of using ERX in sparse graphs

    # ===== Initialization Parameters =====
    GREEDY_SEED_COUNT: int = 5  # Number of best greedy solutions to seed population
    GREEDY_RESTARTS: int = (
        100  # Number of different start cities for greedy construction
    )

    # ===== Local Search Parameters =====
    LOCAL_SEARCH_ENABLED: bool = True
    LOCAL_SEARCH_MAX_ITERS: int = 1  # Max iterations per 2-opt application

    # When to apply 2-opt (selective to save computation)
    LSO_APPLY_IF_BEATS_ANY_PARENT: bool = True  # Apply if offspring beats a parent
    LSO_NEAR_BEST_FRAC: float = 0.01  # Apply if within 1% of best
    LSO_ALWAYS_IMPROVE_TOP_K: int = 2  # Always apply to top K offspring
    LSO_LOG_COUNTS: bool = False  # Log how many get local search

    # ===== Sparse Graph Strategy =====
    SPARSITY_THRESHOLD: float = 0.1  # Fraction of inf edges to trigger sparse mode
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
    LNS_STALL_THRESHOLD: int = 80  # Apply LNS after this many stagnant generations
    LNS_CHECK_INTERVAL: int = 20  # Check every N generations when stalled
    LNS_NUM_RESTARTS: int = 20  # Number of LNS restart attempts
    LNS_2OPT_ITERS: int = 30  # 2-opt iterations after LNS reconstruction
    LNS_DESTROY_FRACTION: float = 0.4  # Fraction of tour to destroy/rebuild
    LNS_ACCEPTANCE_UPHILL_DELTA: int = (
        100  # Accept uphill moves up to this delta when stalled
    )
    LNS_ACCEPTANCE_STALL_THRESHOLD: int = 200  # Only accept uphill if stalled this long

    # ===== Logging Parameters =====
    LOG_INTERVAL: int = 10  # Print statistics every N generations
    LOG_SPARSE_OFFSPRING_STATS_INTERVAL: int = 20  # Log offspring generation stats


GA = GAParams()


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

    def __init__(
        self,
        distance_matrix: np.ndarray,
        filename: Optional[str] = None,
    ):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.filename = filename

        # Precompute feasibility matrix for faster sparse graph handling
        self.feasible = np.isfinite(distance_matrix)

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
        Evaluate tour fitness using Numba-accelerated distance calculation.

        Returns:
            Total tour length, or np.inf if tour contains invalid edges
        """
        self.fitness = evaluate_tour_numba(self.tour, problem.distance_matrix)
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


def edge_diversity(population: List[Individual]) -> float:
    """
    Measure edge-level diversity across the population.

    Counts unique edges used across all tours (treating edges as undirected).
    Higher values indicate more diverse routing patterns.

    Returns:
        Fraction of possible edge occurrences that are unique
    """
    if not population:
        return 0.0

    edge_sets = []
    n = len(population[0].tour)

    for ind in population:
        tour = ind.tour
        edges = set()
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            # Store both directions (undirected edge)
            edges.add((a, b))
            edges.add((b, a))
        edge_sets.append(edges)

    # Union of all edges used
    union_edges = set.union(*edge_sets)

    # Maximum possible: each individual uses n edges * 2 directions
    max_edges = len(population) * n * 2

    return len(union_edges) / max_edges


def find_most_similar(
    individual: Individual,
    population: List[Individual],
) -> Optional[Individual]:
    """
    Find the individual in the population with closest fitness value.

    This is a computationally cheap proxy for tour similarity,
    used in deterministic crowding for diversity preservation.
    """
    if not population:
        return None

    fitness_differences = [
        abs(individual.fitness - other.fitness) for other in population
    ]

    most_similar_index = int(np.argmin(fitness_differences))
    return population[most_similar_index]


# ==============================================================
# POPULATION INITIALIZATION
# ==============================================================


def perturb_double_bridge(tour: np.ndarray) -> np.ndarray:
    """
    Apply double-bridge perturbation (effective for escaping local optima).

    Splits tour into 4 segments and reconnects them in a different order.
    """
    n = len(tour)
    if n < 8:
        return tour.copy()

    # Choose 3 cut points to create 4 segments
    p1, p2, p3 = sorted(random.sample(range(1, n), 3))
    return np.concatenate([tour[:p1], tour[p2:p3], tour[p1:p2], tour[p3:]])


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
    num_cities = problem.num_cities
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)

    tour: List[int] = [start_city]
    current_city = start_city

    while unvisited:
        best_next_city: Optional[int] = None
        best_distance = float("inf")

        # Find nearest unvisited city
        for candidate in unvisited:
            distance = problem.get_distance(current_city, candidate)
            if distance < best_distance:
                best_distance = distance
                best_next_city = candidate

        # If no valid edge exists, construction fails
        if best_next_city is None or np.isinf(best_distance):
            return None

        tour.append(best_next_city)
        unvisited.remove(best_next_city)
        current_city = best_next_city

    return np.asarray(tour, dtype=int)


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
    max_restarts = 5
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
    Deterministic crowding survivor selection.

    Each offspring competes with its most similar individual in the
    current population (similarity measured by fitness distance).
    The better one survives.

    This promotes diversity while maintaining quality.
    """
    if not offspring:
        return population[:population_size]

    survivors = list(population)

    for child in offspring:
        if len(survivors) < population_size:
            survivors.append(child)
            continue

        # Find most similar individual
        rival = find_most_similar(child, survivors)

        # Child replaces rival if better
        if rival is not None and child.fitness < rival.fitness:
            survivors.remove(rival)
            survivors.append(child)

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
            population = (
                elimination_with_crowding(population, offspring, GA.POPULATION_SIZE)
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
                stall_gens > GA.LNS_STALL_THRESHOLD
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
            self.is_sparse = is_sparse_matrix(problem.distance_matrix)
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
            if self.is_sparse:
                child = self._create_offspring_sparse(problem, parent1, parent2)
            else:
                child = self._create_offspring_dense(
                    problem, parent1, parent2, diversity
                )

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
        if offspring and GA.LOCAL_SEARCH_ENABLED:
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
        else:
            child = Individual(
                tour=np.copy(parent1.tour), mutation_rate=parent1.mutation_rate
            )

        # Adapt mutation rate if diversity was measured
        if diversity is not None:
            adaptive_mutation_rate(child, diversity)

        # Apply standard mutation
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
    dist_csv = "src/benchmark/tour500.csv"
    tour_csv = "316,20,487,399,372,402,210,279,306,9,336,335,247,131,468,135,139,7,320,17,394,344,486,283,225,119,128,51,278,303,266,201,137,244,386,259,22,460,380,246,400,73,497,349,445,475,452,430,321,41,133,147,309,81,5,88,134,439,491,78,214,185,343,361,441,302,351,67,238,477,115,166,495,261,264,49,424,197,140,490,110,390,90,457,217,64,221,341,63,112,255,443,101,484,200,56,305,129,59,308,209,418,138,295,241,461,326,397,314,472,310,32,432,426,481,436,132,62,102,412,43,178,409,371,123,223,329,275,160,145,421,458,94,76,307,389,172,106,113,240,356,388,413,33,111,262,339,222,342,179,203,498,291,120,427,453,331,98,284,163,249,61,45,6,442,18,406,230,144,40,464,77,30,489,10,274,28,337,24,84,153,80,103,151,270,340,224,488,467,374,474,107,227,419,281,448,260,218,146,292,470,156,36,280,250,175,37,184,334,256,31,379,393,46,248,239,480,116,190,71,401,75,219,433,454,168,0,318,177,296,431,162,192,323,86,126,363,333,304,72,263,423,482,142,276,365,299,158,191,55,38,364,216,311,471,141,70,369,301,21,93,395,50,143,330,362,39,332,195,232,408,96,228,405,174,183,285,206,435,2,297,205,182,14,357,16,169,462,97,287,425,315,352,494,288,434,52,366,171,170,293,187,121,189,359,83,449,473,148,233,450,438,451,243,345,150,53,499,173,466,282,277,15,429,353,370,294,65,347,447,26,273,161,136,199,384,313,19,368,208,60,289,188,268,420,322,387,245,79,105,11,385,257,396,35,122,27,4,493,269,416,479,213,367,202,87,68,373,455,376,130,47,290,89,428,186,234,378,215,220,492,3,478,231,317,312,92,252,91,149,127,118,383,44,242,155,237,348,398,194,469,354,483,8,415,407,456,154,69,29,114,325,485,211,117,48,25,100,198,437,444,212,95,124,298,236,74,328,410,159,324,355,13,403,465,109,319,258,476,12,350,459,272,417,235,164,23,125,34,265,496,42,152,108,254,375,381,82,446,267,411,167,229,440,176,251,58,104,57,404,204,377,338,286,66,85,391,382,358,207,422,54,99,327,1,226,181,253,165,463,157,196,414,346,300,193,392,360,180,271"

    length = evaluate_tour_from_csv_string(dist_csv, tour_csv)
    print(f"Tour length: {length:.2f}")
