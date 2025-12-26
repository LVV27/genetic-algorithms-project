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

    # Diversity Management
    DIVERSITY_CHECK_INTERVAL: int = 5
    DIVERSITY_TARGET: float = 0.01
    DIVERSITY_REPLACE_FRAC: float = 0.01
    DIVERSITY_SAMPLE_K: int = 100
    DIVERSITY_ACCEPT_OVERLAP_MAX: float = 0.5
    DIVERSITY_MAX_ATTEMPTS_PER_SLOT: int = 50

    # Perturbation Parameters (for diversity injection)
    PERTURB_LIGHT_SWAPS: int = 200  # n // this = number of swaps
    PERTURB_MEDIUM_SWAPS: int = 100
    PERTURB_HEAVY_SWAPS: int = 50
    PERTURB_HEAVY_ITERATIONS: int = 2

    # Logging Parameters
    LOG_INTERVAL: int = 100

    # Penalty for missing edges (Inf)
    USE_PENALTY_FOR_INF: bool = True
    INF_PENALTY_FACTOR: float = 10

    # Candidate list (0 to disable)
    CANDIDATE_LIST_SIZE: int = 0


GA = GAParams()


@dataclass(frozen=True)
class GAFeatures:
    """Feature flags for algorithm components."""

    USE_DIVERSITY_INJECTION: bool = True
    USE_MUTATION_PERTURB: bool = False  # For experimentation
    USE_MUTATION_CLASSIC: bool = True  # Standard mutation operators


FEATURES = GAFeatures()


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


def edge_diversity(population: List[Individual]) -> float:
    """
    Compute population diversity as fraction of unique directed edges.
    Higher values indicate more diversity.
    """
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


@numba.njit(cache=True)
def directed_edge_overlap_count(succ_a: np.ndarray, succ_b: np.ndarray) -> int:
    """Count shared directed edges between two tours."""
    n = succ_a.shape[0]
    count = 0
    for i in range(n):
        if succ_a[i] == succ_b[i]:
            count += 1
    return count


@numba.njit(cache=True)
def avg_overlap_ratio_with_sample(
    child_succ: np.ndarray,
    pop_succs: np.ndarray,
    sample_idxs: np.ndarray,
) -> float:
    """Average overlap ratio between child and population sample."""
    n = child_succ.shape[0]
    total = 0.0

    for k in range(sample_idxs.shape[0]):
        idx = sample_idxs[k]
        total += directed_edge_overlap_count(child_succ, pop_succs[idx]) / n

    return total / sample_idxs.shape[0]


# ==============================================================
# POPULATION INITIALIZATION
# ==============================================================


def nearest_neighbor_greedy(
    problem: TravelingSalesmanProblem,
    start_city: int = 0,
) -> Optional[np.ndarray]:
    """
    Nearest-neighbor heuristic: build tour by repeatedly visiting closest unvisited city.
    Returns None if construction fails (sparse graphs).
    """
    num_cities = problem.num_cities
    tour = [start_city]
    visited = {start_city}
    current_city = start_city

    for _ in range(num_cities - 1):
        best_distance = float("inf")
        best_next_city = None

        for candidate in range(num_cities):
            if candidate in visited:
                continue

            distance = problem.distance_matrix[current_city, candidate]
            if distance < best_distance:
                best_distance = distance
                best_next_city = candidate

        if best_next_city is None or np.isinf(best_distance):
            return None

        tour.append(best_next_city)
        visited.add(best_next_city)
        current_city = best_next_city

    return np.array(tour, dtype=int)


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
    greedy_count = 0
    max_greedy_attempts = num_greedy_target * 3

    start_cities = random.sample(
        range(problem.num_cities), min(problem.num_cities, max_greedy_attempts)
    )

    for start_city in start_cities:
        if greedy_count >= num_greedy_target:
            break

        tour = nearest_neighbor_greedy(problem, start_city)
        if tour is None:
            continue

        ind = Individual(tour=tour)
        ind.evaluate(problem)

        if np.isfinite(ind.fitness):
            population.append(ind)
            greedy_count += 1

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
# PERTURBATION OPERATORS (for diversity injection)
# ==============================================================


def perturb_double_bridge(tour: np.ndarray) -> np.ndarray:
    """
    Double-bridge move: cut tour into 4 segments and reconnect differently.
    More disruptive than simple mutations.
    """
    n = len(tour)
    if n < 8:
        return tour.copy()

    p1, p2, p3 = sorted(random.sample(range(1, n), 3))
    A = tour[:p1]
    B = tour[p1:p2]
    C = tour[p2:p3]
    D = tour[p3:]

    patterns = [
        np.concatenate([A, C, B, D]),
        np.concatenate([A, D, C, B]),
        np.concatenate([A, C, D, B]),
        np.concatenate([A, B, D, C]),
    ]
    return random.choice(patterns)


def perturb_multi_swap(tour: np.ndarray, k: int) -> np.ndarray:
    """Apply k random city swaps."""
    tour = tour.copy()
    n = len(tour)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def perturb_segment_reverse(tour: np.ndarray) -> np.ndarray:
    """Reverse a random contiguous segment."""
    tour = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i : j + 1] = tour[i : j + 1][::-1]
    return tour


def apply_perturbation(tour: np.ndarray, strength: str) -> np.ndarray:
    """
    Apply perturbation of specified strength to diversify tour.

    Args:
        tour: Original tour
        strength: 'light', 'medium', or 'heavy'
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

    # Heavy: apply multiple operations
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


def pmx_crossover(
    problem: TravelingSalesmanProblem,
    parent_a: Individual,
    parent_b: Individual,
) -> Individual:
    """
    Partially Mapped Crossover (PMX): position-based mapping.
    More disruptive than OX, better for escaping local optima.
    """
    n = problem.num_cities
    child_tour = np.copy(parent_a.tour)

    start, end = sorted(random.sample(range(n), 2))

    # Build mapping from segment
    mapping = {}
    for i in range(start, end + 1):
        city_a = parent_a.tour[i]
        city_b = parent_b.tour[i]
        mapping[city_a] = city_b
        child_tour[i] = city_b

    # Fix conflicts outside segment
    for i in list(range(0, start)) + list(range(end + 1, n)):
        city = child_tour[i]

        while city in mapping.values():
            for key, val in mapping.items():
                if val == city:
                    city = key
                    break

        child_tour[i] = city

    return Individual(tour=child_tour, mutation_rate=parent_a.mutation_rate)


# ==============================================================
# 2-OPT LOCAL SEARCH (kept for future use)
# ==============================================================


def two_opt_local_search(
    individual: Individual,
    problem: TravelingSalesmanProblem,
    max_iters: int = 5,
) -> Individual:
    """
    Apply 2-opt local search: remove two edges and reconnect.
    Continues until no improving move found or max iterations reached.

    Currently not integrated into main optimization loop.
    """
    tour = individual.tour
    distance_matrix = problem.distance_matrix
    feasible = problem.feasible

    if individual.fitness is None:
        individual.evaluate(problem)

    iteration = 0

    while iteration < max_iters:
        iteration += 1

        i, j = find_first_2opt_improvement(tour, distance_matrix, feasible)

        if i == -1:
            break  # Local optimum reached

        tour[i : j + 1] = tour[i : j + 1][::-1]

    individual.evaluate(problem)
    return individual


@numba.njit(cache=True)
def find_first_2opt_improvement(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    feasible: np.ndarray,
) -> Tuple[int, int]:
    """
    Find first improving 2-opt move.
    Returns (i, j) for improvement, or (-1, -1) if none found.
    """
    n = tour.shape[0]

    for i in range(1, n - 1):
        a = tour[i - 1]
        b = tour[i]

        for j in range(i + 1, n):
            next_j = (j + 1) % n
            c = tour[j]
            d = tour[next_j]

            if not feasible[a, c] or not feasible[b, d]:
                continue

            cost_removed = distance_matrix[a, b] + distance_matrix[c, d]
            cost_added = distance_matrix[a, c] + distance_matrix[b, d]

            if np.isinf(cost_removed):
                return i, j

            if cost_added < cost_removed - 1e-9:
                return i, j

    return -1, -1


# ==============================================================
# SURVIVOR SELECTION (Deterministic Crowding)
# ==============================================================


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


# ==============================================================
# DIVERSITY INJECTION
# ==============================================================


def inject_diversity(
    problem: TravelingSalesmanProblem,
    population: List[Individual],
) -> int:
    """
    Replace worst individuals with diverse solutions when diversity is low.
    Uses heavy perturbation + overlap checking to ensure diversity.

    Returns: number of replacements made.
    """
    if not population:
        return 0

    cur_diversity = edge_diversity(population)
    if cur_diversity >= GA.DIVERSITY_TARGET:
        return 0

    population.sort(key=lambda ind: ind.fitness)
    pop_size = len(population)
    n = population[0].tour.shape[0]

    # Build successor matrix for overlap checks
    pop_succs = np.empty((pop_size, n), dtype=np.int32)
    for i, ind in enumerate(population):
        pop_succs[i, :] = build_successor(ind.tour)

    # Target worst individuals for replacement
    num_slots = max(1, int(pop_size * GA.DIVERSITY_REPLACE_FRAC))
    victim_indices = list(range(pop_size - num_slots, pop_size))
    replaced = 0

    for victim_idx in victim_indices:
        # Try to generate diverse valid tour
        for _ in range(GA.DIVERSITY_MAX_ATTEMPTS_PER_SLOT):
            # Use strong perturbation on good solution
            parent = random.choice(population[: max(5, pop_size // 3)])
            new_tour = apply_perturbation(parent.tour, "heavy")

            # Evaluate candidate
            cand = Individual(tour=new_tour, mutation_rate=parent.mutation_rate)
            cand.evaluate(problem)

            if not np.isfinite(cand.fitness):
                continue

            # Check diversity: low overlap with population sample
            k = min(GA.DIVERSITY_SAMPLE_K, pop_size)
            sample_idxs = np.array(random.sample(range(pop_size), k), dtype=np.int32)
            cand_succ = build_successor(cand.tour)
            avg_overlap = float(
                avg_overlap_ratio_with_sample(cand_succ, pop_succs, sample_idxs)
            )

            if avg_overlap <= GA.DIVERSITY_ACCEPT_OVERLAP_MAX:
                population[victim_idx] = cand
                pop_succs[victim_idx, :] = cand_succ
                replaced += 1
                break

    population.sort(key=lambda ind: ind.fitness)
    return replaced


# ==============================================================
# MAIN SOLVER
# ==============================================================


class r0123456:
    """
    Genetic Algorithm solver for TSP.

    Features:
        - Mixed initialization (greedy + random)
        - Tournament selection
        - Order crossover (OX)
        - Weighted mutation operators
        - Deterministic crowding survivor selection
        - Diversity injection when population converges
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
        last_improve_gen = 0
        stall_gens = 0

        # Evolution loop
        print_section("EVOLUTION")
        start_time = time.perf_counter()
        checkpoint = start_time
        last_injected = 0

        for generation in range(1, GA.GENERATIONS + 1):
            # Generate offspring
            offspring = self._evolve_one_generation(population, problem)

            # Survivor selection
            population = elimination_with_crowding(
                population, offspring, GA.POPULATION_SIZE
            )

            # Inject diversity if converging
            if (
                FEATURES.USE_DIVERSITY_INJECTION
                and generation % GA.DIVERSITY_CHECK_INTERVAL == 0
            ):
                last_injected = inject_diversity(problem, population)
            else:
                last_injected = 0

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
                    f"Inj: {last_injected:3d} │ "
                    f"Δt: {elapsed:7.2f}s │ NoImp: {stall_gens:4d}"
                )

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

            # Mutation
            if FEATURES.USE_MUTATION_PERTURB:
                # Experimental: use perturbation as mutation
                if random.random() < child.mutation_rate:
                    child.tour = apply_perturbation(child.tour, "light")
            elif FEATURES.USE_MUTATION_CLASSIC:
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
# UTILITY FUNCTIONS
# ==============================================================


def evaluate_tour_from_csv_string(
    distance_csv_path: str,
    tour_csv_string: str,
) -> float:
    """
    Evaluate tour length from CSV string representation.
    Example: "0,5,2,7,1,3,4,6"
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
