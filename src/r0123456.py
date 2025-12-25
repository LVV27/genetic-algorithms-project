import Reporter
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import copy

import numpy as np

import numba

# ==============================================================
# LOGGING
# ==============================================================


class ProfessionalFormatter(logging.Formatter):
    """Minimal, readable log styling per severity level."""

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
logger.handlers = []  # reset handlers when re-running in notebooks/IDEs

_handler = logging.StreamHandler()
_handler.setFormatter(ProfessionalFormatter())
logger.addHandler(_handler)
logger.propagate = False


# ==============================================================
# GA PARAMETERS (single source of truth)
# ==============================================================


@dataclass(frozen=True)
class GAParams:
    # Population
    POPULATION_SIZE: int = 100  # λ
    OFFSPRING_SIZE: int = 100  # μ
    GENERATIONS: int = 10_000_000

    # Selection
    TOURNAMENT_K: int = 7

    # Mutation (self-adaptive within [min, max])
    MUTATION_ALPHA_MIN: float = 0.04
    MUTATION_ALPHA_MAX: float = 0.12
    DIVERSITY_CHECK_INTERVAL: int = 5

    # Crossover
    CROSSOVER_PROB: float = 0.8

    # Greedy seeding
    GREEDY_SEED_COUNT: int = 5
    GREEDY_RESTARTS: int = 100

    # Local search (2-opt and 3-opt)
    LOCAL_SEARCH_ENABLED: bool = True
    LOCAL_SEARCH_MAX_ITERS: int = 1

    # When to apply 2-opt (keep it selective to save time)
    LSO_APPLY_IF_BEATS_ANY_PARENT: bool = True
    LSO_NEAR_BEST_FRAC: float = 0.01
    LSO_ALWAYS_IMPROVE_TOP_K: int = 2
    LSO_LOG_COUNTS: bool = False

    # Diversity preservation
    USE_CROWDING: bool = True
    DIVERSITY_PRESERVATION: float = 0.3


GA = GAParams()


# ==============================================================
# SMALL PRINT HELPERS
# ==============================================================


def print_section(title: str, width: int = 70) -> None:
    """Console-friendly divider for stages (init/evolution/results)."""
    logger.info("")
    logger.info("─" * width)
    logger.info(f" {title}")
    logger.info("─" * width)


def print_stats_table(stats: dict) -> None:
    """Pretty-print a dict of stats with aligned columns."""
    for k, v in stats.items():
        logger.info(
            f"  {k:<30} {v:>12.2f}" if isinstance(v, float) else f"  {k:<30} {v:>12}"
        )


# ==============================================================
# PROBLEM DEFINITION
# ==============================================================


class TravelingSalesmanProblem:
    """
    Traveling Salesman Problem (TSP) instance wrapper.

    Stores:
    - A distance matrix
    - Basic metadata
    - Optional known heuristic benchmark values for sanity checks
    """

    # Optional benchmark "known good" values (for quick sanity checks)
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

    def get_distance(self, city_a: int, city_b: int) -> float:
        """Return the distance between two cities."""
        return self.distance_matrix[city_a, city_b]

    def print_info(self) -> None:
        """
        Print instance metadata and (if available) a known heuristic target.
        """
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
# INDIVIDUAL (solution representation)
# ==============================================================


class Individual:
    """
    A candidate TSP solution.

    Attributes:
    - tour: permutation of city indices
    - mutation_rate: per-individual mutation strength
    - fitness: total tour length (lower is better)
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
                "Either `problem` (random initialization) or `tour` must be provided."
            )

        # Initialize mutation rate
        if mutation_rate is None:
            self.mutation_rate = random.uniform(
                GA.MUTATION_ALPHA_MIN,
                GA.MUTATION_ALPHA_MAX,
            )
        else:
            self.mutation_rate = float(mutation_rate)

        self.fitness: Optional[float] = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        # self.fitness = evaluate_tour_no_numba(self=self, problem=problem)
        self.fitness = evaluate_tour_numba(self.tour, problem.distance_matrix)
        return self.fitness


def evaluate_tour_no_numba(self, problem: TravelingSalesmanProblem) -> float:
    """
    Compute and store the total tour length.

    Returns:
        Total distance of the tour, or np.inf if an edge is invalid.
    """
    num_cities = len(self.tour)
    total_distance = 0.0

    for index in range(num_cities):
        city = int(self.tour[index])
        next_city = int(self.tour[(index + 1) % num_cities])

        distance = problem.get_distance(city, next_city)
        if np.isinf(distance):
            self.fitness = np.inf
            return self.fitness

        total_distance += distance

    self.fitness = total_distance
    return total_distance


@numba.njit(cache=True)
def evaluate_tour_numba(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    total = 0.0
    n = tour.shape[0]

    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        d = distance_matrix[a, b]

        if np.isinf(d):
            return np.inf

        total += d

    return total


# ==============================================================
# DIVERSITY HELPERS
# ==============================================================


def find_most_similar(
    individual: Individual,
    population: List[Individual],
) -> Optional[Individual]:
    """
    Find the individual in the population with the closest fitness value.

    This is a cheap proxy for similarity (crowding):
    individuals with similar fitness are assumed to be similar solutions.
    """
    if not population:
        return None

    fitness_differences = [
        abs(individual.fitness - other.fitness) for other in population
    ]

    most_similar_index = int(np.argmin(fitness_differences))
    return population[most_similar_index]


def population_diversity(population: List[Individual]) -> float:
    """
    Measure population diversity as the fraction of unique tours.

    Exact tour matches are considered duplicates.
    Returns a value in [0, 1].
    """
    if not population:
        return 0.0

    unique_tours = {tuple(individual.tour) for individual in population}

    return len(unique_tours) / len(population)


def edge_diversity(population: List[Individual]) -> float:
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
            edges.add((a, b))
            edges.add((b, a))  # undirected
        edge_sets.append(edges)

    union_edges = set.union(*edge_sets)
    max_edges = len(population) * n * 2

    return len(union_edges) / max_edges



# ==============================================================
# POPULATION INITIALIZATION (sparse-aware)
# ==============================================================


def perturb_double_bridge(tour: np.ndarray) -> np.ndarray:
    n = len(tour)
    if n < 8:
        return tour.copy()

    p1, p2, p3 = sorted(random.sample(range(1, n), 3))
    return np.concatenate([tour[:p1], tour[p2:p3], tour[p1:p2], tour[p3:]])


def perturb_multi_swap(tour: np.ndarray, k: int) -> np.ndarray:
    tour = tour.copy()
    n = len(tour)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def perturb_segment_reverse(tour: np.ndarray) -> np.ndarray:
    tour = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i : j + 1] = tour[i : j + 1][::-1]
    return tour


def apply_perturbation(tour: np.ndarray, strength: str) -> np.ndarray:
    n = len(tour)

    if strength == "light":
        return perturb_multi_swap(tour, max(1, n // 200))

    if strength == "medium":
        return random.choice(
            [
                lambda t: perturb_double_bridge(t),
                lambda t: perturb_multi_swap(t, max(2, n // 100)),
                lambda t: perturb_segment_reverse(t),
            ]
        )(tour)

    # heavy
    t = tour.copy()
    for _ in range(2):
        t = random.choice(
            [
                perturb_double_bridge,
                lambda x: perturb_multi_swap(x, max(3, n // 50)),
                perturb_segment_reverse,
            ]
        )(t)
    return t


def repair_tour(
    problem: TravelingSalesmanProblem,
    tour: np.ndarray,
    max_repairs: int = 100,
) -> np.ndarray | None:
    """
    Attempt to repair a tour by fixing invalid (inf) edges via local swaps.
    Returns a repaired tour or None if repair fails.
    """
    tour = tour.copy()
    n = len(tour)

    for _ in range(max_repairs):
        repaired_any = False

        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]

            if not np.isfinite(problem.get_distance(a, b)):
                # Try swapping b with a later city c
                for j in range(i + 2, n):
                    c = tour[j]

                    if np.isfinite(problem.get_distance(a, c)) and np.isfinite(
                        problem.get_distance(c, b)
                    ):
                        tour[(i + 1) % n], tour[j] = tour[j], tour[(i + 1) % n]
                        repaired_any = True
                        break

                if not repaired_any:
                    return None  # unrecoverable edge

        if not repaired_any:
            return tour  # fully repaired

    return None


def fill_with_perturbations(
    problem: TravelingSalesmanProblem,
    base_population: List[Individual],
    target_size: int,
) -> int:
    added = 0
    attempts = 0
    max_attempts = (target_size - len(base_population)) * 50

    templates = sorted(base_population, key=lambda ind: ind.fitness)
    templates = templates[: max(1, len(templates) // 2)]

    while len(base_population) < target_size and attempts < max_attempts:
        attempts += 1
        parent = random.choice(templates)

        strength = random.choices(
            ["light", "medium", "heavy"],
            weights=[0.4, 0.4, 0.2],
            k=1,
        )[0]

        tour = apply_perturbation(parent.tour, strength)
        ind = Individual(tour=tour, mutation_rate=parent.mutation_rate)
        ind.evaluate(problem)

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
    Build a tour using the nearest-neighbor heuristic.

    Starting from `start_city`, repeatedly choose the closest *unvisited* city.
    Returns None if the tour cannot be completed due to missing/infinite edges.
    """
    num_cities = problem.num_cities
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)

    tour: List[int] = [start_city]
    current_city = start_city

    while unvisited:
        best_next_city: Optional[int] = None
        best_distance = float("inf")

        for candidate in unvisited:
            distance = problem.get_distance(current_city, candidate)
            if distance < best_distance:
                best_distance = distance
                best_next_city = candidate

        # If we can't move anywhere (or only missing edges remain), fail.
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
    Generate greedy nearest-neighbor tours from many start cities.
    Only valid (finite) tours are returned.
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
) -> tuple[List[Individual], set]:
    """
    Select the best unique tours (by exact sequence).
    Returns selected individuals and the set of seen tour keys.
    """
    seeds: List[Individual] = []
    seen_keys: set = set()

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
    Clone existing individuals until population reaches target_size.
    Returns number of clones added.
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
    # --------------------------------------------------------------
    # 1) Generate greedy seeds
    # --------------------------------------------------------------
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
        raise RuntimeError("No valid greedy tours found")

    # --------------------------------------------------------------
    # 2) Fill with perturbations of greedy solutions
    # --------------------------------------------------------------
    perturb_count = fill_with_perturbations(
        problem,
        population,
        population_size,
    )

    # --------------------------------------------------------------
    # 3) Clone only if absolutely necessary
    # --------------------------------------------------------------
    clone_count = 0
    if len(population) < population_size:
        clone_count = clone_to_size(population, population_size)

    # --------------------------------------------------------------
    # 4) Logging
    # --------------------------------------------------------------
    logger.info(
        f"Initialized {len(population)} individuals "
        f"({greedy_count} greedy, "
        f"{perturb_count} perturbations, "
        f"{clone_count} clone)"
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

    A subset of `tournament_size` individuals is sampled uniformly at random
    from the population. The individual with the best (lowest) fitness wins.
    """
    competitors = random.sample(population, tournament_size)
    winner = min(competitors, key=lambda ind: ind.fitness)
    return winner


# ==============================================================
# MUTATION OPERATORS
# ==============================================================


def mutation_swap(tour: np.ndarray) -> None:
    """Swap two random cities."""
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]


def mutation_inversion(tour: np.ndarray) -> None:
    """Reverse a random contiguous segment."""
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
    Apply a mutation to an individual using a weighted operator mix.
    Mutation happens with probability = individual.mutation_rate.
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
    Apply mutation while ensuring the result remains a valid tour.

    For sparse graphs (many missing edges), mutations often create invalid tours.
    This function retries several times, keeping the first valid mutation.
    """
    if random.random() >= individual.mutation_rate:
        return

    original_tour = individual.tour.copy()
    original_fitness = individual.fitness

    max_attempts = 100

    for _ in range(max_attempts):
        operator = random.choice(["swap", "double_swap", "inversion"])

        if operator == "swap":
            mutation_swap(individual.tour)

        elif operator == "double_swap":
            mutation_swap(individual.tour)
            mutation_swap(individual.tour)

        else:  # inversion with size guard
            start, end = sorted(random.sample(range(len(individual.tour)), 2))
            if end - start > len(individual.tour) // 3:
                continue
            individual.tour[start : end + 1] = individual.tour[start : end + 1][::-1]

        # Keep mutation if it produces a valid (finite) tour
        if np.isfinite(individual.evaluate(problem)):
            return

        # Otherwise revert and try again
        individual.tour = original_tour.copy()
        individual.fitness = original_fitness


# ==============================================================
# CROSSOVER
# ==============================================================


def order_crossover(
    problem: TravelingSalesmanProblem,
    parent_a: Individual,
    parent_b: Individual,
) -> Individual:
    """
    Order-based crossover (OX-style).

    Steps:
      1) Copy a contiguous slice from parent_a.
      2) Fill remaining positions using parent_b's order,
         skipping cities already used.
    """
    num_cities = problem.num_cities
    start, end = sorted(random.sample(range(num_cities), 2))

    child_tour = np.full(num_cities, -1, dtype=int)

    # 1) Copy slice from first parent
    child_tour[start : end + 1] = parent_a.tour[start : end + 1]
    used_cities = set(child_tour[start : end + 1])

    # 2) Fill remaining slots using the second parent's order
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
# EDGE-AWARE CROSSOVER (ERX) — SPARSE-AWARE
# ==============================================================


def _cyclic_neighbors(tour: np.ndarray) -> Dict[int, Set[int]]:
    """
    Build an undirected adjacency table from a tour.
    For each city, include its predecessor and successor (cyclic).
    """
    n = len(tour)
    adj: Dict[int, Set[int]] = {int(c): set() for c in tour}

    for i in range(n):
        city = int(tour[i])
        left = int(tour[(i - 1) % n])
        right = int(tour[(i + 1) % n])
        adj[city].add(left)
        adj[city].add(right)

    return adj


def _merge_edge_tables(p1: np.ndarray, p2: np.ndarray) -> Dict[int, Set[int]]:
    """Union of neighbor sets from both parents."""
    a1 = _cyclic_neighbors(p1)
    a2 = _cyclic_neighbors(p2)
    merged = {city: set(a1[city]) | set(a2[city]) for city in a1.keys()}
    return merged


def _remove_city_from_all(edge_table: Dict[int, Set[int]], city: int) -> None:
    """When a city is used, remove it from every neighbor list."""
    for nbrs in edge_table.values():
        nbrs.discard(city)


def _feasible_neighbors(
    problem: TravelingSalesmanProblem,
    current: int,
    candidates: Set[int],
) -> List[int]:
    """Filter candidates to those with a finite edge from `current`."""
    return [c for c in candidates if np.isfinite(problem.get_distance(current, c))]


def _choose_next_city_erx(
    problem: TravelingSalesmanProblem,
    edge_table: Dict[int, Set[int]],
    current: int,
    remaining: Set[int],
) -> int:
    """
    ERX rule of thumb:
      1) Prefer feasible neighbors from the edge table (i.e., edges present in parents).
      2) Among them, choose the one with the smallest neighbor-list size (avoids dead-ends).
      3) Break ties randomly.
      4) If no feasible edge-table neighbors exist, fall back to any remaining city that is feasible.
      5) If nothing is feasible, pick any remaining city (will likely be invalid; caller can detect).
    """
    # 1) Try parent-neighbors first (edge-aware)
    parent_neighbors = edge_table.get(current, set())
    options = _feasible_neighbors(problem, current, parent_neighbors & remaining)

    if options:
        # 2) Prefer the city whose edge_table list is smallest (classic ERX heuristic)
        min_deg = min(len(edge_table[c]) for c in options)
        best = [c for c in options if len(edge_table[c]) == min_deg]
        return random.choice(best)

    # 4) Fall back: any remaining city that is feasible from current
    feasible_any = [
        c for c in remaining if np.isfinite(problem.get_distance(current, c))
    ]
    if feasible_any:
        return random.choice(feasible_any)

    # 5) Last resort: will probably create an invalid edge; sparse-aware wrapper can fall back
    return random.choice(list(remaining))


def edge_recombination_crossover_sparse_aware(
    problem: TravelingSalesmanProblem,
    p1: Individual,
    p2: Individual,
    *,
    start_city: Optional[int] = None,
    max_restarts: int = 5,
) -> Individual:
    """
    Edge Recombination Crossover (ERX), adapted for sparse graphs.

    - Builds child by preserving parent edges when possible.
    - Tries a few restarts (different start cities) to increase chance of validity.
    - If all attempts yield an invalid tour, clones the better parent.

    Why ERX helps on sparse TSP:
    - Parent edges are known to exist => child is more likely to stay feasible than OX.
    """
    n = problem.num_cities
    best_child: Optional[Individual] = None

    # Candidate start cities: user-provided + a few random ones
    starts: List[int] = []
    if start_city is not None:
        starts.append(int(start_city))
    starts.extend(random.sample(range(n), k=min(n, max_restarts)))
    # Deduplicate while preserving order
    seen_start = set()
    starts = [s for s in starts if not (s in seen_start or seen_start.add(s))]

    for s in starts:
        edge_table = _merge_edge_tables(p1.tour, p2.tour)
        remaining = set(int(c) for c in range(n))

        current = int(s)
        child_tour: List[int] = [current]
        remaining.remove(current)
        _remove_city_from_all(edge_table, current)

        while remaining:
            nxt = _choose_next_city_erx(problem, edge_table, current, remaining)
            child_tour.append(nxt)
            remaining.remove(nxt)

            _remove_city_from_all(edge_table, nxt)
            current = nxt

        child = Individual(
            tour=np.asarray(child_tour, dtype=int), mutation_rate=p1.mutation_rate
        )
        child.evaluate(problem)

        # Keep the first valid child; or remember best (finite) child if you want
        if np.isfinite(child.fitness):
            return child

        # Track the best invalid/finite attempt just in case (optional behavior)
        if best_child is None or child.fitness < best_child.fitness:
            best_child = child

    # If ERX couldn't produce a valid child, fall back to cloning better parent
    better = p1 if p1.fitness < p2.fitness else p2
    clone = Individual(tour=np.copy(better.tour), mutation_rate=better.mutation_rate)
    clone.fitness = better.fitness
    return clone


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

    Each offspring competes against the most similar individual
    in the current population (similarity ≈ fitness distance).
    The better one survives.
    """
    if not offspring:
        return population[:population_size]

    survivors = list(population)

    for child in offspring:
        if len(survivors) < population_size:
            survivors.append(child)
            continue

        rival = find_most_similar(child, survivors)
        if rival is not None and child.fitness < rival.fitness:
            survivors.remove(rival)
            survivors.append(child)

    survivors.sort(key=lambda ind: ind.fitness)
    return survivors[:population_size]


def elimination_diversity_preserved(
    population: List[Individual],
    offspring: List[Individual],
    population_size: int,
) -> List[Individual]:
    """
    Survivor selection with explicit diversity preservation.

    Strategy:
      1) Keep a fraction of elite (best-fitness) individuals.
      2) Fill remaining slots with individuals that are far away
         in fitness from the current survivors.
    """
    combined = population + offspring

    if len(combined) <= population_size:
        return combined

    # Sort by fitness (lower is better)
    combined.sort(key=lambda ind: ind.fitness)

    elite_count = int(population_size * (1 - GA.DIVERSITY_PRESERVATION))
    survivors = combined[:elite_count]
    candidates = combined[elite_count:]

    # Greedily add individuals that maximize fitness distance
    while len(survivors) < population_size and candidates:
        best_candidate = None
        best_score = -1.0

        for candidate in candidates:
            avg_distance = float(
                np.mean([abs(candidate.fitness - s.fitness) for s in survivors])
            )
            if avg_distance > best_score:
                best_candidate = candidate
                best_score = avg_distance

        if best_candidate is None:
            break

        survivors.append(best_candidate)
        candidates.remove(best_candidate)

    # Fallback: fill remaining slots (if any)
    while len(survivors) < population_size and candidates:
        survivors.append(candidates.pop(0))

    return survivors


# ==============================================================
# ADAPTIVE MUTATION RATE
# ==============================================================


def adaptive_mutation_rate(
    individual: Individual,
    population_diversity: float,
) -> None:
    """
    Adapt an individual's mutation rate based on population diversity.

    Lower diversity  → higher mutation rate (more exploration)
    Higher diversity → lower mutation rate (more exploitation)

    The mutation rate is always clamped to a safe predefined range.
    """
    min_rate = GA.MUTATION_ALPHA_MIN
    max_rate = GA.MUTATION_ALPHA_MAX

    # Baseline mutation rate (midpoint of allowed range)
    base_rate = (min_rate + max_rate) / 2.0

    # Diversity adjustment: low diversity increases mutation
    adjustment = (1.0 - population_diversity) * 0.5

    new_rate = base_rate + adjustment

    individual.mutation_rate = float(np.clip(new_rate, min_rate, max_rate))


# ==============================================================
# LOCAL SEARCH (2-opt)
# ==============================================================

MoveKey = Tuple[int, int, int, int]  # canonicalized for symmetric/undirected caching


@dataclass
class TwoOptMoveCache:
    """
    Cache of 2-opt moves that are known to be:
      - infeasible (would create inf edges), or
      - non-improving (delta >= 0)

    This is safe because delta/feasibility depend only on distances, not on tour context.
    """

    seen_bad: Set[MoveKey] = field(default_factory=set)
    max_size: int = 1_000_000  # cap to avoid unbounded memory growth

    def has(self, key: MoveKey) -> bool:
        return key in self.seen_bad

    def add(self, key: MoveKey) -> None:
        if len(self.seen_bad) >= self.max_size:
            # simple eviction: clear (or swap for random eviction/LRU if you prefer)
            self.seen_bad.clear()
        self.seen_bad.add(key)


def two_opt_local_search(
    individual: Individual,
    problem: TravelingSalesmanProblem,
    max_iters: int = 5,
    cache: Optional[TwoOptMoveCache] = None,
) -> Individual:
    tour = individual.tour
    dm = problem.distance_matrix
    feasible = problem.feasible

    if individual.fitness is None:
        individual.evaluate(problem)

    iteration = 0

    while iteration < max_iters:
        iteration += 1

        i, j = find_first_2opt_improvement(tour, dm, feasible)

        if i == -1:
            break  # local optimum reached

        # Apply swap
        tour[i : j + 1] = tour[i : j + 1][::-1]

    individual.evaluate(problem)
    return individual


@numba.njit(cache=True)
def find_first_2opt_improvement(
    tour: np.ndarray,
    dm: np.ndarray,
    feasible: np.ndarray,
) -> tuple[int, int]:
    """
    Scan tour and return (i, j) of the first improving 2-opt move.
    Returns (-1, -1) if none found.
    """
    n = tour.shape[0]

    for i in range(1, n - 1):
        a = tour[i - 1]
        b = tour[i]

        for j in range(i + 1, n):
            nj = (j + 1) % n
            c = tour[j]
            d = tour[nj]

            # Safety check: new edges must be finite
            if not feasible[a, c] or not feasible[b, d]:
                continue

            cost_removed = dm[a, b] + dm[c, d]
            cost_added = dm[a, c] + dm[b, d]

            # Repair case
            if np.isinf(cost_removed):
                return i, j

            # Strict improvement
            if cost_added < cost_removed - 1e-9:
                return i, j

    return -1, -1


# ===================
# LNS
# ===================

def lns_destroy_repair(
    problem: TravelingSalesmanProblem,
    tour: np.ndarray,
    destroy_frac: float = 0.08,
) -> Optional[np.ndarray]:
    n = len(tour)
    k = int(n * destroy_frac)

    # 1) Remove k consecutive cities
    # start = random.randint(0, n - k)
    removed_idx = random.sample(range(n), k)
    removed = list(tour[i] for i in removed_idx)
    random.shuffle(removed)

    remaining = [c for c in tour if c not in removed]

    # 2) Reinsert greedily
    for city in removed:
        best_pos = None
        best_cost = float("inf")

        for i in range(len(remaining)):
            a = remaining[i - 1]
            b = remaining[i]
            if not np.isfinite(problem.get_distance(a, city)):
                continue
            if not np.isfinite(problem.get_distance(city, b)):
                continue

            cost = (
                problem.get_distance(a, city)
                + problem.get_distance(city, b)
                - problem.get_distance(a, b)
            )

            if cost < best_cost * (1.0 + random.uniform(-0.02, 0.02)):
                best_cost = cost
                best_pos = i

        if best_pos is None:
            return None

        remaining.insert(best_pos, city)

    return np.array(remaining, dtype=int)

def destruction_fraction(stall_gens: int) -> float:
    return 0.4  # nuclear



# ==============================================================
# HYBRID LOCAL SEARCH APPLICATION
# ==============================================================


def _compute_near_best_threshold(
    pop_best: Optional[float],
    overall_best: Optional[float],
) -> Optional[float]:
    """
    Compute a fitness threshold for "near best" selection.
    Smaller is better (tour length).
    """
    thresholds: List[float] = []

    if pop_best is not None and np.isfinite(pop_best):
        thresholds.append(pop_best * (1.0 + GA.LSO_NEAR_BEST_FRAC))

    if overall_best is not None and np.isfinite(overall_best):
        thresholds.append(overall_best * (1.0 + GA.LSO_NEAR_BEST_FRAC))

    return min(thresholds) if thresholds else None


def _offspring_beats_parent(ind: Individual) -> bool:
    """Return True if ind beats at least one parent fitness stored on the object."""
    if not GA.LSO_APPLY_IF_BEATS_ANY_PARENT:
        return False

    p1 = getattr(ind, "_p1_fitness", None)
    p2 = getattr(ind, "_p2_fitness", None)
    if p1 is None or p2 is None:
        return False

    # beats the worse parent (so it beats at least one)
    return (ind.fitness + 1e-9) < max(p1, p2)


def apply_local_search_hybrid(
    offspring: List[Individual],
    problem: TravelingSalesmanProblem,
    pop_best_fitness: Optional[float] = None,
    best_overall_fitness: Optional[float] = None,
    two_opt_cache: Optional[TwoOptMoveCache] = None,
) -> None:
    """
    Apply local search strategically to offspring:

    1. Apply 3-opt to top K offspring (most promising, worth expensive search)
    2. Apply 2-opt to remaining promising offspring (faster, wider coverage)

    This hybrid approach balances solution quality with computational cost.
    """
    if not offspring:
        return

    # Ensure fitness is available for sorting
    offspring.sort(key=lambda ind: ind.fitness)

    # Phase 2: Apply 2-opt to other promising offspring
    # (Skip the ones that already got 3-opt to avoid redundant work)
    top_k_2opt = int(GA.LSO_ALWAYS_IMPROVE_TOP_K)
    selected_for_2opt_ids = {id(ind) for ind in offspring[:top_k_2opt]}

    near_best_threshold = _compute_near_best_threshold(
        pop_best=pop_best_fitness,
        overall_best=best_overall_fitness,
    )

    # Also select offspring that beat parents or are near best
    for ind in offspring[top_k_2opt:]:
        if id(ind) in selected_for_2opt_ids:
            continue

        near_best = (
            near_best_threshold is not None and ind.fitness <= near_best_threshold
        )

        if _offspring_beats_parent(ind) or near_best:
            selected_for_2opt_ids.add(id(ind))

    if GA.LSO_LOG_COUNTS:
        logger.info(
            f"  LSO: 3-opt on {GA.THREE_OPT_TOP_K} offspring, "
            f"2-opt on {len(selected_for_2opt_ids)} offspring"
        )

    # Apply 2-opt to selected individuals
    for ind in offspring:
        if id(ind) in selected_for_2opt_ids:
            two_opt_local_search(
                ind, problem, max_iters=GA.LOCAL_SEARCH_MAX_ITERS, cache=two_opt_cache
            )


# ==============================================================
# SPARSITY DETECTION
# ==============================================================


def is_sparse_matrix(distance_matrix: np.ndarray, threshold: float = 0.1) -> bool:
    """Sparse = many off-diagonal inf edges (missing connections)."""
    n = distance_matrix.shape[0]
    off = ~np.eye(n, dtype=bool)
    inf_count = int(np.isinf(distance_matrix[off]).sum())
    total = n * (n - 1)
    sparsity = inf_count / total if total else 0.0
    logger.info(f"  Matrix sparsity: {sparsity:.1%} of edges are infinite")
    return sparsity > threshold


# ==============================================================
# MAIN SOLVER
# ==============================================================


class r0123456:
    """Genetic Algorithm for TSP with sparse-graph fallbacks + hybrid 2-opt/3-opt local search."""

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.is_sparse = False

    def optimize(self, filename: str) -> int:
        problem = TravelingSalesmanProblem(
            self._read_distance_matrix(filename), os.path.basename(filename)
        )
        problem.print_info()

        problem.feasible = np.isfinite(problem.distance_matrix)

        print_section("INITIALIZATION")
        population = initialize_population_greedy_sparse_aware(
            problem, GA.POPULATION_SIZE
        )
        self._log_population_stats(population, "Initial Population")

        best_overall = min(population, key=lambda x: x.fitness)
        best_overall_fitness = best_overall.fitness
        last_improve_gen, stall_gens = 0, 0

        print_section("EVOLUTION")
        start = time.perf_counter()
        checkpoint = start

        two_opt_cache = TwoOptMoveCache()

        for gen in range(1, GA.GENERATIONS + 1):
            offspring = self._evolve_one_generation(
                population, problem, gen, two_opt_cache, best_overall_fitness
            )

            population = (
                elimination_with_crowding(population, offspring, GA.POPULATION_SIZE)
                if GA.USE_CROWDING
                else elimination_diversity_preserved(
                    population, offspring, GA.POPULATION_SIZE
                )
            )

            gen_best = min(population, key=lambda x: x.fitness)

            # Track best-so-far + stall for progress visibility
            if gen_best.fitness < best_overall_fitness - 1e-9:
                best_overall, best_overall_fitness = gen_best, gen_best.fitness
                last_improve_gen, stall_gens = gen, 0
            else:
                stall_gens += 1

            gen_mean = float(np.mean([ind.fitness for ind in population]))

            # Periodic compact log line (consistent columns)
            if gen % 10 == 0:
                dt = time.perf_counter() - checkpoint
                checkpoint = time.perf_counter()
                div = edge_diversity(population)
                logger.info(
                    "  Gen {g:4d} │ Mean: {m:12.2f} │ Best: {b:12.2f} │ Div: {d:8.2%} │ "
                    "Δt: {t:7.2f}s │ NoImp: {s:4d} (last@{l:4d})".format(
                        g=gen,
                        m=gen_mean,
                        b=gen_best.fitness,
                        d=div,
                        t=dt,
                        s=stall_gens,
                        l=last_improve_gen,
                    )
                )

            if stall_gens > 80 and gen % 20 == 0:
                logger.info("DO SOMETHING")
                logger.info("LNS")
                frac = destruction_fraction(stall_gens)
                cand = lns_destroy_repair(problem, best_overall.tour, destroy_frac=frac)
                best_lns = None
                best_fit = best_overall_fitness
                for _ in range(20):  #← only TWO, keep it simple
                    cand = lns_destroy_repair(problem, best_overall.tour, destroy_frac=frac)
                    if cand is None:
                        continue
                ind = Individual(tour=cand)
                ind.evaluate(problem)
                two_opt_local_search(ind, problem, max_iters=30)
                if ind.fitness < best_fit:
                    best_lns = ind
                    best_fit = ind.fitness
                    if best_lns is not None:
                        delta = best_lns.fitness - best_overall_fitness # Accept improvement OR small uphill move when stalled
                        if delta < 0 or (stall_gens > 200 and delta < 30):
                            population[0] = best_lns
                            best_overall = best_lns
                            best_overall_fitness = best_lns.fitness
                            stall_gens = 0

            # Reporter handles time limit; negative means "stop"
            if self.reporter.report(gen_mean, gen_best.fitness, gen_best.tour) < 0:
                logger.info("\n  ⏱  Time limit reached")
                break

        total = time.perf_counter() - start
        print_section("RESULTS")
        print_stats_table(
            {
                "Best Fitness": best_overall.fitness,
                "Generations": gen,
                "Total Time (s)": total,
                "Avg Time/Gen (s)": total / gen,
                "Final Diversity": population_diversity(population),
            }
        )
        logger.info("")
        return 0

    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        """CSV → numpy distance matrix."""
        with open(filename, "r") as f:
            return np.loadtxt(f, delimiter=",")

    def _evolve_one_generation(
        self,
        population: List[Individual],
        problem: TravelingSalesmanProblem,
        generation: int,
        two_opt_cache: TwoOptMoveCache,
        best_overall_fitness: float,
    ) -> List[Individual]:
        """Create offspring (strategy switches for sparse vs dense graphs)."""
        if generation == 1:
            self.is_sparse = is_sparse_matrix(problem.distance_matrix)
            if self.is_sparse:
                logger.info("  🔍 Sparse matrix detected - using adapted strategy")
                logger.info(
                    "  Strategy: Crowding + sparse-safe mutation + hybrid local search (2-opt + 3-opt)"
                )
            else:
                logger.info(
                    "  Strategy: Hybrid local search (2-opt + 3-opt on top offspring)"
                )

        offspring: List[Individual] = []
        pop_best = min(population, key=lambda x: x.fitness).fitness

        diversity = (
            population_diversity(population)
            if generation % GA.DIVERSITY_CHECK_INTERVAL == 0
            else None
        )

        # In sparse graphs: fewer valid children per attempt → reduce target, raise attempts a bit
        if self.is_sparse:
            target = GA.OFFSPRING_SIZE
            max_attempts = target * 3   # NOT 10
        else:
            target = GA.OFFSPRING_SIZE
            max_attempts = target * 50

        attempts = 0
        while len(offspring) < target and attempts < max_attempts:
            attempts += 1
            p1 = tournament_selection(population, GA.TOURNAMENT_K)
            p2 = tournament_selection(population, GA.TOURNAMENT_K)

            if self.is_sparse:
                if random.random() < 0.15:
                    child = edge_recombination_crossover_sparse_aware(
                        problem, p1, p2
                    )  # evaluated inside
                else:
                    better = p1 if p1.fitness < p2.fitness else p2
                    child = Individual(
                        tour=np.copy(better.tour), mutation_rate=better.mutation_rate
                    )
                    child.fitness = better.fitness

                # Store parent fitness for "beats-parent" 2-opt trigger
                child._p1_fitness, child._p2_fitness = p1.fitness, p2.fitness
                mutation_sparse_aware(child, problem)  # re-evaluates as needed

            else:
                child = (
                    order_crossover(problem, p1, p2)
                    if random.random() < GA.CROSSOVER_PROB
                    else Individual(
                        tour=np.copy(p1.tour), mutation_rate=p1.mutation_rate
                    )
                )
                if diversity is not None:
                    adaptive_mutation_rate(child, diversity)

                mutation(child)
                child.evaluate(problem)
                child._p1_fitness, child._p2_fitness = p1.fitness, p2.fitness

            if np.isfinite(child.fitness):
                offspring.append(child)

        # If offspring generation struggled, clone to keep selection/elimination stable
        min_offspring = max(3, len(population))
        cloned = 0
        while len(offspring) < min_offspring and population:
            p = random.choice(population)
            clone = Individual(tour=np.copy(p.tour), mutation_rate=p.mutation_rate)
            clone.fitness = p.fitness
            clone._p1_fitness = clone._p2_fitness = p.fitness
            offspring.append(clone)
            cloned += 1

        if generation % 20 == 0 and (len(offspring) < target or cloned):
            logger.info(
                f"  Gen {generation}: Generated {len(offspring) - cloned}/{target} offspring, cloned {cloned}"
            )

        # Apply hybrid local search: 3-opt to best, 2-opt to rest
        if offspring and GA.LOCAL_SEARCH_ENABLED:
            apply_local_search_hybrid(
                offspring,
                problem,
                pop_best_fitness=pop_best,
                best_overall_fitness=best_overall_fitness,
                two_opt_cache=two_opt_cache,
            )

        return offspring

    def _log_population_stats(self, population: List[Individual], label: str) -> None:
        """Quick snapshot: mean/best/worst fitness."""
        fits = [ind.fitness for ind in population]
        print_stats_table(
            {
                "Mean": float(np.mean(fits)),
                "Best": float(np.min(fits)),
                "Worst": float(np.max(fits)),
            }
        )


if __name__ == "__main__":
    print("TSP Genetic Algorithm Solver")

    filename = "src/benchmark/tour1000.csv"  # <-- your path

    # Paste your tour here (keep the trailing comma if you want; we filter empties)
    tour_str = """776,978,164,458,968,581,836,340,711,778,659,443,15,660,614,350,692,32,380,144,577,943,185,626,884,365,63,349,68,183,39,645,619,476,790,192,596,612,178,896,131,632,845,196,685,392,609,22,428,695,591,456,561,998,75,269,370,624,926,95,439,46,742,565,36,193,534,731,741,997,573,955,812,571,588,189,58,570,537,83,401,989,798,454,237,296,208,100,505,858,460,513,966,467,372,360,954,436,115,451,31,242,363,390,230,600,468,453,182,927,246,839,120,840,292,973,564,848,899,66,249,499,179,471,352,399,300,78,508,754,717,794,413,26,700,886,402,184,204,320,518,715,913,705,306,662,475,74,244,756,162,990,631,347,295,62,554,86,276,547,803,684,487,457,792,566,140,878,500,929,405,427,636,101,414,145,272,881,512,932,161,953,838,466,761,928,866,915,325,595,307,416,312,87,971,110,582,400,996,823,601,567,60,494,956,10,238,447,539,368,843,377,939,607,851,811,374,283,713,972,51,857,469,288,545,698,376,766,525,874,835,393,701,560,667,156,142,634,102,553,227,613,70,957,481,321,116,265,617,622,773,260,521,29,807,719,979,268,126,969,187,462,666,994,147,895,34,98,194,351,171,630,438,519,650,419,233,267,217,668,806,678,708,88,841,354,388,257,281,135,355,449,517,718,904,816,109,703,648,725,664,572,485,765,328,597,690,976,746,336,287,383,129,672,6,146,702,200,326,275,706,524,274,491,991,735,55,919,834,375,598,651,371,576,463,562,813,496,28,940,121,985,799,863,975,605,386,958,801,71,633,9,982,90,736,942,890,452,484,802,358,871,649,822,501,122,97,279,241,916,714,253,432,592,740,781,172,348,56,934,906,820,542,825,877,759,205,176,828,936,273,198,82,459,406,879,190,865,540,679,912,30,720,188,652,404,285,442,389,586,861,738,367,974,37,49,931,604,918,797,686,159,585,661,384,298,775,42,332,91,157,141,465,472,45,123,530,47,450,961,693,606,177,85,473,236,207,13,258,516,656,880,40,426,783,732,584,917,316,898,730,99,317,755,615,50,544,455,620,749,331,112,817,950,492,290,495,67,944,637,461,339,959,757,113,209,80,105,643,694,579,251,344,796,199,299,160,259,84,947,54,420,933,533,888,446,197,464,782,408,999,699,854,61,621,313,827,910,409,337,166,387,528,733,809,11,330,440,128,278,810,397,223,480,830,346,422,343,829,286,329,557,945,611,751,254,627,4,425,212,72,543,885,522,657,984,323,215,747,623,875,563,602,470,696,970,552,641,255,832,526,682,206,752,264,833,963,289,777,154,925,949,924,407,247,670,65,222,559,210,94,831,21,322,252,77,903,900,106,186,868,774,319,894,12,395,640,593,280,504,986,793,213,486,133,587,424,568,130,628,937,922,779,849,114,789,663,240,1,506,385,477,905,366,511,155,431,411,680,599,219,76,964,381,625,434,448,856,433,195,535,658,862,791,675,153,379,203,589,753,353,357,64,909,478,134,245,748,41,677,935,418,787,538,396,44,308,497,261,762,722,435,314,768,610,8,988,415,764,334,729,96,655,412,921,987,536,515,860,294,149,25,488,243,53,356,79,503,763,665,359,218,489,304,688,911,482,318,785,948,786,691,93,946,165,158,952,555,704,341,410,124,5,256,616,117,309,132,104,89,853,441,136,234,270,726,20,284,138,479,148,639,111,707,382,490,181,855,202,556,498,819,721,282,324,653,373,398,14,897,771,262,814,527,529,3,770,674,739,493,689,788,859,24,608,795,882,175,709,977,335,169,818,887,35,876,644,758,629,960,18,509,277,914,842,669,214,800,523,724,421,143,216,728,315,228,784,293,430,16,727,852,221,43,152,301,271,837,231,2,697,225,646,951,369,769,574,889,846,870,867,902,531,310,510,291,118,676,361,302,33,108,967,417,603,444,163,549,167,0,180,305,850,804,17,981,403,712,734,137,750,883,737,760,303,558,445,710,869,107,920,52,48,575,808,235,378,201,745,962,901,923,23,546,681,364,983,780,826,716,647,151,391,995,872,239,250,892,168,767,437,327,311,220,333,248,743,362,654,864,635,127,683,550,590,980,338,815,27,345,7,174,507,930,394,541,638,583,73,191,429,844,671,232,263,580,81,594,551,139,69,19,226,965,170,772,821,119,907,805,150,618,993,847,723,514,891,673,938,224,532,893,502,211,941,266,908,229,642,548,57,125,873,578,687,103,483,520,59,992,342,297,744,423,824,173,38,474,92,569"""

    # --- Parse tour safely (handles trailing comma) ---
    tour = np.array(
        [int(x) for x in tour_str.replace("\n", "").split(",") if x.strip()],
        dtype=int,
    )

    problem = TravelingSalesmanProblem(
        distance_matrix=np.loadtxt(filename, delimiter=","),
        filename=os.path.basename(filename),
    )

    # --- Sanity checks: size + permutation ---
    n = problem.num_cities
    print(f"Instance: {filename}")
    print(f"Cities in instance: {n}")
    print(f"Cities in tour:     {len(tour)}")

    if len(tour) != n:
        print("❌ Tour length does not match number of cities!")
    else:
        print("✅ Tour length matches number of cities.")

    unique = len(set(tour.tolist()))
    if unique != len(tour):
        print(f"❌ Tour contains duplicates! unique={unique} total={len(tour)}")
        # show some duplicates
        from collections import Counter

        dupes = [c for c, cnt in Counter(tour.tolist()).items() if cnt > 1]
        print("Duplicate city ids (first 20):", dupes[:20])
    else:
        print("✅ Tour has no duplicates.")

    # check range
    if np.any(tour < 0) or np.any(tour >= n):
        bad = tour[(tour < 0) | (tour >= n)]
        print("❌ Tour contains out-of-range city ids:", bad[:20])
    else:
        print("✅ All city ids are within range 0..n-1.")

    # --- Evaluate using Option 1 ---
    ind = Individual(tour=tour)
    length = ind.evaluate(problem)
    print("Tour length:", length)
    print("Valid tour:", np.isfinite(length))

    # --- If invalid: find the FIRST inf edge (and show a few) ---
    if not np.isfinite(length):
        dm = problem.distance_matrix
        inf_edges = []

        for i in range(len(tour)):
            a = int(tour[i])
            b = int(tour[(i + 1) % len(tour)])  # wrap-around included
            d = dm[a, b]
            if np.isinf(d):
                inf_edges.append((i, a, b))

        print(f"\nFound {len(inf_edges)} invalid (inf) edges.")
        if inf_edges:
            i, a, b = inf_edges[0]
            print(f"First invalid edge at position {i}: {a} -> {b} is inf")

            print("\nFirst 10 invalid edges (pos, a->b):")
            for pos, aa, bb in inf_edges[:10]:
                print(f"  {pos:4d}: {aa} -> {bb}")

            # also show the closing edge explicitly
            last_a = int(tour[-1])
            first_b = int(tour[0])
            print(
                f"\nClosing edge (last -> first): {last_a} -> {first_b} = {dm[last_a, first_b]}"
            )
