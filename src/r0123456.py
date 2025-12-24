import Reporter
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

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
    POPULATION_SIZE: int = 50  # λ
    OFFSPRING_SIZE: int = 50  # μ
    GENERATIONS: int = 10_000

    # Selection
    TOURNAMENT_K: int = 2

    # Mutation (self-adaptive within [min, max])
    MUTATION_ALPHA_MIN: float = 0.02
    MUTATION_ALPHA_MAX: float = 0.20
    DIVERSITY_CHECK_INTERVAL: int = 5

    # Crossover
    CROSSOVER_PROB: float = 0.8

    # Greedy seeding
    GREEDY_SEED_COUNT: int = 5
    GREEDY_RESTARTS: int = 50

    # Local search (2-opt)
    LOCAL_SEARCH_ENABLED: bool = True
    LOCAL_SEARCH_MAX_ITERS: int = 2

    # Diversity preservation
    USE_CROWDING: bool = True
    DIVERSITY_PRESERVATION: float = 0.2

    # When to apply 2-opt (keep it selective to save time)
    LSO_APPLY_IF_BEATS_ANY_PARENT: bool = True
    LSO_NEAR_BEST_FRAC: float = 0.01
    LSO_ALWAYS_IMPROVE_TOP_K: int = 5
    LSO_LOG_COUNTS: bool = False


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
    """TSP instance wrapper: distance matrix + light helpers."""

    # Optional benchmark "known good" values (for quick sanity checks)
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

    def get_distance(self, city1: int, city2: int) -> float:
        return self.distance_matrix[city1, city2]

    def print_info(self) -> None:
        """One place to show instance metadata + heuristic target if known."""
        print_section("PROBLEM INSTANCE")
        stats = {"Instance": self.filename or "Unknown", "Cities": self.num_cities}
        hv = self.HEURISTIC_VALUES.get(self.filename)
        if hv is not None:
            stats["Known Heuristic"] = float(hv)
        print_stats_table(stats)


# ==============================================================
# INDIVIDUAL (solution representation)
# ==============================================================


class Individual:
    """
    A candidate tour + its mutation rate.
    Fitness = total tour length (lower is better).
    """

    def __init__(
        self,
        problem: Optional[TravelingSalesmanProblem] = None,
        tour: Optional[np.ndarray] = None,
        mutation_rate: Optional[float] = None,
    ):
        if tour is not None:
            self.tour = np.array(tour, dtype=int)
        elif problem is not None:
            self.tour = np.random.permutation(problem.num_cities)
        else:
            raise ValueError(
                "Provide either a problem (random init) or an explicit tour."
            )

        self.mutation_rate = (
            random.uniform(GA.MUTATION_ALPHA_MIN, GA.MUTATION_ALPHA_MAX)
            if mutation_rate is None
            else float(mutation_rate)
        )
        self.fitness: Optional[float] = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        """Compute tour length; returns inf if any required edge is missing."""
        n = len(self.tour)
        total_distance = 0.0

        for i in range(n):
            current_city = int(self.tour[i])
            next_city = int(self.tour[(i + 1) % n])
            distance = problem.get_distance(current_city, next_city)
            if np.isinf(distance):
                self.fitness = np.inf
                return self.fitness
            total_distance += distance

        self.fitness = total_distance
        return total_distance


# ==============================================================
# DIVERSITY HELPERS
# ==============================================================


def find_most_similar(
    individual: Individual, population: List[Individual]
) -> Optional[Individual]:
    """Crowding proxy: most similar by fitness distance (cheap + effective)."""
    if not population:
        return None
    diffs = [abs(individual.fitness - ind.fitness) for ind in population]  # type: ignore[arg-type]
    return population[int(np.argmin(diffs))]


def population_diversity(population: List[Individual]) -> float:
    """Diversity score = fraction of unique tours (exact match uniqueness)."""
    return len(set(tuple(ind.tour) for ind in population)) / max(1, len(population))


# ==============================================================
# GREEDY HEURISTIC (for seeding)
# ==============================================================


def nearest_neighbor_greedy(
    problem: TravelingSalesmanProblem, start_city: int = 0
) -> Optional[np.ndarray]:
    """Build a tour by repeatedly going to the closest unvisited city."""
    n = problem.num_cities
    unvisited = set(range(n))
    tour = [start_city]
    unvisited.remove(start_city)
    cur = start_city

    while unvisited:
        nxt, best = None, float("inf")
        for cand in unvisited:
            d = problem.get_distance(cur, cand)
            if d < best:
                nxt, best = cand, d
        if nxt is None or np.isinf(best):
            return None  # no feasible continuation
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    return np.array(tour, dtype=int)


# ==============================================================
# POPULATION INITIALIZATION (sparse-aware)
# ==============================================================


def initialize_population_greedy_sparse_aware(
    problem: TravelingSalesmanProblem, population_size: int
) -> List[Individual]:
    """
    Seed with greedy tours; fill remaining with random tours if feasible.

    For very sparse graphs, random permutations are almost always invalid,
    so we over-sample greedy restarts and allow cloning to reach minimum size.
    """
    n = problem.num_cities
    dm = problem.distance_matrix
    off = ~np.eye(n, dtype=bool)
    sparsity = float(np.isinf(dm[off]).sum()) / float(n * (n - 1))
    p_valid_random_tour = (1.0 - sparsity) ** n
    random_ok = p_valid_random_tour >= 1e-6

    greedy_seeds = GA.GREEDY_SEED_COUNT
    greedy_restarts = GA.GREEDY_RESTARTS
    if not random_ok:
        greedy_restarts = max(greedy_restarts, population_size * 40)
        greedy_seeds = max(greedy_seeds, min(population_size, 20))

    greedy_candidates: List[Individual] = []
    for start in random.sample(range(n), min(n, greedy_restarts)):
        tour = nearest_neighbor_greedy(problem, start)
        if tour is None:
            continue
        ind = Individual(tour=tour)
        ind.evaluate(problem)
        if np.isfinite(ind.fitness):
            greedy_candidates.append(ind)

    # Keep best unique greedy tours (uniqueness by exact tour)
    seen = set()
    population: List[Individual] = []
    for ind in sorted(greedy_candidates, key=lambda x: x.fitness):
        key = (int(ind.fitness), tuple(ind.tour))  # stable dedup for identical tours
        if key not in seen:
            population.append(ind)
            seen.add(key)
        if len(population) >= greedy_seeds:
            break

    # Fill remaining slots
    if random_ok:
        attempts, max_attempts = 0, population_size * 10_000
        while len(population) < population_size and attempts < max_attempts:
            ind = Individual(problem)
            if np.isfinite(ind.evaluate(problem)):
                population.append(ind)
            attempts += 1
    else:
        # No feasible random tours → clone what we have to keep GA running
        while len(population) < population_size and population:
            src = random.choice(population)
            clone = Individual(tour=np.copy(src.tour), mutation_rate=src.mutation_rate)
            clone.fitness = src.fitness
            population.append(clone)

    # Ensure a small minimum population even in extreme sparsity
    min_pop = max(3, population_size // 10)
    if len(population) < min_pop:
        logger.error(f"Critical: Only {len(population)} valid individuals found.")
        while len(population) < min_pop and population:
            src = random.choice(population)
            clone = Individual(tour=np.copy(src.tour), mutation_rate=src.mutation_rate)
            clone.fitness = src.fitness
            population.append(clone)
    elif len(population) < population_size:
        logger.warning(
            f"Incomplete population: {len(population)}/{population_size} individuals"
        )
    else:
        logger.info(
            f"  Initialized {len(population)} individuals "
            f"({min(len(seen), greedy_seeds)} greedy, {len(population) - min(len(seen), greedy_seeds)} random/clone)"
        )

    return population


# ==============================================================
# SELECTION
# ==============================================================


def tournament_selection(population: List[Individual], k: int) -> Individual:
    """Pick best out of k random competitors (pressure via k)."""
    return min(random.sample(population, k), key=lambda ind: ind.fitness)


# ==============================================================
# MUTATION OPERATORS
# ==============================================================


def _mutation_swap(tour: np.ndarray) -> None:
    """Swap two positions."""
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]


def _mutation_inversion(tour: np.ndarray) -> None:
    """Reverse a contiguous segment."""
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i : j + 1] = tour[i : j + 1][::-1]


def _mutation_scramble(tour: np.ndarray) -> None:
    """Shuffle a segment in-place."""
    i, j = sorted(random.sample(range(len(tour)), 2))
    seg = tour[i : j + 1].copy()
    random.shuffle(seg)
    tour[i : j + 1] = seg


def _mutation_insertion(individual: Individual) -> None:
    """Remove one city and insert it elsewhere."""
    n = len(individual.tour)
    i, j = random.sample(range(n), 2)
    gene = individual.tour[i]
    if i < j:
        individual.tour = np.concatenate(
            [
                individual.tour[:i],
                individual.tour[i + 1 : j + 1],
                [gene],
                individual.tour[j + 1 :],
            ]
        )
    else:
        individual.tour = np.concatenate(
            [
                individual.tour[:j],
                [gene],
                individual.tour[j:i],
                individual.tour[i + 1 :],
            ]
        )


def mutation(individual: Individual) -> None:
    """Dense-matrix mutation: operator mix + per-individual rate."""
    if random.random() >= individual.mutation_rate:
        return

    strategy = random.choices(
        ["swap", "inversion", "scramble", "insertion"],
        weights=[0.25, 0.35, 0.25, 0.15],
        k=1,
    )[0]

    if strategy == "swap":
        _mutation_swap(individual.tour)
    elif strategy == "inversion":
        _mutation_inversion(individual.tour)
    elif strategy == "scramble":
        _mutation_scramble(individual.tour)
    else:
        _mutation_insertion(individual)


def mutation_sparse_aware(
    individual: Individual, problem: TravelingSalesmanProblem
) -> None:
    """
    Sparse-safe mutation: keep trying until a valid child appears (or give up).
    This avoids wasting generations on invalid tours (inf fitness).
    """
    if random.random() >= individual.mutation_rate:
        return

    original_tour = individual.tour.copy()
    original_fit = individual.fitness
    max_tries = 30

    for _ in range(max_tries):
        t = random.choice(["swap", "inversion", "double_swap"])

        if t == "swap":
            _mutation_swap(individual.tour)
        elif t == "double_swap":
            _mutation_swap(individual.tour)
            _mutation_swap(individual.tour)
        else:
            i, j = sorted(random.sample(range(len(individual.tour)), 2))
            if j - i > len(individual.tour) // 3:
                continue
            individual.tour[i : j + 1] = individual.tour[i : j + 1][::-1]

        if np.isfinite(individual.evaluate(problem)):
            return  # keep the first valid mutation

        # revert and try again
        individual.tour = original_tour.copy()
        individual.fitness = original_fit


# ==============================================================
# CROSSOVER
# ==============================================================


def recombination(
    problem: TravelingSalesmanProblem, p1: Individual, p2: Individual
) -> Individual:
    """
    Order-based slice fill (often called OX-style):
    - Copy a slice from p1
    - Fill remaining positions in p2 order
    """
    n = problem.num_cities
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=int)

    child[a : b + 1] = p1.tour[a : b + 1]
    used = set(int(x) for x in child[a : b + 1])
    fill = [int(c) for c in p2.tour if int(c) not in used]

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1

    return Individual(tour=child, mutation_rate=p1.mutation_rate)


def crossover_sparse_aware(
    problem: TravelingSalesmanProblem, p1: Individual, p2: Individual
) -> Individual:
    """Try crossover; if invalid, clone the better parent."""
    child = recombination(problem, p1, p2)
    child.evaluate(problem)
    if np.isinf(child.fitness):
        better = p1 if p1.fitness < p2.fitness else p2
        child = Individual(
            tour=np.copy(better.tour), mutation_rate=better.mutation_rate
        )
        child.fitness = better.fitness
    return child


# ==============================================================
# SURVIVOR SELECTION
# ==============================================================


def elimination_with_crowding(
    population: List[Individual], offspring: List[Individual], pop_size: int
) -> List[Individual]:
    """
    Deterministic crowding:
    Each child competes against the most similar existing individual.
    """
    if not offspring:
        return population[:pop_size]

    new_pop = list(population)
    for child in offspring:
        if len(new_pop) >= pop_size:
            rival = find_most_similar(child, new_pop)
            if rival and child.fitness < rival.fitness:
                new_pop.remove(rival)
                new_pop.append(child)
        else:
            new_pop.append(child)

    new_pop.sort(key=lambda x: x.fitness)
    return new_pop[:pop_size]


def elimination_diversity_preserved(
    population: List[Individual], offspring: List[Individual], pop_size: int
) -> List[Individual]:
    """
    Keep elites + fill the rest with individuals that are far in fitness
    (cheap diversity heuristic when not using crowding).
    """
    combined = population + offspring
    if len(combined) <= pop_size:
        return combined

    combined.sort(key=lambda x: x.fitness)
    elites = int(pop_size * (1 - GA.DIVERSITY_PRESERVATION))
    new_pop = combined[:elites]
    candidates = combined[elites:]

    while len(new_pop) < pop_size and candidates:
        best, best_score = None, -1.0
        for c in candidates:
            score = float(np.mean([abs(c.fitness - ind.fitness) for ind in new_pop]))
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        new_pop.append(best)
        candidates.remove(best)

    while len(new_pop) < pop_size and candidates:
        new_pop.append(candidates.pop(0))

    return new_pop


# ==============================================================
# ADAPTIVE MUTATION RATE
# ==============================================================


def adaptive_mutation_rate(individual: Individual, diversity: float) -> None:
    """Lower diversity → slightly higher mutation (clamped to safe range)."""
    base = (GA.MUTATION_ALPHA_MIN + GA.MUTATION_ALPHA_MAX) / 2
    factor = (1 - diversity) * 0.5
    individual.mutation_rate = float(
        np.clip(base + factor, GA.MUTATION_ALPHA_MIN, GA.MUTATION_ALPHA_MAX)
    )


# ==============================================================
# LOCAL SEARCH (2-opt)
# ==============================================================


def two_opt_local_search(
    individual: Individual, problem: TravelingSalesmanProblem, max_iters: int = 5
) -> Individual:
    """
    2-opt with delta evaluation:
    swap edges (a-b, c-d) → (a-c, b-d) if it shortens tour.
    Skips moves that would introduce infinite edges (sparse-safe).
    """
    n = len(individual.tour)
    if individual.fitness is None:
        individual.evaluate(problem)

    tour = individual.tour
    best = float(individual.fitness)
    dm = problem.distance_matrix

    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nj = (j + 1) % n
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[nj]

                # Skip invalid edge combos quickly
                if (
                    np.isinf(dm[a, b])
                    or np.isinf(dm[c, d])
                    or np.isinf(dm[a, c])
                    or np.isinf(dm[b, d])
                ):
                    continue

                cur = dm[a, b] + dm[c, d]
                new = dm[a, c] + dm[b, d]
                delta = new - cur

                if delta < -1e-9:
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    best += float(delta)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    individual.tour = tour
    individual.fitness = best
    return individual


def apply_two_opt_to_offspring_when_it_matters(
    offspring: List[Individual],
    problem: TravelingSalesmanProblem,
    max_iters: int = 2,
    pop_best_fitness: Optional[float] = None,
    best_overall_fitness: Optional[float] = None,
) -> None:
    """
    Apply 2-opt selectively:
    - always top-K offspring
    - plus offspring that beat a parent OR are near best (within a small fraction)
    """
    if not offspring:
        return

    offspring.sort(key=lambda x: x.fitness)
    k = int(GA.LSO_ALWAYS_IMPROVE_TOP_K)
    selected = {id(ind) for ind in offspring[: max(0, k)]}

    thresholds = []
    if pop_best_fitness is not None and np.isfinite(pop_best_fitness):
        thresholds.append(pop_best_fitness * (1.0 + GA.LSO_NEAR_BEST_FRAC))
    if best_overall_fitness is not None and np.isfinite(best_overall_fitness):
        thresholds.append(best_overall_fitness * (1.0 + GA.LSO_NEAR_BEST_FRAC))
    thr = min(thresholds) if thresholds else None

    for ind in offspring:
        if id(ind) in selected:
            continue

        p1 = getattr(ind, "_p1_fitness", None)
        p2 = getattr(ind, "_p2_fitness", None)

        beats_parent = (
            GA.LSO_APPLY_IF_BEATS_ANY_PARENT
            and p1 is not None
            and p2 is not None
            and (ind.fitness + 1e-9) < max(p1, p2)
        )
        near_best = thr is not None and ind.fitness <= thr

        if beats_parent or near_best:
            selected.add(id(ind))

    if GA.LSO_LOG_COUNTS:
        logger.info(f"  LSO: 2-opt on {len(selected)}/{len(offspring)} offspring")

    for ind in offspring:
        if id(ind) in selected:
            two_opt_local_search(ind, problem, max_iters=max_iters)


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
    """Genetic Algorithm for TSP with sparse-graph fallbacks + optional 2-opt."""

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.is_sparse = False

    def optimize(self, filename: str) -> int:
        problem = TravelingSalesmanProblem(
            self._read_distance_matrix(filename), os.path.basename(filename)
        )
        problem.print_info()

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

        for gen in range(1, GA.GENERATIONS + 1):
            offspring = self._evolve_one_generation(population, problem, gen)

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
                div = population_diversity(population)
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
    ) -> List[Individual]:
        """Create offspring (strategy switches for sparse vs dense graphs)."""
        if generation == 1:
            self.is_sparse = is_sparse_matrix(problem.distance_matrix)
            if self.is_sparse:
                logger.info("  🔍 Sparse matrix detected - using adapted strategy")
                logger.info(
                    "  Strategy: Crowding + sparse-safe mutation + selective 2-opt"
                )

        offspring: List[Individual] = []
        pop_best = min(population, key=lambda x: x.fitness).fitness

        diversity = (
            population_diversity(population)
            if generation % GA.DIVERSITY_CHECK_INTERVAL == 0
            else None
        )

        # In sparse graphs: fewer valid children per attempt → reduce target, raise attempts a bit
        target = (
            min(GA.OFFSPRING_SIZE, len(population) * 3)
            if self.is_sparse
            else GA.OFFSPRING_SIZE
        )
        max_attempts = target * (10 if self.is_sparse else 100)

        attempts = 0
        while len(offspring) < target and attempts < max_attempts:
            attempts += 1
            p1 = tournament_selection(population, GA.TOURNAMENT_K)
            p2 = tournament_selection(population, GA.TOURNAMENT_K)

            if self.is_sparse:
                if random.random() < 0.4:
                    child = crossover_sparse_aware(problem, p1, p2)  # evaluated inside
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
                    recombination(problem, p1, p2)
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

        # 2-opt is expensive; apply selectively where it likely pays off
        if offspring and GA.LOCAL_SEARCH_ENABLED:
            apply_two_opt_to_offspring_when_it_matters(
                offspring,
                problem,
                max_iters=GA.LOCAL_SEARCH_MAX_ITERS,
                pop_best_fitness=pop_best,
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
    print("Import and use r0123456 class to optimize TSP instances")
