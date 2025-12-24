import Reporter
import numpy as np
import random
import os
import logging
import time
from typing import List, Tuple, Optional


# ------------------------------
# LOGGING SETUP
# ------------------------------
class ProfessionalFormatter(logging.Formatter):
    """Custom formatter for professional-looking logs."""

    FORMATS = {
        logging.INFO: "%(message)s",
        logging.WARNING: "⚠️  %(message)s",
        logging.ERROR: "❌ %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, "%(message)s")
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers
logger.handlers = []

# Add custom handler
handler = logging.StreamHandler()
handler.setFormatter(ProfessionalFormatter())
logger.addHandler(handler)
logger.propagate = False


# ------------------------------
# GLOBAL GA PARAMETERS
# ------------------------------
GA_PARAMS = {
    # Population parameters
    "POPULATION_SIZE": 50,  # λ (number of individuals in population)
    "OFFSPRING_SIZE": 50,  # μ (number of offspring per generation)
    "GENERATIONS": 10000,  # Maximum number of generations
    # Selection parameters
    "TOURNAMENT_K": 2,  # Tournament size for selection
    # Mutation parameters
    "MUTATION_ALPHA_MIN": 0.02,  # Minimum mutation rate
    "MUTATION_ALPHA_MAX": 0.2,  # Maximum mutation rate
    "DIVERSITY_CHECK_INTERVAL": 5,  # Check diversity every N generations
    # Crossover parameters
    "CROSSOVER_PROB": 0.8,  # Probability of crossover vs cloning
    # Initialization parameters
    "GREEDY_SEED_COUNT": 5,  # Number of greedy solutions to seed
    "GREEDY_RESTARTS": 50,  # Number of random starts for greedy
    # Local Search Operator (2-opt) parameters
    "LOCAL_SEARCH_ENABLED": True,  # Enable/disable 2-opt local search
    "LOCAL_SEARCH_MAX_ITERS": 2,  # Maximum iterations per individual in 2-opt
    # Diversity promotion parameters
    "USE_CROWDING": True,  # Enable deterministic crowding
    "DIVERSITY_PRESERVATION": 0.2,  # Keep 20% of population for diversity
    "LSO_APPLY_IF_BEATS_PARENT": True,
    "LSO_NEAR_BEST_FRAC": 0.01,  # apply if child <= (1+5%) * best_in_population
    "LSO_ALWAYS_IMPROVE_TOP_K": 5,  # always 2-opt top-K offspring by fitness
    "LSO_LOG_COUNTS": False,
}


# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================


def print_header(text: str, width: int = 70):
    """Print a formatted header."""
    logger.info("")
    logger.info("┌" + "─" * (width - 2) + "┐")
    logger.info(f"│ {text.center(width - 4)} │")
    logger.info("└" + "─" * (width - 2) + "┘")


def print_section(title: str, width: int = 70):
    """Print a section divider."""
    logger.info("")
    logger.info(f"{'─' * width}")
    logger.info(f" {title}")
    logger.info(f"{'─' * width}")


def print_stats_table(stats: dict, width: int = 70):
    """Print statistics in a clean table format."""
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key:<30} {value:>12.2f}")
        else:
            logger.info(f"  {key:<30} {value:>12}")


# ==============================================================
# PROBLEM REPRESENTATION
# ==============================================================


class TravelingSalesmanProblem:
    """
    Represents a TSP instance with distance matrix and helper methods.
    """

    # Known heuristic values for benchmark instances
    HEURISTIC_VALUES = {
        "tour50.csv": 15665,
        "tour250.csv": 87874,
        "tour500.csv": 119458,
        "tour750.csv": 140149,
        "tour1000.csv": 70468,
    }

    def __init__(self, distance_matrix: np.ndarray, filename: str = None):
        """
        Initialize TSP problem.

        Args:
            distance_matrix: NxN matrix of distances between cities
            filename: Optional name of the problem instance
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.filename = filename

    def get_distance(self, city1: int, city2: int) -> float:
        """Get distance between two cities."""
        return self.distance_matrix[city1, city2]

    def get_num_cities(self) -> int:
        """Get total number of cities."""
        return self.num_cities

    def print_info(self):
        """Print problem information and known heuristic values."""
        print_section("PROBLEM INSTANCE")
        stats = {
            "Instance": self.filename if self.filename else "Unknown",
            "Cities": self.num_cities,
        }
        heuristic_value = self.HEURISTIC_VALUES.get(self.filename)
        if heuristic_value is not None:
            stats["Known Heuristic"] = float(heuristic_value)
        print_stats_table(stats)


# ==============================================================
# INDIVIDUAL REPRESENTATION
# ==============================================================


class Individual:
    """
    Represents a candidate solution (tour) with adaptive mutation rate.
    """

    def __init__(
        self,
        problem: TravelingSalesmanProblem = None,
        tour: np.ndarray = None,
        mutation_rate: float = None,
    ):
        """
        Create an individual with a tour and mutation rate.

        Args:
            problem: TSP instance (for random initialization)
            tour: Explicit tour array (if not random)
            mutation_rate: Initial mutation rate (randomized if None)
        """
        # Initialize tour
        if tour is not None:
            self.tour = np.array(tour, dtype=int)
        elif problem is not None:
            self.tour = np.random.permutation(problem.num_cities)
        else:
            raise ValueError("Must provide TSP instance or explicit tour")

        # Initialize mutation rate
        if mutation_rate is None:
            self.mutation_rate = random.uniform(
                GA_PARAMS["MUTATION_ALPHA_MIN"], GA_PARAMS["MUTATION_ALPHA_MAX"]
            )
        else:
            self.mutation_rate = mutation_rate

        self.fitness = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        """
        Calculate tour length (fitness) for this individual.

        Args:
            problem: TSP instance with distance matrix

        Returns:
            Total tour length (lower is better)
        """
        n = len(self.tour)
        total_distance = 0.0

        for i in range(n):
            current_city = self.tour[i]
            next_city = self.tour[(i + 1) % n]
            distance = problem.get_distance(current_city, next_city)

            # Check for invalid distances
            if np.isinf(distance):
                self.fitness = np.inf
                return self.fitness

            total_distance += distance

        self.fitness = total_distance
        return total_distance


# ==============================================================
# DIVERSITY METRICS
# ==============================================================


def tour_distance(tour1: np.ndarray, tour2: np.ndarray) -> int:
    """
    Calculate edit distance between two tours.
    Counts how many positions have different cities.
    """
    return np.sum(tour1 != tour2)


def fitness_similarity(ind1: Individual, ind2: Individual) -> float:
    """
    Calculate fitness similarity (0=identical, 1=very different).
    """
    if ind1.fitness == ind2.fitness:
        return 0.0
    return abs(ind1.fitness - ind2.fitness) / max(ind1.fitness, ind2.fitness)


def find_most_similar(
    individual: Individual, population: List[Individual]
) -> Individual:
    """
    Find the most similar individual in population (for crowding).
    Uses fitness difference as primary metric.
    """
    if not population:
        return None

    # Find individual with most similar fitness
    similarities = [abs(individual.fitness - ind.fitness) for ind in population]
    most_similar_idx = np.argmin(similarities)
    return population[most_similar_idx]


# ==============================================================
# GREEDY HEURISTIC
# ==============================================================


def nearest_neighbor_greedy(
    problem: TravelingSalesmanProblem, start_city: int = 0
) -> Optional[np.ndarray]:
    """
    Construct a tour using nearest neighbor heuristic.

    Args:
        problem: TSP instance
        start_city: Starting city for the tour

    Returns:
        Tour array or None if no valid tour exists
    """
    n = problem.get_num_cities()
    unvisited = set(range(n))
    tour = [start_city]
    unvisited.remove(start_city)
    current_city = start_city

    while unvisited:
        nearest_city = None
        best_distance = float("inf")

        # Find nearest unvisited city
        for candidate in unvisited:
            distance = problem.get_distance(current_city, candidate)
            if distance < best_distance:
                nearest_city = candidate
                best_distance = distance

        # Check if valid move was found
        if nearest_city is None or np.isinf(best_distance):
            return None

        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    return np.array(tour, dtype=int)


# ==============================================================
# POPULATION INITIALIZATION
# ==============================================================


def initialize_population_greedy_sparse_aware(
    problem: TravelingSalesmanProblem, population_size: int
) -> List[Individual]:
    """
    Initialize population for sparse matrices - allows incomplete populations.

    SIMPLE CHANGE:
    - If feasibility-based sparsity says random tours are basically impossible,
      rely almost entirely on greedy restarts instead of random permutations.
    """
    population: List[Individual] = []
    greedy_candidates: List[Individual] = []

    # --- decide whether random tours are feasible ---
    n = problem.get_num_cities()
    dm = problem.distance_matrix
    off = ~np.eye(n, dtype=bool)
    sparsity = float(np.isinf(dm[off]).sum()) / float(n * (n - 1))
    p_valid_random_tour = (1.0 - sparsity) ** n

    random_tours_feasible = p_valid_random_tour >= 1e-6  # same as detector

    greedy_seeds = GA_PARAMS["GREEDY_SEED_COUNT"]
    greedy_restarts = GA_PARAMS["GREEDY_RESTARTS"]

    # If random tours are not feasible, increase greedy restarts (simple + effective)
    if not random_tours_feasible:
        greedy_restarts = max(
            greedy_restarts, population_size * 40
        )  # e.g. 2000 for pop=50
        greedy_seeds = max(greedy_seeds, min(population_size, 20))

    # Generate greedy tours from many starting cities
    start_cities = random.sample(range(n), min(n, greedy_restarts))

    for start in start_cities:
        tour = nearest_neighbor_greedy(problem, start)
        if tour is None:
            continue
        individual = Individual(tour=tour)
        individual.evaluate(problem)
        if not np.isinf(individual.fitness):
            greedy_candidates.append(individual)

    # Select best unique greedy solutions
    seen_tours = set()
    unique_greedy: List[Individual] = []

    for individual in sorted(greedy_candidates, key=lambda x: x.fitness):
        tour_key = (int(individual.fitness), tuple(individual.tour))
        if tour_key not in seen_tours:
            unique_greedy.append(individual)
            seen_tours.add(tour_key)
        if len(unique_greedy) >= greedy_seeds:
            break

    population.extend(unique_greedy)

    # Fill remainder
    # - If random tours are feasible: try random permutations
    # - If not: just clone greedy tours (still keeps GA running)
    if random_tours_feasible:
        attempts = 0
        max_attempts = population_size * 10000

        while len(population) < population_size and attempts < max_attempts:
            individual = Individual(problem)
            if not np.isinf(individual.evaluate(problem)):
                population.append(individual)
            attempts += 1
    else:
        # No feasible way to get random tours; clone from existing valid tours
        while len(population) < population_size and len(population) > 0:
            template = random.choice(population)
            clone = Individual(
                tour=np.copy(template.tour), mutation_rate=template.mutation_rate
            )
            clone.fitness = template.fitness
            population.append(clone)

    # Allow small populations for very sparse matrices
    min_population = max(3, population_size // 10)

    if len(population) < min_population:
        logger.error(f"Critical: Only {len(population)} valid individuals found.")
        while len(population) < min_population and len(population) > 0:
            template = random.choice(population)
            clone = Individual(tour=np.copy(template.tour))
            clone.fitness = template.fitness
            population.append(clone)
    elif len(population) < population_size:
        logger.warning(
            f"Incomplete population: {len(population)}/{population_size} individuals"
        )
    else:
        logger.info(
            f"  Initialized {len(population)} individuals "
            f"({len(unique_greedy)} greedy, {len(population) - len(unique_greedy)} random/clone)"
        )

    return population


# ==============================================================
# SELECTION
# ==============================================================


def tournament_selection(population: List[Individual], k: int) -> Individual:
    """
    Select individual using tournament selection.
    """
    competitors = random.sample(population, k)
    return min(competitors, key=lambda ind: ind.fitness)


# ==============================================================
# MUTATION OPERATORS - INCREASED DIVERSITY
# ==============================================================


def mutation(individual: Individual):
    """
    Apply mutation to an individual with adaptive rate.

    Randomly selects one of four mutation operators:
    - Swap: Exchange two cities
    - Inversion: Reverse a segment
    - Scramble: Shuffle a segment
    - Insertion: Move a city to a new position

    Args:
        individual: Individual to mutate (modified in-place)
    """
    if random.random() >= individual.mutation_rate:
        return

    n = len(individual.tour)

    # More diverse mutation operator mix
    operator_weights = {
        "swap": 0.25,
        "inversion": 0.35,
        "scramble": 0.25,
        "insertion": 0.15,
    }

    strategy = random.choices(
        list(operator_weights.keys()), weights=list(operator_weights.values()), k=1
    )[0]

    if strategy == "swap":
        _mutation_swap(individual.tour, n)
    elif strategy == "inversion":
        _mutation_inversion(individual.tour, n)
    elif strategy == "scramble":
        _mutation_scramble(individual.tour, n)
    elif strategy == "insertion":
        _mutation_insertion(individual, n)


def _mutation_swap(tour: np.ndarray, n: int):
    """Swap two random cities."""
    i, j = random.sample(range(n), 2)
    tour[i], tour[j] = tour[j], tour[i]


def _mutation_inversion(tour: np.ndarray, n: int):
    """Reverse a random segment of the tour."""
    i, j = sorted(random.sample(range(n), 2))
    tour[i : j + 1] = tour[i : j + 1][::-1]


def _mutation_scramble(tour: np.ndarray, n: int):
    """Randomly shuffle a segment of the tour."""
    i, j = sorted(random.sample(range(n), 2))
    segment = tour[i : j + 1].copy()
    random.shuffle(segment)
    tour[i : j + 1] = segment


def _mutation_insertion(individual: Individual, n: int):
    """Move a city to a different position in the tour."""
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


def mutation_sparse_aware(individual: Individual, problem: TravelingSalesmanProblem):
    """Safe mutation for sparse matrices with increased randomness."""
    if random.random() >= individual.mutation_rate:
        return

    n = len(individual.tour)
    original_tour = individual.tour.copy()
    original_fitness = individual.fitness
    max_tries = 30  # Increased from 20 for more variation attempts

    for _ in range(max_tries):
        # More diverse mutation types for sparse matrices
        mutation_type = random.choice(["swap", "inversion", "double_swap"])

        if mutation_type == "swap":
            i, j = random.sample(range(n), 2)
            individual.tour[i], individual.tour[j] = (
                individual.tour[j],
                individual.tour[i],
            )
        elif mutation_type == "double_swap":
            # Two consecutive swaps for more variation
            i, j = random.sample(range(n), 2)
            individual.tour[i], individual.tour[j] = (
                individual.tour[j],
                individual.tour[i],
            )
            k, l = random.sample(range(n), 2)
            individual.tour[k], individual.tour[l] = (
                individual.tour[l],
                individual.tour[k],
            )
        else:  # inversion
            i, j = sorted(random.sample(range(n), 2))
            if j - i > n // 3:  # Allow larger inversions (was n//4)
                continue
            individual.tour[i : j + 1] = individual.tour[i : j + 1][::-1]

        # Check if the new tour is valid
        new_fitness = individual.evaluate(problem)

        if not np.isinf(new_fitness):
            # Valid mutation found!
            return
        else:
            # Revert the mutation
            individual.tour = original_tour.copy()
            individual.fitness = original_fitness


# ==============================================================
# CROSSOVER
# ==============================================================


def recombination(
    problem: TravelingSalesmanProblem, parent1: Individual, parent2: Individual
) -> Individual:
    """
    Create offspring using Cycle Crossover (CX).

    CX preserves the absolute position of cities from parents while
    ensuring each city appears exactly once.

    Args:
        problem: TSP instance
        parent1: First parent
        parent2: Second parent

    Returns:
        New offspring individual
    """
    n = problem.get_num_cities()
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=int)

    child[a : b + 1] = parent1.tour[a : b + 1]
    used = set(int(x) for x in child[a : b + 1])

    fill = []
    for city in parent2.tour:
        c = int(city)
        if c not in used:
            fill.append(c)

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1

    return Individual(problem, child, mutation_rate=parent1.mutation_rate)


def crossover_sparse_aware(
    problem: TravelingSalesmanProblem, parent1: Individual, parent2: Individual
) -> Individual:
    """
    Edge-preserving crossover that validates the child tour.
    Falls back to cloning best parent if crossover creates invalid tour.
    """
    # Try cycle crossover
    child = recombination(problem, parent1, parent2)
    child.evaluate(problem)

    # If invalid, just clone the better parent
    if np.isinf(child.fitness):
        better_parent = parent1 if parent1.fitness < parent2.fitness else parent2
        child = Individual(
            tour=np.copy(better_parent.tour), mutation_rate=better_parent.mutation_rate
        )
        child.fitness = better_parent.fitness

    return child


# ==============================================================
# SURVIVAL SELECTION - DIVERSITY-PRESERVING
# ==============================================================


def elimination_with_crowding(
    population: List[Individual], offspring: List[Individual], population_size: int
) -> List[Individual]:
    """
    Deterministic crowding: offspring compete with most similar parents.
    Promotes diversity by preventing similar individuals from dominating.
    """
    if not offspring:
        return population[:population_size]

    new_population = list(population)

    for child in offspring:
        if len(new_population) >= population_size:
            # Find most similar individual
            most_similar = find_most_similar(child, new_population)

            # Replace if child is better
            if most_similar and child.fitness < most_similar.fitness:
                new_population.remove(most_similar)
                new_population.append(child)
        else:
            new_population.append(child)

    # Final sorting and truncation
    new_population.sort(key=lambda x: x.fitness)
    return new_population[:population_size]


def elimination_diversity_preserved(
    population: List[Individual], offspring: List[Individual], population_size: int
) -> List[Individual]:
    """
    Hybrid elimination: Keep best performers + diverse individuals.
    Balances exploitation (LSO) with exploration (diversity).
    """
    combined = population + offspring

    if len(combined) <= population_size:
        return combined

    # Sort by fitness
    combined.sort(key=lambda x: x.fitness)

    # Calculate how many to keep for diversity
    elites_count = int(population_size * (1 - GA_PARAMS["DIVERSITY_PRESERVATION"]))
    diversity_count = population_size - elites_count

    # Keep best performers
    new_population = combined[:elites_count]

    # Add diverse individuals from remainder
    candidates = combined[elites_count:]

    while len(new_population) < population_size and candidates:
        # Find most diverse candidate (different from current population)
        best_candidate = None
        max_diversity = -1

        for candidate in candidates:
            # Calculate average fitness distance to current population
            avg_distance = np.mean(
                [abs(candidate.fitness - ind.fitness) for ind in new_population]
            )

            if avg_distance > max_diversity:
                max_diversity = avg_distance
                best_candidate = candidate

        if best_candidate:
            new_population.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break

    # Fill remainder if needed
    while len(new_population) < population_size and candidates:
        new_population.append(candidates.pop(0))

    return new_population


# ==============================================================
# ADAPTIVE MUTATION - LIMITED RANGE
# ==============================================================


def adaptive_mutation_rate(individual: Individual, diversity: float):
    """
    Adjust mutation rate with LIMITED range to prevent over-exploitation.
    """
    # More conservative adaptation
    base_rate = (GA_PARAMS["MUTATION_ALPHA_MIN"] + GA_PARAMS["MUTATION_ALPHA_MAX"]) / 2
    diversity_factor = (1 - diversity) * 0.5  # Reduced from 0.8

    # Clamp to defined range
    new_rate = base_rate + diversity_factor
    individual.mutation_rate = np.clip(
        new_rate, GA_PARAMS["MUTATION_ALPHA_MIN"], GA_PARAMS["MUTATION_ALPHA_MAX"]
    )


def population_diversity(population: List[Individual]) -> float:
    """
    Calculate population diversity as ratio of unique tours.
    """
    unique_tours = len(set(tuple(ind.tour) for ind in population))
    return unique_tours / len(population)


# ==============================================================
# LOCAL SEARCH OPERATOR (2-OPT)
# ==============================================================


def two_opt_local_search(
    individual: Individual, problem: TravelingSalesmanProblem, max_iters: int = 5
) -> Individual:
    """
    Improve individual's tour using 2-opt with delta evaluation.
    Optimized for sparse matrices by skipping infinite edges.
    """
    n = len(individual.tour)
    if individual.fitness is None:
        individual.evaluate(problem)

    tour = individual.tour
    best_fitness = individual.fitness
    distance_matrix = problem.distance_matrix

    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                next_j = (j + 1) % n

                # Skip if any edge is infinite
                if (
                    np.isinf(distance_matrix[tour[i - 1], tour[i]])
                    or np.isinf(distance_matrix[tour[j], tour[next_j]])
                    or np.isinf(distance_matrix[tour[i - 1], tour[j]])
                    or np.isinf(distance_matrix[tour[i], tour[next_j]])
                ):
                    continue

                current_dist = (
                    distance_matrix[tour[i - 1], tour[i]]
                    + distance_matrix[tour[j], tour[next_j]]
                )
                new_dist = (
                    distance_matrix[tour[i - 1], tour[j]]
                    + distance_matrix[tour[i], tour[next_j]]
                )

                delta = new_dist - current_dist

                if delta < -1e-9:
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    best_fitness += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    individual.tour = tour
    individual.fitness = best_fitness
    return individual


def apply_two_opt_to_offspring_when_it_matters(
    offspring: list[Individual],
    problem: TravelingSalesmanProblem,
    max_iters: int = 2,
    pop_best_fitness: float | None = None,
    best_overall_fitness: float | None = None,
) -> None:
    if not offspring:
        return

    offspring.sort(key=lambda x: x.fitness)

    k = int(GA_PARAMS.get("LSO_ALWAYS_IMPROVE_TOP_K", 5))
    selected = set(id(ind) for ind in offspring[: max(0, k)])

    near_frac = float(GA_PARAMS.get("LSO_NEAR_BEST_FRAC", 0.01))
    thresholds = []
    if pop_best_fitness is not None and np.isfinite(pop_best_fitness):
        thresholds.append(pop_best_fitness * (1.0 + near_frac))
    if best_overall_fitness is not None and np.isfinite(best_overall_fitness):
        thresholds.append(best_overall_fitness * (1.0 + near_frac))
    thr = min(thresholds) if thresholds else None

    for ind in offspring:
        if id(ind) in selected:
            continue

        p1 = getattr(ind, "_p1_fitness", None)
        p2 = getattr(ind, "_p2_fitness", None)

        beats_any_parent = False
        if (
            GA_PARAMS.get("LSO_APPLY_IF_BEATS_ANY_PARENT", True)
            and p1 is not None
            and p2 is not None
        ):
            beats_any_parent = (ind.fitness + 1e-9) < max(p1, p2)

        near_best = thr is not None and ind.fitness <= thr

        if beats_any_parent or near_best:
            selected.add(id(ind))

    if GA_PARAMS.get("LSO_LOG_COUNTS", True):
        logger.info(f"  LSO: 2-opt on {len(selected)}/{len(offspring)} offspring")

    for ind in offspring:
        if id(ind) in selected:
            two_opt_local_search(ind, problem, max_iters=max_iters)


# ==============================================================
# SPARSE MATRIX DETECTION - FIXED
# ==============================================================


def is_sparse_matrix(distance_matrix: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Detect if distance matrix is sparse (many infinite/missing edges).
    FIXED: Only counts off-diagonal infinite values.
    """
    n = distance_matrix.shape[0]

    # Create mask for off-diagonal elements
    off_diagonal_mask = ~np.eye(n, dtype=bool)

    # Count infinite values in off-diagonal elements only
    off_diagonal_values = distance_matrix[off_diagonal_mask]
    inf_count = np.sum(np.isinf(off_diagonal_values))

    total_edges = n * (n - 1)
    sparsity = inf_count / total_edges if total_edges > 0 else 0.0

    logger.info(f"  Matrix sparsity: {sparsity:.1%} of edges are infinite")
    return sparsity > threshold


# ==============================================================
# MAIN GA SOLVER
# ==============================================================


class r0123456:
    """
    Main genetic algorithm solver for TSP with diversity promotion.
    """

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.is_sparse = False

    def optimize(self, filename: str) -> int:
        distance_matrix = self._read_distance_matrix(filename)
        problem = TravelingSalesmanProblem(distance_matrix, os.path.basename(filename))
        problem.print_info()

        # Initialize population
        print_section("INITIALIZATION")
        population = self._init_population(problem)
        self._log_population_stats(population, "Initial Population")

        # Track best solution
        best_overall = min(population, key=lambda x: x.fitness)
        best_overall_fitness = best_overall.fitness
        last_improve_gen = 0
        stall_gens = 0

        # Main evolution loop
        print_section("EVOLUTION")
        start_time = time.perf_counter()
        checkpoint_time = start_time

        for generation in range(1, GA_PARAMS["GENERATIONS"] + 1):
            # Generate offspring
            offspring = self._evolve_one_generation(population, problem, generation)

            # Use diversity-preserving elimination
            if GA_PARAMS["USE_CROWDING"]:
                population = elimination_with_crowding(
                    population, offspring, GA_PARAMS["POPULATION_SIZE"]
                )
            else:
                population = elimination_diversity_preserved(
                    population, offspring, GA_PARAMS["POPULATION_SIZE"]
                )

            generation_best = min(population, key=lambda x: x.fitness)

            # --- improvement / stall tracking ---
            if generation_best.fitness < best_overall_fitness - 1e-9:
                best_overall = generation_best
                best_overall_fitness = generation_best.fitness
                last_improve_gen = generation
                stall_gens = 0
            else:
                stall_gens += 1

            generation_mean = float(np.mean([ind.fitness for ind in population]))

            # Log diversity periodically
            if generation % 10 == 0:
                elapsed = time.perf_counter() - checkpoint_time
                checkpoint_time = time.perf_counter()
                diversity = population_diversity(population)

                # Fixed-width, consistent columns
                logger.info(
                    "  Gen {gen:4d} │ "
                    "Mean: {mean:12.2f} │ "
                    "Best: {best:12.2f} │ "
                    "Div: {div:8.2%} │ "
                    "Δt: {dt:7.2f}s │ "
                    "NoImp: {stall:4d} (last@{last:4d})".format(
                        gen=generation,
                        mean=generation_mean,
                        best=generation_best.fitness,
                        div=diversity,
                        dt=elapsed,
                        stall=stall_gens,
                        last=last_improve_gen,
                    )
                )

            time_remaining = self._report_and_check(generation_mean, generation_best)
            if time_remaining < 0:
                logger.info("\n  ⏱  Time limit reached")
                break

        # Final summary
        total_time = time.perf_counter() - start_time
        print_section("RESULTS")

        final_stats = {
            "Best Fitness": best_overall.fitness,
            "Generations": generation,
            "Total Time (s)": total_time,
            "Avg Time/Gen (s)": total_time / generation,
            "Final Diversity": population_diversity(population),
        }
        print_stats_table(final_stats)
        logger.info("")

        return 0

    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        """Load distance matrix from CSV file."""
        with open(filename, "r") as f:
            return np.loadtxt(f, delimiter=",")

    def _init_population(self, problem: TravelingSalesmanProblem) -> List[Individual]:
        """Initialize population using sparse-aware greedy seeding strategy."""
        return initialize_population_greedy_sparse_aware(
            problem, GA_PARAMS["POPULATION_SIZE"]
        )

    def _evolve_one_generation(
        self,
        population: List[Individual],
        problem: TravelingSalesmanProblem,
        generation: int,
    ) -> List[Individual]:
        """
        Evolution step that adapts to sparse matrices.
        """
        if generation == 1:
            self.is_sparse = is_sparse_matrix(problem.distance_matrix)
            if self.is_sparse:
                logger.info("  🔍 Sparse matrix detected - using adapted strategy")
                logger.info("  Strategy: Crowding + diverse mutations + limited LSO")

        offspring = []
        pop_best_fitness = min(population, key=lambda x: x.fitness).fitness

        # Calculate diversity periodically
        diversity = None
        if generation % GA_PARAMS["DIVERSITY_CHECK_INTERVAL"] == 0:
            diversity = population_diversity(population)

        if self.is_sparse:
            # For sparse matrices: focus on cloning + local search
            target_offspring = min(GA_PARAMS["OFFSPRING_SIZE"], len(population) * 3)
            max_attempts = target_offspring * 10
        else:
            # For dense matrices: normal GA
            target_offspring = GA_PARAMS["OFFSPRING_SIZE"]
            max_attempts = target_offspring * 100

        attempts = 0

        # Generate offspring
        while len(offspring) < target_offspring and attempts < max_attempts:
            attempts += 1

            parent1 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])
            parent2 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])

            if self.is_sparse:
                if random.random() < 0.4:
                    child = crossover_sparse_aware(
                        problem, parent1, parent2
                    )  # evaluates inside
                else:
                    better = parent1 if parent1.fitness < parent2.fitness else parent2
                    child = Individual(
                        tour=np.copy(better.tour), mutation_rate=better.mutation_rate
                    )
                    child.fitness = better.fitness

                # Tag parents BEFORE/AFTER mutation; doesn't matter, but must exist
                child._p1_fitness = parent1.fitness
                child._p2_fitness = parent2.fitness

                mutation_sparse_aware(child, problem)  # may re-evaluate

            else:
                if random.random() < GA_PARAMS["CROSSOVER_PROB"]:
                    child = recombination(problem, parent1, parent2)
                else:
                    child = Individual(
                        tour=np.copy(parent1.tour), mutation_rate=parent1.mutation_rate
                    )

                if diversity is not None:
                    adaptive_mutation_rate(child, diversity)

                mutation(child)
                child.evaluate(problem)

                # Tag parents (needed for "beats parent" criterion)
                child._p1_fitness = parent1.fitness
                child._p2_fitness = parent2.fitness

            if not np.isinf(child.fitness):
                offspring.append(child)

        # Fallback cloning
        min_offspring = max(3, len(population))
        cloned_count = 0
        while len(offspring) < min_offspring and len(population) > 0:
            parent = random.choice(population)
            clone = Individual(
                tour=np.copy(parent.tour), mutation_rate=parent.mutation_rate
            )
            clone.fitness = parent.fitness

            # Optional: tag parents so near-best criterion still works fine either way
            clone._p1_fitness = parent.fitness
            clone._p2_fitness = parent.fitness

            offspring.append(clone)
            cloned_count += 1

        if generation % 20 == 0 and (
            len(offspring) < target_offspring or cloned_count > 0
        ):
            logger.info(
                f"  Gen {generation}: Generated {len(offspring) - cloned_count}/{target_offspring} "
                f"offspring, cloned {cloned_count}"
            )

        # Apply 2-opt (crucial for sparse matrices!)
        if offspring and GA_PARAMS.get("LOCAL_SEARCH_ENABLED", False):
            apply_two_opt_to_offspring_when_it_matters(
                offspring,
                problem,
                max_iters=GA_PARAMS["LOCAL_SEARCH_MAX_ITERS"],
                pop_best_fitness=pop_best_fitness,
                # best_overall_fitness=best_overall_fitness,
            )

        return offspring

    def _report_and_check(
        self, generation_mean_fitness: float, generation_best: Individual
    ) -> float:
        """
        Report generation statistics and check time remaining.

        Args:
            generation_mean_fitness: Mean fitness of generation
            generation_best: Best individual in generation

        Returns:
            Time remaining in seconds (negative if exceeded)
        """
        return self.reporter.report(
            generation_mean_fitness, generation_best.fitness, generation_best.tour
        )

    def _log_population_stats(self, population: List[Individual], label: str):
        """Log population statistics."""
        fitnesses = [ind.fitness for ind in population]
        mean_fitness = float(np.mean(fitnesses))
        best_fitness = float(np.min(fitnesses))
        worst_fitness = float(np.max(fitnesses))

        stats = {
            "Mean": mean_fitness,
            "Best": best_fitness,
            "Worst": worst_fitness,
        }
        print_stats_table(stats)


if __name__ == "__main__":
    print("TSP Genetic Algorithm Solver")
    print("Import and use r0123456 class to optimize TSP instances")
