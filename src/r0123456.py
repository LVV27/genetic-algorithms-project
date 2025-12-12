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
    "POPULATION_SIZE": 200,  # λ (number of individuals in population)
    "OFFSPRING_SIZE": 200,  # μ (number of offspring per generation)
    "GENERATIONS": 200,  # Maximum number of generations
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


def initialize_population(
    problem: TravelingSalesmanProblem, population_size: int
) -> List[Individual]:
    """
    Initialize population with random valid tours.

    Args:
        problem: TSP instance
        population_size: Target population size

    Returns:
        List of valid individuals
    """
    population = []

    while len(population) < population_size:
        individual = Individual(problem)
        if not np.isinf(individual.evaluate(problem)):
            population.append(individual)

    return population


def initialize_population_greedy(
    problem: TravelingSalesmanProblem, population_size: int
) -> List[Individual]:
    """
    Initialize population with greedy seeds + random individuals.

    Creates diverse high-quality initial population by:
    1. Running greedy heuristic from multiple random starting cities
    2. Keeping the best unique solutions as seeds
    3. Filling remainder with random valid tours

    Args:
        problem: TSP instance
        population_size: Target population size

    Returns:
        List of individuals (greedy seeds + random)
    """
    population = []
    greedy_candidates = []

    greedy_seeds = GA_PARAMS["GREEDY_SEED_COUNT"]
    greedy_restarts = GA_PARAMS["GREEDY_RESTARTS"]

    # Generate greedy tours from random starting cities
    start_cities = random.sample(
        range(problem.get_num_cities()), min(problem.get_num_cities(), greedy_restarts)
    )

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
    unique_greedy = []

    for individual in sorted(greedy_candidates, key=lambda x: x.fitness):
        tour_key = (int(individual.fitness), tuple(individual.tour))

        if tour_key not in seen_tours:
            unique_greedy.append(individual)
            seen_tours.add(tour_key)

        if len(unique_greedy) >= greedy_seeds:
            break

    population.extend(unique_greedy)

    # Fill remainder with random individuals
    attempts = 0
    max_attempts = population_size * 100

    while len(population) < population_size and attempts < max_attempts:
        individual = Individual(problem)
        if not np.isinf(individual.evaluate(problem)):
            population.append(individual)
        attempts += 1

    # Log initialization results
    if len(population) < population_size:
        logger.warning(
            f"Incomplete population: {len(population)}/{population_size} individuals"
        )
    else:
        logger.info(
            f"  Initialized {len(population)} individuals "
            f"({len(unique_greedy)} greedy, {len(population) - len(unique_greedy)} random)"
        )

    return population


# ==============================================================
# SELECTION
# ==============================================================


def tournament_selection(population: List[Individual], k: int) -> Individual:
    """
    Select individual using tournament selection.

    Args:
        population: Population to select from
        k: Tournament size

    Returns:
        Winner of the tournament (best fitness)
    """
    competitors = random.sample(population, k)
    return min(competitors, key=lambda ind: ind.fitness)


# ==============================================================
# MUTATION OPERATORS
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
    # Check if mutation should occur
    if random.random() >= individual.mutation_rate:
        return

    n = len(individual.tour)

    # Mutation operator probabilities
    operator_weights = {
        "swap": 0.2,
        "inversion": 0.4,
        "scramble": 0.2,
        "insertion": 0.2,
    }

    # Select mutation operator
    strategy = random.choices(
        list(operator_weights.keys()), weights=list(operator_weights.values()), k=1
    )[0]

    # Apply selected mutation
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
    size = problem.get_num_cities()
    tour1, tour2 = parent1.tour, parent2.tour

    child_tour = np.full(size, -1, dtype=int)

    # Copy a random segment from parent1
    segment_start, segment_end = sorted(random.sample(range(size), 2))
    child_tour[segment_start : segment_end + 1] = tour1[segment_start : segment_end + 1]

    # Build lookup for parent2 positions
    tour2_positions = {int(city): idx for idx, city in enumerate(tour2)}
    cities_in_child = set(child_tour[segment_start : segment_end + 1])

    # Map remaining cities from parent2
    for i in range(segment_start, segment_end + 1):
        city = int(tour2[i])
        if city in cities_in_child:
            continue

        # Find position for this city
        position = i
        while child_tour[position] != -1:
            mapped_city = int(tour1[position])
            position = tour2_positions[mapped_city]

        child_tour[position] = city
        cities_in_child.add(city)

    # Fill remaining positions from parent2
    for i in range(size):
        if child_tour[i] == -1:
            child_tour[i] = int(tour2[i])

    return Individual(problem, child_tour, mutation_rate=parent1.mutation_rate)


# ==============================================================
# SURVIVAL SELECTION
# ==============================================================


def elimination_lambda_plus_mu(
    population: List[Individual], offspring: List[Individual], population_size: int
) -> List[Individual]:
    """
    Select survivors using (λ+μ) strategy.

    Combines parents and offspring, then keeps the best individuals.

    Args:
        population: Current population
        offspring: Newly generated offspring
        population_size: Target population size

    Returns:
        New population of best individuals
    """
    combined = population + offspring
    combined.sort(key=lambda x: x.fitness)
    return combined[:population_size]


# ==============================================================
# ADAPTIVE MUTATION
# ==============================================================


def adaptive_mutation_rate(individual: Individual, diversity: float):
    """
    Adjust mutation rate based on population diversity.

    Strategy:
    - Low diversity → increase mutation (explore more)
    - High diversity → keep low mutation (exploit)

    Args:
        individual: Individual to adjust (modified in-place)
        diversity: Population diversity metric (0=low, 1=high)
    """
    base_rate = (GA_PARAMS["MUTATION_ALPHA_MIN"] + GA_PARAMS["MUTATION_ALPHA_MAX"]) / 2

    # Inverse relationship: less diversity → higher mutation
    diversity_factor = (1 - diversity) * 0.8
    individual.mutation_rate = min(1.0, base_rate + diversity_factor)


def population_diversity(population: List[Individual]) -> float:
    """
    Calculate population diversity as ratio of unique tours.

    Args:
        population: Population to analyze

    Returns:
        Diversity value in [0,1] where 1=all unique, 0=all identical
    """
    unique_tours = len(set(tuple(ind.tour) for ind in population))
    return unique_tours / len(population)


# ==============================================================
# MAIN GA SOLVER
# ==============================================================


class r0123456:
    """
    Main genetic algorithm solver for TSP using (μ+λ) strategy.
    """

    def __init__(self):
        """Initialize solver with reporter."""
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename: str) -> int:
        """
        Run genetic algorithm optimization on TSP instance.

        Args:
            filename: Path to distance matrix CSV file

        Returns:
            0 on success
        """
        # Load problem
        distance_matrix = self._read_distance_matrix(filename)
        problem = TravelingSalesmanProblem(distance_matrix, os.path.basename(filename))
        problem.print_info()

        # Initialize population
        print_section("INITIALIZATION")
        population = self._init_population(problem)
        self._log_population_stats(population, "Initial Population")

        # Track best solution
        best_overall = min(population, key=lambda x: x.fitness)

        # Main evolution loop
        print_section("EVOLUTION")
        start_time = time.perf_counter()
        checkpoint_time = start_time

        for generation in range(1, GA_PARAMS["GENERATIONS"] + 1):
            # Generate offspring
            offspring = self._evolve_one_generation(population, problem, generation)

            # Select survivors
            population = elimination_lambda_plus_mu(
                population, offspring, GA_PARAMS["POPULATION_SIZE"]
            )

            # Track statistics
            generation_best = min(population, key=lambda x: x.fitness)
            generation_mean = float(np.mean([ind.fitness for ind in population]))

            # Log progress every 50 generations
            if generation % 50 == 0:
                elapsed = time.perf_counter() - checkpoint_time
                checkpoint_time = time.perf_counter()

                logger.info(
                    f"  Gen {generation:3d}  │  Mean: {generation_mean:>9.2f}  │  "
                    f"Best: {generation_best.fitness:>9.2f}  │  {elapsed:.2f}s"
                )

            # Report to framework and check time limit
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
        }
        print_stats_table(final_stats)
        logger.info("")

        return 0

    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        """Load distance matrix from CSV file."""
        with open(filename, "r") as f:
            return np.loadtxt(f, delimiter=",")

    def _init_population(self, problem: TravelingSalesmanProblem) -> List[Individual]:
        """Initialize population using greedy seeding strategy."""
        return initialize_population_greedy(problem, GA_PARAMS["POPULATION_SIZE"])

    def _evolve_one_generation(
        self,
        population: List[Individual],
        problem: TravelingSalesmanProblem,
        generation: int,
    ) -> List[Individual]:
        """
        Generate offspring for one generation.

        Args:
            population: Current population
            problem: TSP instance
            generation: Current generation number

        Returns:
            List of offspring individuals
        """
        offspring = []

        # Calculate diversity periodically
        diversity = None
        if generation % GA_PARAMS["DIVERSITY_CHECK_INTERVAL"] == 0:
            diversity = population_diversity(population)

        # Generate offspring
        while len(offspring) < GA_PARAMS["OFFSPRING_SIZE"]:
            # Select parents
            parent1 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])
            parent2 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])

            # Create offspring (crossover or clone)
            if random.random() < GA_PARAMS["CROSSOVER_PROB"]:
                child = recombination(problem, parent1, parent2)
            else:
                child = Individual(
                    tour=np.copy(parent1.tour), mutation_rate=parent1.mutation_rate
                )

            # Apply adaptive mutation if diversity was computed
            if diversity is not None:
                adaptive_mutation_rate(child, diversity)

            # Mutate and evaluate
            mutation(child)
            child.evaluate(problem)

            # Keep only valid offspring
            if not np.isinf(child.fitness):
                offspring.append(child)

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
