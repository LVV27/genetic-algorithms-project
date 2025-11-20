import Reporter
import numpy as np
import random
import os

# ------------------------------
# GLOBAL GA PARAMETERS
# ------------------------------
GA_PARAMS = {
    "POPULATION_SIZE": 500,  # λ
    "OFFSPRING_SIZE": 250,  # μ
    "GENERATIONS": 1000,
    "TOURNAMENT_K": 3,
    "MUTATION_ALPHA_MIN": 0.02,
    "MUTATION_ALPHA_MAX": 0.12,
    "CROSSOVER_PROB": 0.8,
    "GREEDY_SEED_COUNT": 10,  # how many greedy tours to seed into initial population
    "GREEDY_RESTARTS": 20,  # number of different starts to try when building greedy seeds (best picked)
}


class TravelingSalesmanProblem:
    def __init__(self, distance_matrix: np.ndarray, filename: str = None):
        """Store the distance matrix and problem metadata"""
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.filename = filename

    def get_distance(self, city1: int, city2: int) -> float:
        """Return the distance between two cities"""
        return self.distance_matrix[city1, city2]

    def get_num_cities(self) -> int:
        """Return the number of cities in the problem"""
        return self.num_cities

    def print_info(self):
        """Print basic information about the TSP problem instance."""
        print(f"Problem: {self.filename if self.filename else 'Unknown'}")
        print(f"Number of cities: {self.num_cities}")

        heuristics = {
            "tour50.csv": 15665,
            "tour250.csv": 87874,
            "tour500.csv": 119458,
            "tour750.csv": 140149,
            "tour1000.csv": 70468,
        }
        heuristic_value = heuristics.get(self.filename, None)
        if heuristic_value is not None:
            print(f"Simple greedy heuristic objective value: {heuristic_value}")
        else:
            print("Heuristic value: unknown")
        print("")


# TODO Modify the class name to match your student number.
class r0123456:
    def __init__(self):
        """Initialize the solver and reporter."""
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename: str) -> int:
        """Main GA loop: read problem, initialize population, evolve, and report."""
        distance_matrix = self._read_distance_matrix(filename)
        filename_only = os.path.basename(filename)

        tsp_problem = TravelingSalesmanProblem(distance_matrix, filename_only)
        tsp_problem.print_info()

        population = self._init_population(tsp_problem)
        best_overall_individual = min(population, key=lambda x: x.fitness)

        for gen in range(1, GA_PARAMS["GENERATIONS"] + 1):
            offspring = self._generate_offspring(population, tsp_problem)

            population = elimination_lambda_plus_mu(population, offspring, GA_PARAMS["POPULATION_SIZE"])

            generation_best = min(population, key=lambda x: x.fitness)
            generation_mean_fitness = np.mean([ind.fitness for ind in population])

            if generation_best.fitness < best_overall_individual.fitness:
                best_overall_individual = generation_best

            time_left = self._report_and_check(generation_mean_fitness, generation_best)
            if time_left < 0:
                break

        return 0

    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        """Read CSV and return the distance matrix as a numpy array."""
        with open(filename, "r") as f:
            distance_matrix = np.loadtxt(f, delimiter=",")
        return distance_matrix

    def _init_population(self, tsp_problem: TravelingSalesmanProblem) -> list:
        """Create and return initial population of Individuals."""
        population = initialize_population_greedy(tsp_problem, GA_PARAMS["POPULATION_SIZE"])
        return population

    def _generate_offspring(self, population: list, tsp_problem: TravelingSalesmanProblem) -> list:
        """Generate offspring via tournament selection, crossover, mutation, and evaluation."""
        offspring = []
        while len(offspring) < GA_PARAMS["OFFSPRING_SIZE"]:
            parent1 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])
            parent2 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])

            if random.random() < GA_PARAMS["CROSSOVER_PROB"]:
                child = recombination(tsp_problem, parent1, parent2)
            else:
                child = Individual(tour=np.copy(parent1.tour), alpha=parent1.alpha)

            mutation(child)
            child.evaluate(tsp_problem)

            # you can choose to keep only valid children maybe better in recombination
            if not np.isinf(child.fitness):
                offspring.append(child)
            # else: discard invalid child
        return offspring

    def _report_and_check(self, generation_mean_fitness: float, generation_best: "Individual") -> float:
        """Report generation stats and return time left from Reporter."""
        return self.reporter.report(
            generation_mean_fitness, generation_best.fitness, generation_best.tour
        )


class Individual:
    def __init__(self, problem: TravelingSalesmanProblem = None, tour: np.ndarray = None, alpha: float = None, ):
        """Initialize an Individual: either a given tour or a random permutation. Set mutation rate alpha."""
        if tour is not None:
            self.tour = np.array(tour)
        elif problem is not None:
            self.tour = np.random.permutation(problem.num_cities)
        else:
            raise ValueError("Must provide tsp instance or explicit tour.")

        # not sure about the calculation
        self.alpha = (max(
            GA_PARAMS["MUTATION_ALPHA_MIN"], GA_PARAMS["MUTATION_ALPHA_MIN"] +
                                             (GA_PARAMS["MUTATION_ALPHA_MAX"] - GA_PARAMS[
                                                 "MUTATION_ALPHA_MIN"]) * random.random(), )
                      if alpha is None else alpha)
        self.fitness = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        """Compute and store total tour distance (fitness) using the problem's distance matrix."""
        n = len(self.tour)
        total_distance = 0.0
        for i in range(n):
            d = problem.get_distance(self.tour[i], self.tour[(i + 1) % n])
            if np.isinf(d):
                self.fitness = np.inf
                return self.fitness
            total_distance += d
        self.fitness = total_distance
        return self.fitness


# ------------------------------
# Greedy nearest-neighbour constructor
# ------------------------------
def nearest_neighbor_greedy(problem: TravelingSalesmanProblem, start: int = 0) -> np.ndarray:
    """
    Build a tour using the nearest neighbour heuristic starting from `start`.
    Returns an array of city indices (permutation).
    """
    n = problem.get_num_cities()
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        # pick nearest reachable city
        next_city = None
        best_dist = float("inf")
        for c in unvisited:
            d = problem.get_distance(current, c)
            if d < best_dist:
                best_dist = d
                next_city = c
        # if there is an unreachable city (infinite distance), abort and return None
        if next_city is None or np.isinf(best_dist):
            return None
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.array(tour, dtype=int)


# ------------------------------
# GA Utilities
# ------------------------------


# ------------------------------
# Population Initialization
# ------------------------------
def initialize_population(problem: TravelingSalesmanProblem, population_size: int) -> list[Individual]:
    """Create a list of random valid Individuals (no infinite tour distances)."""

    population = []

    while len(population) < population_size:
        ind = Individual(problem)
        fitness = ind.evaluate(problem)

        # Skip invalid tours (infinite total distance)
        if not np.isinf(fitness):
            population.append(ind)
    return population


def initialize_population_greedy(problem: TravelingSalesmanProblem, population_size: int) -> list[Individual]:
    """
    Create initial population, seeding with a small number of greedy tours
    (without 2-opt), then fill with random tours.
    """
    population: list[Individual] = []

    # 1) Greedy seeds: try multiple starts and pick the best unique tours
    greedy_candidates = []
    greedy_seeds = GA_PARAMS["GREEDY_SEED_COUNT"]  # 10
    greedy_restarts = GA_PARAMS["GREEDY_RESTARTS"]  # 20

    # sample start cities
    starts = list(range(problem.get_num_cities()))
    if greedy_restarts < len(starts):
        starts = random.sample(starts, greedy_restarts)
    else:
        starts = starts[:greedy_restarts]

    for s in starts:
        tour = nearest_neighbor_greedy(problem, start=s)
        if tour is None:
            continue
        ind = Individual(problem=None, tour=tour)
        ind.evaluate(problem)
        if not np.isinf(ind.fitness):
            greedy_candidates.append(ind)

    # Keep unique tours by fitness + tuple contents
    seen = set()
    unique_greedy = []
    for ind in sorted(greedy_candidates, key=lambda x: x.fitness):
        key = (int(ind.fitness), tuple(int(x) for x in ind.tour))
        if key not in seen:
            unique_greedy.append(ind)
            seen.add(key)
        if len(unique_greedy) >= greedy_seeds:
            break

    # add greedy seeds to population
    for ind in unique_greedy:
        population.append(ind)

    # 2) Fill remaining population with random valid individuals
    attempts = 0
    max_attempts = population_size * 10
    while len(population) < population_size and attempts < max_attempts:
        ind = Individual(problem)
        fitness = ind.evaluate(problem)
        if not np.isinf(fitness):
            population.append(ind)
        attempts += 1

    if len(population) < population_size:
        raise RuntimeError(
            f"Could not build full population (got {len(population)} of {population_size}). "
            "Check graph connectivity or increase max_attempts."
        )

    return population


# ------------------------------
# k-Tournament Selection
# ------------------------------
def tournament_selection(population: list, k: int) -> Individual:
    """Randomly select k individuals and return the one with lowest fitness."""
    assert 1 <= k <= len(population), "Tournament size k must be between 1 and len(population)"
    competitors = random.sample(population, k)
    for ind in competitors:
        if ind.fitness is None:
            raise ValueError("Individual fitness not evaluated")
    return min(competitors, key=lambda ind: ind.fitness)


# ------------------------------
# Mutation Operator (Swap Mutation)
# ------------------------------
def mutation(individual: Individual):
    """With probability alpha, swap two random cities in the tour."""
    if random.random() >= individual.alpha:
        return

    n = len(individual.tour)

    i, j = random.sample(range(n), 2)  # distinct indices
    individual.tour[i], individual.tour[j] = individual.tour[j], individual.tour[i]


# ------------------------------
# Recombination Operator (Partially Mapped Crossover - PMX)
# ------------------------------
def recombination(problem, parent1, parent2):
    """
    Robust PMX crossover:
    - copy a random segment from parent1
    - for each gene in parent2's segment that is not yet in child,
      find the position to place it by following the mapping via indices
      in parent2 until a free slot is found.
    """
    size = problem.get_num_cities()
    p1 = parent1.tour
    p2 = parent2.tour

    child = np.full(size, -1, dtype=int)

    # pick two distinct points
    a, b = sorted(random.sample(range(size), 2))
    # copy segment from parent1
    child[a: b + 1] = p1[a: b + 1]

    # precompute index of each city in parent2 for O(1) lookup
    p2_pos = {int(val): idx for idx, val in enumerate(p2)}

    # For each index in crossover segment, place p2 gene if not already present
    for i in range(a, b + 1):
        city = int(p2[i])
        if city in child:
            continue
        pos = i
        # follow mapping until we find a free position in child
        while child[pos] != -1:
            # mapped gene from parent1 at same position
            mapped_gene = int(p1[pos])
            # find where that mapped_gene sits in parent2
            pos = p2_pos[mapped_gene]
        child[pos] = city

    # fill remaining slots from parent2
    for i in range(size):
        if child[i] == -1:
            child[i] = int(p2[i])

    return Individual(problem, child)


# ------------------------------
# Elimination / Replacement (λ + μ)
# ------------------------------
def elimination_lambda_plus_mu(population: list[Individual], offspring: list[Individual], population_size: int) -> list[
    Individual]:
    """Select the best individuals from combined population and offspring."""
    combined = population + offspring

    # Ensure all fitness values are calculated
    for ind in combined:
        if ind.fitness is None:
            raise ValueError("All individuals must have fitness evaluated")

    # Sort by fitness (lower is better) and keep top `population_size`
    combined.sort(key=lambda x: x.fitness)
    return combined[:population_size]


if __name__ == "__main__":
    print("hello world!")
