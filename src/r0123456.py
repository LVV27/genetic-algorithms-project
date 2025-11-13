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
        evaluate_population(population, tsp_problem)
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
        population = initialize_population(tsp_problem, GA_PARAMS["POPULATION_SIZE"])
        return population

    def _generate_offspring(self, population: list, tsp_problem: TravelingSalesmanProblem) -> list:
        """Generate offspring via tournament selection, crossover, mutation, and evaluation."""
        offspring = []
        for _ in range(GA_PARAMS["OFFSPRING_SIZE"]):
            parent1 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])
            parent2 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])

            if random.random() < GA_PARAMS["CROSSOVER_PROB"]:
                child = recombination(tsp_problem, parent1, parent2)
            else:
                child = Individual(tour=np.copy(parent1.tour), alpha=parent1.alpha)

            mutation(child)
            child.evaluate(tsp_problem)
            offspring.append(child)
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
        self.fitness = sum(problem.get_distance(self.tour[i], self.tour[(i + 1) % n]) for i in range(n))
        return self.fitness


# ------------------------------
# GA Utilities
# ------------------------------


# ------------------------------
# Population Initialization
# ------------------------------
def initialize_population(problem: TravelingSalesmanProblem, population_size: int) -> list[Individual]:
    """Create a list of Individuals with random tours for the given problem."""
    population = [Individual(problem) for _ in range(population_size)]
    return population


def evaluate_population(population: list, problem: TravelingSalesmanProblem):
    """Evaluate and store fitness for every individual in the population."""
    for ind in population:
        ind.evaluate(problem)


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
def recombination(problem: TravelingSalesmanProblem, parent1: Individual, parent2: Individual) -> Individual:
    """Generate a child by partially mapping crossover (PMX) between two parents."""
    size = problem.get_num_cities()
    order = np.full(size, -1)

    # 1. Choose two random crossover points
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1

    # 1. Copy segment from parent1 to offspring
    for i in range(cx_point1, cx_point2 + 1):
        order[i] = parent1.tour[i]

    # 2-5. Map remaining genes from parent2
    for i in range(cx_point1, cx_point2 + 1):
        gene = parent2.tour[i]
        if gene not in order:  # 2. Elements of parent2 not already in offspring
            pos = i
            # 3-5. Follow the mapping from parent1 to parent2 until we find a free spot
            while order[pos] != -1:
                mapped_gene = parent1.tour[pos]
                pos = int(np.where(parent2.tour == mapped_gene)[0][0])
            order[pos] = gene

    # 6. Fill empty positions with remaining genes from parent2
    for i in range(size):
        if order[i] == -1:
            order[i] = parent2.tour[i]

    return Individual(problem, order)


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
