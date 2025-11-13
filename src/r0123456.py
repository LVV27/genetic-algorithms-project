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
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.filename = filename

    def get_distance(self, city1: int, city2: int) -> float:
        return self.distance_matrix[city1, city2]

    def get_num_cities(self) -> int:
        return self.num_cities

    def print_info(self):
        """Print basic problem information and heuristic reference."""
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


# Modify the class name to match your student number.
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop (now short & readable)
    def optimize(self, filename: str) -> int:
        distance_matrix = self._read_distance_matrix(filename)
        filename_only = os.path.basename(filename)

        tsp_problem = TravelingSalesmanProblem(distance_matrix, filename_only)
        tsp_problem.print_info()

        population = self._init_population(tsp_problem)
        evaluate_population(population, tsp_problem)
        best_overall_individual = min(population, key=lambda x: x.fitness)

        for gen in range(1, GA_PARAMS["GENERATIONS"] + 1):
            offspring = self._generate_offspring(population, tsp_problem)

            # Elimination
            population = elimination_lambda_plus_mu(
                population, offspring, GA_PARAMS["POPULATION_SIZE"]
            )

            generation_best = min(population, key=lambda x: x.fitness)
            generation_mean_fitness = np.mean([ind.fitness for ind in population])

            # Keep track of global best
            if generation_best.fitness < best_overall_individual.fitness:
                best_overall_individual = generation_best

            # Report and check time left
            time_left = self._report_and_check(
                generation_mean_fitness, generation_best
            )
            if time_left < 0:
                break

        return 0

    # -------------------------
    # Helper / extracted methods
    # -------------------------
    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        """Safely read the CSV distance matrix and return a numpy array."""
        with open(filename, "r") as f:
            distance_matrix = np.loadtxt(f, delimiter=",")
        return distance_matrix

    def _init_population(self, tsp_problem: TravelingSalesmanProblem) -> list:
        """Initialize and return the population (list of Individuals)."""
        population = initialize_population(tsp_problem, GA_PARAMS["POPULATION_SIZE"])
        return population

    def _generate_offspring(self, population: list, tsp_problem: TravelingSalesmanProblem) -> list:
        """
        Create offspring using tournament selection, crossover (probabilistic),
        mutation, and immediate evaluation. Returns list of offspring.
        """
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
        """
        Call the reporter with (mean, best, best_tour). Return timeLeft (as the
        reporter returns). Keep this single responsibility so reporting can be
        changed easily later.
        """
        return self.reporter.report(
            generation_mean_fitness, generation_best.fitness, generation_best.tour
        )


class Individual:
    def __init__(self, problem: TravelingSalesmanProblem = None, tour: np.ndarray = None, alpha: float = None, ):
        # Represent objects as a permutation
        # Start with generating a random order of objects
        # self.alpha = alpha
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
        n = len(self.tour)
        self.fitness = sum(
            problem.get_distance(self.tour[i], self.tour[(i + 1) % n])
            for i in range(n)
        )
        return self.fitness


# ------------------------------
# GA Utilities
# ------------------------------


# ------------------------------
# Population Initialization
# ------------------------------
def initialize_population(problem: TravelingSalesmanProblem, population_size: int) -> list[Individual]:
    population = [Individual(problem) for _ in range(population_size)]
    return population


def evaluate_population(population: list, problem: TravelingSalesmanProblem):
    for ind in population:
        ind.evaluate(problem)


# ------------------------------
# k-Tournament Selection
# ------------------------------
def tournament_selection(population: list, k: int) -> Individual:
    competitors = random.sample(population, k)
    for ind in competitors:
        if ind.fitness is None:
            raise ValueError("Individual fitness not evaluated")
    return min(competitors, key=lambda ind: ind.fitness)


# ------------------------------
# Mutation Operator (Swap Mutation)
# ------------------------------
def mutation(individual: Individual):
    if random.random() >= individual.alpha:
        return

    n = len(individual.tour)

    i, j = random.sample(range(n), 2)  # distinct indices
    individual.tour[i], individual.tour[j] = individual.tour[j], individual.tour[i]


# ------------------------------
# Recombination Operator (Partially Mapped Crossover - PMX)
# ------------------------------
def recombination(problem: TravelingSalesmanProblem, parent1: Individual, parent2: Individual) -> Individual:
    # Partially mapped crossover (Eiben-Smith, page 70) :
    # 1. Choose two crossover points at random, and copy the segment between
    # them from the first parent (P1) into the first offspring.
    # 2. Starting from the first crossover point look for elements in that segment
    # of the second parent (P2) that have not been copied.
    # 3. For each of these (say i), look in the offspring to see what element (say j)
    # has been copied in its place from P1.
    # 4. Place i into the position occupied by j in P2, since we know that we will
    # not be putting j there (as we already have it in our string).
    # 5. If the place occupied by j in P2 has already been filled in the offspring by
    # an element k, put i in the position occupied by k in P2.
    # 6. Having dealt with the elements from the crossover segment, the remaining
    # positions in this offspring can be filled from P2, and the second child is
    # created analogously with the parental roles reversed.
    size = problem.get_num_cities()
    order = np.full(size, -1)

    # 1. Randomly select two crossover points
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1

    # 1. Copy the segment from parent1 to offspring
    for i in range(cx_point1, cx_point2 + 1):
        order[i] = parent1.tour[i]

    # 2-5. Map the values from parent2’s segment
    for i in range(cx_point1, cx_point2 + 1):
        gene = parent2.tour[i]
        if gene not in order:  # 2. Elements of parent2 not already in offspring
            pos = i
            # 3-5. Follow the mapping from parent1 to parent2 until we find a free spot
            while order[pos] != -1:
                mapped_gene = parent1.tour[pos]
                pos = int(np.where(parent2.tour == mapped_gene)[0][0])
            order[pos] = gene

    # 6. Fill remaining empty positions with parent2’s genes
    for i in range(size):
        if order[i] == -1:
            order[i] = parent2.tour[i]

    return Individual(problem, order)


# ------------------------------
# Elimination / Replacement (λ + μ)
# ------------------------------
def elimination_lambda_plus_mu(population: list[Individual], offspring: list[Individual], population_size: int) -> list[
    Individual]:
    combined = population + offspring
    for ind in combined:
        if ind.fitness is None:
            raise ValueError("All individuals must have fitness evaluated")
    combined.sort(key=lambda x: x.fitness)
    return combined[:population_size]
