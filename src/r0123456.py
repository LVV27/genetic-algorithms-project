import Reporter
import numpy as np
import random
import os
import logging
from typing import List
import time

# ------------------------------
# LOGGING SETUP
# ------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# GLOBAL GA PARAMETERS
# ------------------------------
GA_PARAMS = {
    "POPULATION_SIZE": 200,  # λ
    "OFFSPRING_SIZE": 200,  # μ
    "GENERATIONS": 200,
    "TOURNAMENT_K": 2,
    "MUTATION_ALPHA_MIN": 0.02,
    "MUTATION_ALPHA_MAX": 0.2,
    "CROSSOVER_PROB": 0.8,
    "GREEDY_SEED_COUNT": 5,
    "GREEDY_RESTARTS": 50,
}


# ==============================================================
# Traveling Salesman Representation
# ==============================================================

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
        logger.info(f"Problem: {self.filename if self.filename else 'Unknown'}")
        logger.info(f"Number of cities: {self.num_cities}")

        heuristics = {
            "tour50.csv": 15665,
            "tour250.csv": 87874,
            "tour500.csv": 119458,
            "tour750.csv": 140149,
            "tour1000.csv": 70468,
        }

        value = heuristics.get(self.filename, None)
        if value is not None:
            logger.info(f"Simple greedy heuristic objective value: {value}\n")
        else:
            logger.info("Heuristic value: unknown\n")


# ==============================================================
# Individual Representation
# ==============================================================

class Individual:
    def __init__(
            self,
            problem: TravelingSalesmanProblem = None,
            tour: np.ndarray = None,
            mutation_rate: float = None,
    ):
        if tour is not None:
            self.tour = np.array(tour, dtype=int)
        elif problem is not None:
            self.tour = np.random.permutation(problem.num_cities)
        else:
            raise ValueError("Must provide tsp instance or explicit tour.")

        # Mutation rate simplified using uniform distribution
        if mutation_rate is None:
            self.mutation_rate = random.uniform(GA_PARAMS["MUTATION_ALPHA_MIN"], GA_PARAMS["MUTATION_ALPHA_MAX"])
        else:
            self.mutation_rate = mutation_rate

        self.fitness = None

    def evaluate(self, problem: TravelingSalesmanProblem) -> float:
        n = len(self.tour)
        total = 0.0

        for i in range(n):
            d = problem.get_distance(self.tour[i], self.tour[(i + 1) % n])
            if np.isinf(d):
                self.fitness = np.inf
                return self.fitness
            total += d

        self.fitness = total
        return total


# ==============================================================
# Greedy Constructor
# ==============================================================

def nearest_neighbor_greedy(problem: TravelingSalesmanProblem, start: int = 0) -> np.ndarray:
    n = problem.get_num_cities()
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)

    current = start
    while unvisited:
        next_city = None
        best_d = float("inf")

        for c in unvisited:
            d = problem.get_distance(current, c)
            if d < best_d:
                next_city = c
                best_d = d

        if next_city is None or np.isinf(best_d):
            return None

        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.array(tour, dtype=int)


# ==============================================================
# Population Initialization
# ==============================================================

def initialize_population(problem: TravelingSalesmanProblem, population_size: int) -> List[Individual]:
    population: List[Individual] = []

    while len(population) < population_size:
        ind = Individual(problem)
        if not np.isinf(ind.evaluate(problem)):
            population.append(ind)

    return population


def initialize_population_greedy(problem: TravelingSalesmanProblem, population_size: int) -> List[Individual]:
    population: List[Individual] = []

    greedy_candidates = []
    greedy_seeds = GA_PARAMS["GREEDY_SEED_COUNT"]
    greedy_restarts = GA_PARAMS["GREEDY_RESTARTS"]

    starts = list(range(problem.get_num_cities()))
    starts = random.sample(starts, min(len(starts), greedy_restarts))

    for s in starts:
        tour = nearest_neighbor_greedy(problem, s)
        if tour is None:
            continue
        ind = Individual(tour=tour)
        ind.evaluate(problem)
        if not np.isinf(ind.fitness):
            greedy_candidates.append(ind)

    seen = set()
    unique_greedy: List[Individual] = []

    for ind in sorted(greedy_candidates, key=lambda x: x.fitness):
        key = (int(ind.fitness), tuple(ind.tour))
        if key not in seen:
            unique_greedy.append(ind)
            seen.add(key)
        if len(unique_greedy) >= greedy_seeds:
            break

    population.extend(unique_greedy)

    attempts = 0
    max_attempts = population_size * 100

    while len(population) < population_size and attempts < max_attempts:
        ind = Individual(problem)
        if not np.isinf(ind.evaluate(problem)):
            population.append(ind)
        attempts += 1

    if len(population) < population_size:
        logger.warning(
            f"Could not initialize full population: only {len(population)} / {population_size} "
            f"individuals generated after {attempts} attempts"
        )
    else:
        logger.info(
            f"Population initialization finished: {len(population)} / {population_size} individuals "
            f"after {attempts} attempts"
        )

    return population


# ==============================================================
# GA Operators
# ==============================================================

def tournament_selection(population: List[Individual], k: int) -> Individual:
    competitors = random.sample(population, k)
    return min(competitors, key=lambda ind: ind.fitness)


def mutation(individual: Individual):
    if random.random() >= individual.mutation_rate:
        return

    n = len(individual.tour)

    operator_probs = {
        "swap": 0.2,
        "inversion": 0.4,
        "scramble": 0.2,
        "insertion": 0.2
    }

    strategy = random.choices(
        list(operator_probs.keys()),
        weights=list(operator_probs.values()),
        k=1
    )[0]

    # SWAP
    if strategy == "swap":
        i, j = random.sample(range(n), 2)
        individual.tour[i], individual.tour[j] = individual.tour[j], individual.tour[i]

    # INVERSION
    elif strategy == "inversion":
        i, j = sorted(random.sample(range(n), 2))
        segment = individual.tour[i:j + 1]
        individual.tour[i:j + 1] = segment[::-1]

    # SCRAMBLE
    elif strategy == "scramble":
        i, j = sorted(random.sample(range(n), 2))
        segment = individual.tour[i:j + 1]
        random.shuffle(segment)
        individual.tour[i:j + 1] = segment

    # INSERTION
    elif strategy == "insertion":
        i, j = random.sample(range(n), 2)
        gene = individual.tour[i]

        if i < j:
            individual.tour = np.concatenate(
                (individual.tour[:i], individual.tour[i + 1:j + 1], [gene], individual.tour[j + 1:])
            )
        else:
            individual.tour = np.concatenate(
                (individual.tour[:j], [gene], individual.tour[j:i], individual.tour[i + 1:])
            )


def recombination(problem: TravelingSalesmanProblem, p1: Individual, p2: Individual) -> Individual:
    size = problem.get_num_cities()
    t1, t2 = p1.tour, p2.tour

    child = np.full(size, -1, dtype=int)

    a, b = sorted(random.sample(range(size), 2))
    child[a:b + 1] = t1[a:b + 1]

    t2_pos = {int(v): i for i, v in enumerate(t2)}
    in_child = set(child[a:b + 1])

    for i in range(a, b + 1):
        city = int(t2[i])
        if city in in_child:
            continue

        pos = i
        while child[pos] != -1:
            mapped = int(t1[pos])
            pos = t2_pos[mapped]

        child[pos] = city
        in_child.add(city)

    for i in range(size):
        if child[i] == -1:
            child[i] = int(t2[i])

    return Individual(problem, child, mutation_rate=p1.mutation_rate)


def elimination_lambda_plus_mu(
        population: List[Individual], offspring: List[Individual], population_size: int
) -> List[Individual]:
    combined = population + offspring
    combined.sort(key=lambda x: x.fitness)
    return combined[:population_size]


# ==============================================================
# Main GA Solver Class
# ==============================================================

class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename: str) -> int:
        distance_matrix = self._read_distance_matrix(filename)
        tsp = TravelingSalesmanProblem(distance_matrix, os.path.basename(filename))

        tsp.print_info()

        # -------------------------------------------------------------
        # Initialize population
        # -------------------------------------------------------------
        population = self._init_population(tsp)

        # Log initial population stats
        initial_fitnesses = [ind.fitness for ind in population]
        init_mean = float(np.mean(initial_fitnesses))
        init_best = float(np.min(initial_fitnesses))
        init_worst = float(np.max(initial_fitnesses))

        logger.info("Initial population:")
        logger.info(f"  Mean fitness = {init_mean:.2f}")
        logger.info(f"  Best fitness = {init_best:.2f}")
        logger.info(f"  Worst fitness = {init_worst:.2f}\n")

        best_overall = min(population, key=lambda x: x.fitness)

        # -------------------------------------------------------------
        # Main GA loop
        # -------------------------------------------------------------
        start_time = time.perf_counter()
        checkpoint_time = start_time

        for gen in range(1, GA_PARAMS["GENERATIONS"] + 1):

            offspring = self._evolve_one_generation(population, tsp, gen)

            population = elimination_lambda_plus_mu(
                population, offspring, GA_PARAMS["POPULATION_SIZE"]
            )

            gen_best = min(population, key=lambda x: x.fitness)
            gen_mean = float(np.mean([ind.fitness for ind in population]))

            if gen_best.fitness < best_overall.fitness:
                best_overall = gen_best

            # Log progress every 100 generations including time
            if gen % 100 == 0:
                now = time.perf_counter()
                elapsed = now - checkpoint_time
                checkpoint_time = now

                logger.info(
                    f"[Gen {gen}] mean={gen_mean:.2f}, best={gen_best.fitness:.2f}, "
                    f"time={elapsed:.3f}s"
                )

            time_left = self._report_and_check(gen_mean, gen_best)
            if time_left < 0:
                logger.info("Stopping early: time limit reached.")
                break

        total_time = time.perf_counter() - start_time
        logger.info(
            f"FINAL BEST = {best_overall.fitness:.2f} "
            f"(total time = {total_time:.2f}s)"
        )
        return 0

    def _read_distance_matrix(self, filename: str) -> np.ndarray:
        with open(filename, "r") as f:
            return np.loadtxt(f, delimiter=",")

    def _init_population(self, tsp: TravelingSalesmanProblem) -> List[Individual]:
        return initialize_population_greedy(tsp, GA_PARAMS["POPULATION_SIZE"])

    def _evolve_one_generation(
            self, population: List[Individual], tsp: TravelingSalesmanProblem, generation: int
    ) -> List[Individual]:
        offspring: List[Individual] = []

        # Only update adaptive mutation every 5 generations
        if generation % 5 == 0:
            diversity = population_diversity(population)

        while len(offspring) < GA_PARAMS["OFFSPRING_SIZE"]:
            p1 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])
            p2 = tournament_selection(population, GA_PARAMS["TOURNAMENT_K"])

            if random.random() < GA_PARAMS["CROSSOVER_PROB"]:
                child = recombination(tsp, p1, p2)
            else:
                child = Individual(tour=np.copy(p1.tour), mutation_rate=p1.mutation_rate)

            # Use the most recently computed diversity
            if generation % 5 == 0:
                adaptive_mutation_rate(child, diversity)

            mutation(child)
            child.evaluate(tsp)

            if not np.isinf(child.fitness):
                offspring.append(child)

        return offspring

    def _report_and_check(
            self, generation_mean_fitness: float, generation_best: Individual
    ) -> float:
        return self.reporter.report(
            generation_mean_fitness, generation_best.fitness, generation_best.tour
        )


def adaptive_mutation_by_fitness(individual: Individual, population: List[Individual]):
    best = min(population, key=lambda x: x.fitness).fitness
    worst = max(population, key=lambda x: x.fitness).fitness

    # Normalize fitness in [0,1] (0=best, 1=worst)
    norm_fitness = (individual.fitness - best) / (worst - best + 1e-8)

    individual.mutation_rate = GA_PARAMS["MUTATION_ALPHA_MIN"] + norm_fitness * (
            GA_PARAMS["MUTATION_ALPHA_MAX"] - GA_PARAMS["MUTATION_ALPHA_MIN"])


def adaptive_mutation_rate(individual: Individual, diversity: float):
    """
    Adjust mutation_rate based on diversity:
    - Low diversity → increase mutation rate
    - High diversity → keep low mutation rate
    """
    base_rate = (GA_PARAMS["MUTATION_ALPHA_MIN"] + GA_PARAMS["MUTATION_ALPHA_MAX"]) / 2
    # Less diversity → higher mutation rate
    individual.mutation_rate = min(1.0, base_rate + (1 - diversity) * 0.8)


def population_diversity(population: List[Individual]) -> float:
    """
    Returns a value in [0,1] indicating population diversity.
    1 = highly diverse, 0 = all individuals are similar.
    """
    tours = [tuple(ind.tour) for ind in population]
    unique_tours = len(set(tours))
    return unique_tours / len(population)


if __name__ == "__main__":
    print("hello world!")
