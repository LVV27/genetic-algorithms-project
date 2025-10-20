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

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distance_matrix = np.loadtxt(file, delimiter=",")
		file.close()

		filename_only = os.path.basename(filename)

		tsp_problem = TravelingSalesmanProblem(distance_matrix, filename_only)
		tsp_problem.print_info()

		# --- Initialize GA ---
		population = initialize_population(tsp_problem, GA_PARAMS["POPULATION_SIZE"])
		evaluate_population(population, tsp_problem.distance_matrix)
		best_overall_individual = min(population, key=lambda x: x.fitness)

		for gen in range(1, GA_PARAMS["GENERATIONS"] + 1):
			# Recombinate population and mutate offspring
			offspring = []
			for _ in range(GA_PARAMS["OFFSPRING_SIZE"]):
				parent1 = selection(tsp_problem, population, GA_PARAMS["TOURNAMENT_K"])
				parent2 = selection(tsp_problem, population, GA_PARAMS["TOURNAMENT_K"])

				if random.random() < GA_PARAMS["CROSSOVER_PROB"]:
					child = recombination(tsp_problem, parent1, parent2)
				else:
					child = Individual(tour=np.copy(parent1.tour), alpha=parent1.alpha)
				
				mutation(child)
				# calculate fitness once per individual, so when they are born calculate fitness and i fitness is needed just call individual.fitness effient ;))f
				child.evaluate(tsp_problem.distance_matrix)
				offspring.append(child)

			# why mutate whole population? children already get mutated
			# # Mutate population
			# for individual in population:
			# 	mutation(individual)

			# Elimination
			population = elimination_lambda_plus_mu(population, offspring, GA_PARAMS["POPULATION_SIZE"])

			generation_best = min(population, key=lambda x: x.fitness)
			generation_mean_fitness = np.mean([ind.fitness for ind in population])

			# print('Iteration: ', iterations, ', mean: ', mean_fitness, ', best: ', best_fitness, ', order: ', best_individual.order)
			# see generated csv file

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(generation_mean_fitness, generation_best.fitness, generation_best.tour)
			if timeLeft < 0:
				break
		return 0

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
		heur_val = heuristics.get(self.filename, None)
		if heur_val is not None:
			print(f"Simple greedy heuristic objective value: {heur_val}")
		else:
			print("Heuristic value: unknown")
		print("")
	
class Individual:
	def __init__(self, problem: TravelingSalesmanProblem = None, tour: np.ndarray = None, alpha : float = None):
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
		self.alpha = (
			max(
				GA_PARAMS["MUTATION_ALPHA_MIN"],
				GA_PARAMS["MUTATION_ALPHA_MIN"]
				+ (GA_PARAMS["MUTATION_ALPHA_MAX"] - GA_PARAMS["MUTATION_ALPHA_MIN"])
				* random.random(),
			)
			if alpha is None
			else alpha
		)
		self.fitness = None

	def evaluate(self, distance_matrix: np.ndarray) -> float:
		self.fitness = sum(
			distance_matrix[self.tour[i], self.tour[(i + 1) % len(self.tour)]]
			for i in range(len(self.tour))
		)
		return self.fitness

# ------------------------------
# GA Utilities
# ------------------------------
def initialize_population(problem: TravelingSalesmanProblem, population_size: int) -> list[Individual]:
	population = [Individual(problem) for _ in range(population_size)]
	return population

def evaluate_population(population: list, distance_matrix: np.ndarray):
    for ind in population:
        ind.evaluate(distance_matrix)

def fitness(problem: TravelingSalesmanProblem, individual: Individual) -> float:
    total_distance = 0
    for i in range(len(individual.tour) - 1):
        total_distance += problem.get_distance(individual.tour[i], individual.tour[i + 1])
    total_distance += problem.get_distance(individual.tour[-1], individual.tour[0])  # Return to start
    return total_distance

def mutation(individual: Individual):
    # Swap two random elements with self adaptivity parameter
    # if random.random() < individual.alpha:
	i = random.randint(0, len(individual.tour)-1)
	j = random.randint(0, len(individual.tour)-1)
	individual.tour[i], individual.tour[j] = individual.tour[j], individual.tour[i]

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
        if gene not in order: # 2. Elements of parent2 not already in offspring
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

def selection(problem: TravelingSalesmanProblem, population: list[Individual], k: int) -> Individual:
    # K-tournament selection
    candidates = random.sample(population, k) # Take k individuals from population
    best = min(candidates, key=lambda ind: fitness(problem, ind)) # Take best (lowest fitness)
    return best

# (μ + λ) elimination  
def elimination_lambda_plus_mu(population: list[Individual], offspring: list[Individual], population_size: int) -> list[Individual]:
    combined = population + offspring
    for ind in combined:
        if ind.fitness is None:
            raise ValueError("All individuals must have fitness evaluated")
    combined.sort(key=lambda x: x.fitness)
    return combined[:population_size]
