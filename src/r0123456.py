import Reporter
import numpy as np
import random

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

		# Your code here.

	

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		problem = TravelingsalesmanProblem(distanceMatrix)
		# Your code here.
		MAX_ITERATIONS = 1000
		iterations = 0
		LAMBDA = 100
		MU = 80
		K = 3

		# Generate individuals
		population = initialize(problem, LAMBDA)

		while( iterations < MAX_ITERATIONS ):
			# Recombinate population and mutate offspring
			offspring = []
			for j in range(MU):
				parent1 = selection(problem, population, K)
				parent2 = selection(problem, population, K)
				child = recombination(problem, parent1, parent2)
				mutation(child)
				offspring.append(child)

			# Mutate population
			for individual in population:
				mutation(individual)

			# Elimination
			population = elimination(problem, population, offspring, LAMBDA) # (Not super efficient, since overwriting veriable)
			fitnesses = [fitness(problem, individual) for individual in population]
			best_index = int(np.argmin(fitnesses))
			best_fitness = fitnesses[best_index]
			best_individual = population[best_index]
			mean_fitness = np.mean(fitnesses)

			print('Iteration: ', iterations, ', mean: ', mean_fitness, ', best: ', best_fitness, ', order: ', best_individual.order)

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(best_fitness, mean_fitness, best_individual.order)
			if timeLeft < 0:
				break

			iterations += 1
		return 0

class TravelingsalesmanProblem:
	def __init__(self, distance_matrix: np.ndarray):
		self.distance_matrix = distance_matrix
		self.num_cities = distance_matrix.shape[0]

	def get_distance(self, city1: int, city2: int) -> float:
		return self.distance_matrix[city1, city2]

	def get_num_cities(self) -> int:
		return self.num_cities
	
class Individual:
	def __init__(self, problem: TravelingsalesmanProblem, order: np.ndarray | None = None):#, alpha: float = max(0.01, 0.1 + 0.02 * np.random.randn())):
		# Represent objects as a permutation
		# Start with generating a random order of objects
		# self.alpha = alpha
		if order is None:
			self.order = np.random.permutation(problem.num_cities)
		else: self.order = order 


def initialize(problem: TravelingsalesmanProblem, population_size: int) -> list[Individual]:
	population = [Individual(problem) for _ in range(population_size)]
	return population

def fitness(problem: TravelingsalesmanProblem, individual: Individual) -> float:
    total_distance = 0
    for i in range(len(individual.order) - 1):
        total_distance += problem.get_distance(individual.order[i], individual.order[i + 1])
    total_distance += problem.get_distance(individual.order[-1], individual.order[0])  # Return to start
    return total_distance

def mutation(individual: Individual):
    # Swap two random elements with self adaptivity parameter
    # if random.random() < individual.alpha:
	i = random.randint(0, len(individual.order)-1)
	j = random.randint(0, len(individual.order)-1)
	individual.order[i], individual.order[j] = individual.order[j], individual.order[i]

def recombination(problem: TravelingsalesmanProblem, parent1: Individual, parent2: Individual) -> Individual:
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
        order[i] = parent1.order[i]

    # 2-5. Map the values from parent2’s segment
    for i in range(cx_point1, cx_point2 + 1):
        gene = parent2.order[i]
        if gene not in order: # 2. Elements of parent2 not already in offspring
            pos = i
            # 3-5. Follow the mapping from parent1 to parent2 until we find a free spot
            while order[pos] != -1:
                mapped_gene = parent1.order[pos]
                pos = int(np.where(parent2.order == mapped_gene)[0][0])
            order[pos] = gene

	# 6. Fill remaining empty positions with parent2’s genes
    for i in range(size):
        if order[i] == -1:
            order[i] = parent2.order[i]
		
    return Individual(problem, order)

def selection(problem: TravelingsalesmanProblem, population: list[Individual], k: int) -> Individual:
    # K-tournament selection
    candidates = random.sample(population, k) # Take k individuals from population
    best = min(candidates, key=lambda ind: fitness(problem, ind)) # Take best (lowest fitness)
    return best

def elimination(problem: TravelingsalesmanProblem, population: list[Individual], offspring: list[Individual], size: int) -> list[Individual]:
    # (μ + λ) elimination  
    combined = population + offspring
    combined.sort(key=lambda ind: fitness(problem, ind)) # Sort on fitness (lowest is best)
    return combined[:size]