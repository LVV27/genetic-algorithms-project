import Reporter
import numpy as np

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
		MAX_ITERATIONS = 25
		iterations = 0
		LAMBDA = 5
		MU = 80

		# Generate individuals
		population = initialize(problem, LAMBDA)
		print(population)
		print(fitness(problem, population[0]))

		# while( iterations < MAX_ITERATIONS ):
		# 	# Recombinate population and mutate offspring
		# 	offspring = []
		# 	for j in range(MU):
		# 		parent1 = selection(problem, population, parameters.k)
		# 		parent2 = selection(problem, population, parameters.k)
		# 		child = recombination(problem, parent1, parent2)
		# 		mutation(child)
		# 		offspring.append(child)

		# 	# Mutate population
		# 	for individual in population:
		# 		mutation(individual)

		# 	# Elimination
		# 	population = elimination(problem, population, offspring, parameters.lamb) # (Not super efficient, since overwriting veriable)
		# 	fitnesses = [fitness(problem, individual) for individual in population]
		# 	best_index = int(np.argmax(fitnesses))
		# 	best_fitness = fitnesses[best_index]
		# 	best_individual = population[best_index]
		# 	knapsack = in_knapsack(problem, best_individual)
		# 	print(i, ': mean fitness:', np.mean(fitnesses), ', best fitness:', best_fitness)#, ', knapsack: ', knapsack)
		# 	print(iterations)
		# 	meanObjective = 0.0
		# 	bestObjective = 0.0
		# 	bestSolution = np.array([1,2,3,4,5])

		# 	# Your code here.

		# 	# Call the reporter with:
		# 	#  - the mean objective function value of the population
		# 	#  - the best objective function value of the population
		# 	#  - a 1D numpy array in the cycle notation containing the best solution 
		# 	#    with city numbering starting from 0
		# 	timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
		# 	if timeLeft < 0:
		# 		break

		# 	iterations += 1
		# return 0

class TravelingsalesmanProblem:
	def __init__(self, distance_matrix: np.ndarray):
		self.distance_matrix = distance_matrix
		self.num_cities = distance_matrix.shape[0]

	def get_distance(self, city1: int, city2: int) -> float:
		return self.distance_matrix[city1, city2]

	def get_num_cities(self) -> int:
		return self.num_cities
	
class Individual:
	def __init__(self, problem: TravelingsalesmanProblem, order: list[int] | None = None):#, alpha: float = max(0.01, 0.1 + 0.02 * np.random.randn())):
		# Represent objects as a permutation
		# Start with generating a random order of objects
		# self.alpha = alpha
		if not order:
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
