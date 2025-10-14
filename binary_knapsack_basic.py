"""
An evolutionary algorithm to solve the 0/1 knapsack problem.

The 0/1 knapsack problem is a classic optimization problem where we aim to maximize the total value of items placed in a knapsack without exceeding its weight capacity. Each item can either be included (1) or excluded (0) from the knapsack. Each item has a specific weight and value.
"""

import numpy as np
import random


"""Representation of the binary knapsack problem"""
class KnapsackProblem:

    def __init__(self, num_objects: int):
        # Generate random values and weights (2 ^ randn(num_objects))
        self.values = 2 ** np.random.randn(num_objects)
        self.weights = 2 ** np.random.randn(num_objects)
        # Capacity = 0.25 * sum of weights
        self.capacity = 0.25 * np.sum(self.weights)
        self.num_objects = num_objects

class Individual:

    def __init__(self, problem: KnapsackProblem, order: list[int] | None = None, alpha: float = max(0.01, 0.1 + 0.02 * np.random.randn())):
        # Represent objects as a permutation
        # Start with generating a random order of objects
        self.alpha = alpha
        if not order:
            self.order = np.random.permutation(problem.num_objects)
        else: self.order = order 

class Parameters:  

    def __init__(self, lamb, mu, k, iterations):
        self.lamb = lamb                    # Population size
        self.mu = mu                        # Offspring size
        self.k = k                          # K-tournament selection
        self.iterations = iterations        # Number of iterations 



def evolutionary_algorithm(problem: KnapsackProblem, parameters: Parameters):
    # Generate individuals
    population = initialize(problem, parameters.lamb)

    for i in range(parameters.iterations):
        # Recombinate population and mutate offspring
        offspring = []
        for j in range(parameters.mu):
            parent1 = selection(problem, population, parameters.k)
            parent2 = selection(problem, population, parameters.k)
            child = recombination(problem, parent1, parent2)
            mutation(child)
            offspring.append(child)

        # Mutate population
        for individual in population:
            mutation(individual)

        # Elimination
        population = elimination(problem, population, offspring, parameters.lamb) # (Not super efficient, since overwriting veriable)
        fitnesses = [fitness(problem, individual) for individual in population]
        best_index = int(np.argmax(fitnesses))
        best_fitness = fitnesses[best_index]
        best_individual = population[best_index]
        knapsack = in_knapsack(problem, best_individual)
        print(i, ': mean fitness:', np.mean(fitnesses), ', best fitness:', best_fitness)#, ', knapsack: ', knapsack)


def initialize(problem: KnapsackProblem, population_size: int) -> list[Individual]:
    population = [Individual(problem) for _ in range(population_size)]
    return population

def fitness(problem: KnapsackProblem, individual: Individual) -> float:
    total_value = 0
    total_weight = 0
    for i in individual.order:
        if total_weight + problem.weights[i] <= problem.capacity:
            total_weight += problem.weights[i]
            total_value += problem.values[i]
        else: break
    return total_value

def in_knapsack(problem: KnapsackProblem, individual: Individual) -> list[int]:
    total_weight = 0
    items = []
    for i in individual.order:
        if total_weight + problem.weights[i] <= problem.capacity:
            total_weight += problem.weights[i]
            items.append(i)
        else: break
    return items

def mutation(individual: Individual):
    # Swap two random elements with self adaptivity parameter
    if random.random() < individual.alpha:
        i = random.randint(0, len(individual.order)-1)
        j = random.randint(0, len(individual.order)-1)
        individual.order[i], individual.order[j] = individual.order[j], individual.order[i]

def recombination(problem: KnapsackProblem, parent1: Individual, parent2: Individual) -> Individual:
    # Use the fact that the order doesn't matter (multiple orderings give the same result).
    # Use subsets for the recombination step.
    subset1 = set(in_knapsack(problem, parent1))
    subset2 = set(in_knapsack(problem, parent2))

    # Elements that are present in both subsets
    offspring = subset1 & subset2 
    
    # If object in one of the two, 50% chance to be copied in the offspring.
    symmetric_difference = subset1 ^ subset2
    for object in symmetric_difference:
        if random.random() <= 0.5:
            offspring.add(object)
    
    # Remaining objects
    all_objects = set(range(problem.num_objects))
    remaining = all_objects - offspring
    
    # Make permutation of objects, since not all might fit in the knapsack.
    offspring_list = list(offspring)
    random.shuffle(offspring_list)

    # Combine with the remaining items.
    remaining_list = list(remaining)
    random.shuffle(remaining_list)
    order = offspring_list + remaining_list

    # Mutation probabilities.
    beta = 2*random.random() - 0.5 # between -0.5 and 1.5
    alpha = parent1.alpha + beta*(parent2.alpha - parent1.alpha)

    return Individual(problem, order, alpha)

def selection(problem: KnapsackProblem, population: list[Individual], k: int) -> Individual:
    # K-tournament selection
    candidates = random.sample(population, k) # Take k individuals from population
    best = max(candidates, key=lambda ind: fitness(problem, ind)) # Take best
    return best

def elimination(problem: KnapsackProblem, population: list[Individual], offspring: list[Individual], size: int) -> list[Individual]:
    # (μ + λ) elimination  
    combined = population + offspring
    combined.sort(key=lambda ind: fitness(problem, ind), reverse=True)
    return combined[:size]

def main():
    problem = KnapsackProblem(1000)
    parameters  = Parameters(100, 100, 5, 100)
    print("Values:", problem.values)
    print("Weights:", problem.weights)
    print("Capacity:", problem.capacity)
    evolutionary_algorithm(problem, parameters)

    # Heuristic based on value by weight
    indices = list(range(len(problem.values)))
    heuristic_order = sorted(indices, key=lambda i: problem.values[i] / problem.weights[i], reverse=True)
    heurestic_best = Individual(problem, heuristic_order, 0)
    print('Heuristic objective value: ', fitness(problem, heurestic_best))

main()