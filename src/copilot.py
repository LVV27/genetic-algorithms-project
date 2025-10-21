
import Reporter
import numpy as np
import random

class rcopilot:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def evaluate(self, individual, distanceMatrix):
        return sum(distanceMatrix[individual[i], individual[(i+1)%len(individual)]] for i in range(len(individual)))

    def initialize_population(self, pop_size, num_cities):
        return [np.random.permutation(num_cities) for _ in range(pop_size)]

    def tournament_selection(self, population, scores, k=3):
        selected = random.sample(list(zip(population, scores)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def swap_mutation(self, individual):
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
        return individual

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1]*size
        child[a:b+1] = parent1[a:b+1]
        fill = [item for item in parent2 if item not in child]
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return np.array(child)

    def optimize(self, filename):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        num_cities = len(distanceMatrix)
        pop_size = 1000
        population = self.initialize_population(pop_size, num_cities)
        scores = [self.evaluate(ind, distanceMatrix) for ind in population]

        while True:
            new_population = []
            for _ in range(pop_size):
                parent1 = self.tournament_selection(population, scores)
                parent2 = self.tournament_selection(population, scores)
                child = self.order_crossover(parent1, parent2)
                if random.random() < 0.2:
                    child = self.swap_mutation(child)
                new_population.append(child)

            population = new_population
            scores = [self.evaluate(ind, distanceMatrix) for ind in population]
            meanObjective = np.mean(scores)
            best_idx = np.argmin(scores)
            bestObjective = scores[best_idx]
            bestSolution = population[best_idx]

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0
