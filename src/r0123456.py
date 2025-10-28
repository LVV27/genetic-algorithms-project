import numpy as np
import Reporter
import time 
from numba import njit

MAX_GENERATIONS = 100000
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 5
SELECTION_SIZE = 75
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05


class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):
        # Read distance matrix from file
        distanceMatrix = np.loadtxt(filename, delimiter=",")
        
        best_cost = float('inf')
        number_nodes = distanceMatrix.shape[0]
        population = self.initialize_population(number_nodes)

        generation = 0
        while generation < MAX_GENERATIONS:
            generation += 1

            # 1. Evaluate the population fitness
            fitness_scores = [self.calculate_tour_cost(tour, distanceMatrix) for tour in population]  

            # 2. Select parents using tournament selection
            selected_parents = self.tournament_selection(population, fitness_scores, TOURNAMENT_SIZE, SELECTION_SIZE)  

            # 3. Generate offspring through crossover and mutation
            new_population = []
            for i in range(0, len(selected_parents), 2):
                
                parent1 = selected_parents[i]
                parent2 = selected_parents[i+1] if i+1 < len(selected_parents) else selected_parents[0]

                offspring_1 = parent1
                offspring_2 = parent2

                # Apply crossover to create offspring
                if np.random.rand() < CROSSOVER_RATE:
                    
                    offspring_1 = self.order_crossover(parent1, parent2)
                    offspring_2 = self.order_crossover(parent1, parent2)
                     

                # Apply mutation
                if np.random.rand() < MUTATION_RATE:
                    offspring_1 = self.mutate(parent1)
                    offspring_2 = self.mutate(parent2)
                    

                new_population.append(offspring_1)
                new_population.append(offspring_2)

            # 4. Elimination
            population = self.tournament_selection(new_population, fitness_scores, TOURNAMENT_SIZE, POPULATION_SIZE)

            # 5. Evaluate the new population and find the best solution
            for tour in population:
                tour_cost = self.calculate_tour_cost(tour, distanceMatrix)
                if tour_cost < best_cost:
                    best_cost = tour_cost
                    best_solution = tour    


            # Call the reporter
            timeLeft = self.reporter.report(0, best_cost, best_solution)
            if timeLeft < 0:
                generation = float('inf')

            print(f"\r Time Left: {round(timeLeft,1)}, Generation: {generation}, Best Cost: {round(best_cost, 2)}", end="")

        print("Done")
        return best_solution

    def tournament_selection(self, population, fitness_scores, tournament_size, selection_size):
        selected = []
        for _ in range(selection_size):
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_index])
        return selected


    def mutate(self, solution):
        i, j = np.random.choice(len(solution), size=2, replace=False)
        solution[i], solution[j] = solution[j], solution[i]
        return solution     
    
    def calculate_tour_cost(self, tour, distanceMatrix):
        extended_tour = np.empty(len(tour) + 1, dtype=tour.dtype)
        extended_tour[:-1] = tour
        extended_tour[-1] = tour[0]
        
        cost = 0.0
        for i in range(len(extended_tour) - 1):
            cost += distanceMatrix[extended_tour[i], extended_tour[i + 1]]
        
        return cost

    def initialize_population(self, number_nodes):
        population = []
        for _ in range(POPULATION_SIZE):
            solution = np.arange(number_nodes)
            np.random.shuffle(solution)
            population.append(solution)

        return population
    
    @staticmethod
    @njit
    def order_crossover(parent1, parent2):
        size = len(parent1)
        child = np.full(size, -1)
        start, end = sorted(np.random.choice(np.arange(size), 2, replace=False))

        # Copy part of route from parent1 to child
        child[start:end] = parent1[start:end]

        # Map the nodes from parent2 to the child
        for i in range(start, end):
            if parent2[i] not in child:
                pos = i
                while child[pos % size] != -1:
                    pos += 1
                child[pos % size] = parent2[i]

        # Fill remaining positions with values from parent2
        for i in range(size):
            if child[i] == -1:
                for value in parent2:
                    if value not in child:
                        child[i] = value
                        break

        return child

