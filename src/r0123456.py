import Reporter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SELECTION_NODES = 10
MUTATION_AMOUNT = 10

# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.all_best_costs = []
        self.all_mean_costs = []
        
    # Helper function to calculate the total cost of a given solution (route)
    def calculate_total_cost(self, solution, matrix):
        """Calculates the total cost of a solution cycle."""
        cost = 0
        num_nodes = len(solution)
        for i in range(num_nodes):
            current_node = solution[i]
            next_node = solution[(i + 1) % num_nodes] 
            cost += matrix[current_node, next_node]
        return cost  
        

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        # Initialize
        num_nodes = distanceMatrix.shape[0]
        # Inital random solution
        current_solution = np.arange(num_nodes)
        np.random.shuffle(current_solution)
        
        best_solution = np.copy(current_solution)
        best_cost = self.calculate_total_cost(current_solution, distanceMatrix)

        timeLeft = self.reporter.allowedTime 
        iteration = 0
        # Convergence algorithm

        while( timeLeft > 0 ):
            
            #1: Calculate node costs
            node_costs = np.zeros(num_nodes)
            for i in range(num_nodes):
                previous_node = current_solution[i-1]
                current_node = current_solution[i]
                next_node = current_solution[(i+1) % num_nodes]
                node_costs[i] = distanceMatrix[previous_node, current_node] + distanceMatrix[current_node, next_node]
                
            # 2. Extract inifinite cost nodes
            infinite_cost_indices = np.where(np.isinf(node_costs))[0]
            infinite_cost_nodes = current_solution[infinite_cost_indices]
            selection_nodes = []
            if len(infinite_cost_nodes) <= SELECTION_NODES:
                selection_nodes.extend(infinite_cost_nodes)
            else:
                selection_nodes.extend(np.random.choice(infinite_cost_nodes, SELECTION_NODES, replace=False))
            
            # 3. Fitness selection non infinite cost nodes
            finite_cost_indices = np.where(~np.isinf(node_costs))[0]
            nodes_to_select = SELECTION_NODES - len(selection_nodes)
            if nodes_to_select > 0 and len(finite_cost_indices) > 0:
                finite_cost_nodes = current_solution[finite_cost_indices]
                finite_node_costs = node_costs[finite_cost_indices]
                if nodes_to_select > len(finite_cost_indices):
                    nodes_to_select = len(finite_cost_indices)
                total_fitness = np.sum(finite_node_costs)
                
                if total_fitness > 0:
                    probabilities = finite_node_costs / total_fitness
                    chosen_finite_cost_nodes = np.random.choice(finite_cost_nodes, nodes_to_select, replace=False, p=probabilities)
                    selection_nodes.extend(chosen_finite_cost_nodes)
                else:
                    chosen_finite_cost_nodes = np.random.choice(finite_cost_nodes, nodes_to_select, replace=False)
                    selection_nodes.extend(chosen_finite_cost_nodes)
                
            ### Mutation
            mutated_solutions = []
            
            # 1. Mutate selected nodes
            indices_selection_nodes = [np.where(current_solution == node)[0][0] for node in selection_nodes]
            for _ in range(MUTATION_AMOUNT):
                permutated_nodes = np.random.permutation(selection_nodes)
                mutation_solution = np.copy(current_solution)
                for index, value in enumerate(indices_selection_nodes):
                    mutation_solution[value] = permutated_nodes[index]
                mutated_solutions.append(mutation_solution)
            
            # 2. Include original solution
            mutated_solutions.append(current_solution)
            
            # 3. Calculate cost mutated solution
            mutation_costs = [self.calculate_total_cost(sol, distanceMatrix) for sol in mutated_solutions]
            
   
            # 4. Choose best mutation best on cost
            best_mutation_cost = min(mutation_costs)
            mean_mutation_cost = np.mean(mutation_costs)
            best_mutation_index = mutation_costs.index(best_mutation_cost)
            current_solution = mutated_solutions[best_mutation_index]
            
            if (best_mutation_cost < best_cost):
                best_cost = best_mutation_cost
                best_solution = current_solution
                print(f"Iter {iteration}: New Best Cost = {best_cost:.2f} (Mean: {mean_mutation_cost:.2f})")
                
            self.all_best_costs.append(best_cost)
            self.all_mean_costs.append(mean_mutation_cost)
            
            meanObjective = mean_mutation_cost
            bestObjective = best_cost
            
        

            timeLeft = self.reporter.report(meanObjective, bestObjective, current_solution)
            iteration += 1
            
        print(f"\nOptimization finished. Final best cost: {best_cost}")
        
        # --- Plotting ---
        self.plot_convergence()

        # Return the best solution
        return best_solution

    def plot_convergence(self):
        """Saves a convergence plot to a PNG file."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.all_best_costs, label="Best Overall Cost", color='blue', linewidth=2)
            plt.plot(self.all_mean_costs, label="Mean Mutation Cost per Iteration", color='orange', linestyle='--', alpha=0.8)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title(f"Convergence Plot: {self.reporter.filename.replace('.csv', '')}")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            plot_filename = self.reporter.filename.replace('.csv', '.png')
            plt.savefig(plot_filename)
            print(f"Convergence plot saved to {plot_filename}")
        except Exception as e:
            print(f"Error generating plot: {e}")

