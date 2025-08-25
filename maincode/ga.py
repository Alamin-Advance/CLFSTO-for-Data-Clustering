
# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Liver dataset"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Initialize the population
def initialize_population(pop_size, dim, lb, ub):
    return np.random.uniform(lb, ub, (pop_size, dim))

# Evaluate the objective function
def evaluate_objective(population, data):
    scores = []
    for individual in population:
        try:
            labels = np.random.randint(0, max(2, int(individual[0]) + 1), len(data))
            silhouette = silhouette_score(data, labels)
            normalized_fitness = silhouette + 1  # Shift range from [-1, 1] to [0, 2]
            scores.append(normalized_fitness)
        except ValueError:
            scores.append(0.0)  # Assign poor fitness if silhouette fails
    return np.array(scores)

# Crossover operation
def crossover(parent1, parent2):
    alpha = np.random.rand()
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = alpha * parent2 + (1 - alpha) * parent1
    return offspring1, offspring2

# Mutation operation
def mutate(individual, mutation_rate, lb, ub):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-1, 1)
            individual[i] = np.clip(individual[i], lb[i], ub[i])
    return individual

# Genetic Algorithm
def genetic_algorithm(N, max_evaluations, lb, ub, data):
    dim = len(lb)
    population = initialize_population(N, dim, lb, ub)
    fitness = evaluate_objective(population, data)

    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    T = int(np.ceil((max_evaluations - N) / (4 * N)))
    mutation_rate = 0.2

    #iteration_data = []
    fitness_values_per_run = []

    for t in range(T):
        new_population = []
        for i in range(N):
            fitness_sum = np.nansum(fitness)
            probabilities = (fitness / fitness_sum) if fitness_sum > 0 else np.ones(N) / N

            parent1_idx, parent2_idx = np.random.choice(range(N), size=2, replace=False, p=probabilities)
            parent1, parent2 = population[parent1_idx], population[parent2_idx]

            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, mutation_rate, lb, ub)
            offspring2 = mutate(offspring2, mutation_rate, lb, ub)

            new_population.extend([offspring1, offspring2])

        new_population = np.array(new_population[:N])
        new_fitness = evaluate_objective(new_population, data)

        population = new_population
        fitness = new_fitness

        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]

        #iteration_data.append({"Iteration": t + 1, "Best Fitness": best_fitness})
        fitness_values_per_run.append(best_fitness)

    return best_solution, best_fitness, fitness_values_per_run

# Main function
def main():
    dataset_path = '/content/drive/My Drive/Colab Notebooks/Clustering/Data_set/liver_pro.csv'
    dataset = pd.read_csv(dataset_path)

    features = dataset
    data = StandardScaler().fit_transform(features)

    N = 30
    dim = data.shape[1]
    lb = np.array([-100.0] * dim)
    ub = np.array([100.0] * dim)
    num_variables = len(lb)
    max_evaluations = 10000 * num_variables

    output_path = "/content/drive/My Drive/Colab Notebooks/Clustering/results/ga"
    os.makedirs(output_path, exist_ok=True)

    all_iterations = []

    for run in range(30):  # Run the woa algorithm 30 times
        try:
            _, _, fitness_values_per_run = genetic_algorithm(N, max_evaluations, lb, ub, data)

            # Store iteration-wise fitness for this run
            for iteration, fitness in enumerate(fitness_values_per_run, start=1):
                all_iterations.append({"Iteration": iteration, "Fitness": fitness})

        except ValueError as e:
            print(f"Run {run + 1} failed with error: {e}")
            continue

    # Aggregate iteration-wise fitness across all runs
    iteration_df = pd.DataFrame(all_iterations).groupby("Iteration", as_index=False)["Fitness"].mean()

    # Save to CSV
    csv_path = os.path.join(output_path, "liver_iter_GA.csv")
    iteration_df.to_csv(csv_path, index=False)
    print(f"Iteration-wise fitness results saved to {csv_path}")

    # Calculate and display overall statistics
    overall_min = iteration_df["Fitness"].min()
    overall_max = iteration_df["Fitness"].max()
    overall_mean = iteration_df["Fitness"].mean()
    overall_std = iteration_df["Fitness"].std()

    summary_table = pd.DataFrame({
        "Metric": ["Overall Min", "Overall Max", "Overall Mean", "Overall Std Dev"],
        "Value": [overall_min, overall_max, overall_mean, overall_std]
    })
    print("\nFinal Summary Statistics:")
    print(summary_table)

if __name__ == "__main__":
    main()