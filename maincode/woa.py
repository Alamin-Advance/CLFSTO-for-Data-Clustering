
# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# glass dataset"""

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
    if len(population) == 0:
        raise ValueError("Population is empty during objective evaluation.")

    scores = []
    for individual in population:
        try:
            labels = np.random.randint(0, int(individual[0]) + 1, len(data))
            silhouette = silhouette_score(data, labels)
            normalized_fitness = silhouette + 1  # Shift range from [-1, 1] to [0, 2]
            scores.append(normalized_fitness)
        except ValueError:
            scores.append(float("-inf"))  # Assign poor fitness if silhouette fails
    return np.array(scores)

# WOA Algorithm
def whale_optimization_algorithm(N, max_evaluations, lb, ub, data):
    dim = len(lb)
    population = initialize_population(N, dim, lb, ub)
    fitness = evaluate_objective(population, data)

    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    T = int(np.ceil((max_evaluations - N) // (4 * N)))

    fitness_values_per_run = []
    # Iterative process
    for t in range(T):
        a = 2 - 2 * (t / T)  # Linearly decreases from 2 to 0
        for i in range(N):
            r = np.random.rand()
            A = 2 * a * r - a  # [-a, a]
            C = 2 * np.random.rand()  # [0, 2]

            p = np.random.rand()
            D = np.abs(C * best_solution - population[i])

            if p < 0.5:
                if np.abs(A) < 1:
                    # Shrinking encircling mechanism
                    new_position = best_solution - A * D
                else:
                    # Search for prey (random whale chosen)
                    rand_idx = np.random.randint(0, N)
                    D_rand = np.abs(C * population[rand_idx] - population[i])
                    new_position = population[rand_idx] - A * D_rand
            else:
                # Spiral updating position
                b = 1  # Spiral shape constant
                l = np.random.uniform(-1, 1)
                distance_to_best = np.abs(best_solution - population[i])
                new_position = (
                    distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
                )

            # Boundary handling
            new_position = np.clip(new_position, lb, ub)

            # Greedy selection
            new_fitness = evaluate_objective([new_position], data)[0]
            if new_fitness > fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        # Update the best solution
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]

        fitness_values_per_run.append(best_fitness)

    return best_solution, best_fitness, fitness_values_per_run

# Main function
def main():
    # Path to the dataset
    dataset_path = '/content/drive/My Drive/Colab Notebooks/Clustering/Data_set/glass.csv'
    dataset = pd.read_csv(dataset_path)

    # Preprocess the dataset
    features = dataset
    data = StandardScaler().fit_transform(features)

    N = 30  # Population size
    dim = data.shape[1]
    lb = np.array([-100.0] * dim)
    ub = np.array([100.0] * dim)
    num_variables = len(lb)
    max_evaluations = 10000 * num_variables

    output_path = "/content/drive/My Drive/Colab Notebooks/Clustering/results/woa"
    os.makedirs(output_path, exist_ok=True)

    all_iterations = []

    for run in range(30):  # Run the woa algorithm 30 times
        try:
            _, _, fitness_values_per_run = whale_optimization_algorithm(N, max_evaluations, lb, ub, data)

            # Store iteration-wise fitness for this run
            for iteration, fitness in enumerate(fitness_values_per_run, start=1):
                all_iterations.append({"Iteration": iteration, "Fitness": fitness})

        except ValueError as e:
            print(f"Run {run + 1} failed with error: {e}")
            continue

    # Aggregate iteration-wise fitness across all runs
    iteration_df = pd.DataFrame(all_iterations).groupby("Iteration", as_index=False)["Fitness"].mean()

    # Save to CSV
    csv_path = os.path.join(output_path, "glass_iter_woa.csv")
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

