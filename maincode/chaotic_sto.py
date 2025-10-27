
"""# Liver dataset"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import os

# Chaotic Logistic Map
def chaotic_logistic_map(size, x0=0.5, r=4):
    chaos = np.zeros(size)
    chaos[0] = x0
    for i in range(1, size):
        chaos[i] = r * chaos[i - 1] * (1 - chaos[i - 1])
    return chaos

# Function to initialize the population
def initialize_population(N, num_variables, lb, ub):
    return np.random.uniform(lb, ub, size=(N, num_variables))

# Evaluate objective (normalized for positive values)
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

# Greedy selection for maximization
def greedy_selection(new_sol, sol, fitness, data):
    try:
        new_fitness = evaluate_objective([new_sol], data)[0]
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        new_fitness = float("-inf")
    return (new_sol, new_fitness) if new_fitness > fitness else (sol, fitness)

# STO Algorithm
def sto_algorithm(N, max_evaluations, lb, ub, data):
    num_variables = len(lb)
    T = int(np.ceil((max_evaluations - N) // (4 * N)))
    #T= 500
    population = initialize_population(N, num_variables, lb, ub)

    fitness = evaluate_objective(population, data)
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    fitness_values_per_run = []
    chaos_sequence = chaotic_logistic_map(max_evaluations)

    for t in range(T):
        for i in range(N):
            # Phase 1: Prey hunting
            Xbest = population[np.argmax(fitness)]
            PP = population[np.where(fitness > fitness[i])]
            if len(PP) == 0:
                PP = np.append(PP, [Xbest], axis=0)
            TP = PP[np.random.randint(len(PP))]
            l = np.random.randint(1, 3, size=num_variables)
            r = chaos_sequence[t % len(chaos_sequence)]

            P1S1 = population[i] + r * (TP - l * Xbest)
            P1S1 = np.clip(P1S1, lb, ub)
            population[i], fitness[i] = greedy_selection(P1S1, population[i], fitness[i], data)

            r = chaos_sequence[(t + 1) % len(chaos_sequence)]
            P1S2 = population[i] + (r * (ub - lb)) / (t + 1)
            P1S2 = np.clip(P1S2, lb, ub)
            population[i], fitness[i] = greedy_selection(P1S2, population[i], fitness[i], data)

            # Phase 2: Fighting with bear
            k = np.random.choice([x for x in range(N) if x != i])
            r = chaos_sequence[(t + 2) % len(chaos_sequence)]
            if fitness[k] > fitness[i]:
                P2S1 = population[i] + r * (population[k] - l * population[i])
            else:
                P2S1 = population[i] + r * (population[i] - l * population[k])
            P2S1 = np.clip(P2S1, lb, ub)
            population[i], fitness[i] = greedy_selection(P2S1, population[i], fitness[i], data)

            P2S2 = population[i] + (r * (ub - lb)) / (t + 1)
            P2S2 = np.clip(P2S2, lb, ub)
            population[i], fitness[i] = greedy_selection(P2S2, population[i], fitness[i], data)

        fitness_values_per_run.append(np.max(fitness))

        # Update the best solution
        if np.max(fitness) > best_fitness:
            best_fitness = np.max(fitness)
            best_solution = population[np.argmax(fitness)]

    return best_solution, best_fitness, fitness_values_per_run

# Main function
def main():
    # Path to the dataset in Google Drive
    dataset_path = '/content/drive/My Drive/Colab Notebooks/Clustering/Data_set/liver_pro.csv'
    dataset = pd.read_csv(dataset_path)

    # Preprocess the dataset
    features = dataset
    #features = dataset.iloc[:, :-1]  # Assume the last column is the target
    data = StandardScaler().fit_transform(features)

    N = 30  # Population size
    dim = data.shape[1]
    lb = np.array([-100.0] * dim)
    ub = np.array([100.0] * dim)
    num_variables = len(lb)
    #T = (max_evaluations - N) // (4 * N)
    max_evaluations = 10000 * num_variables

    output_path = "/content/drive/My Drive/Colab Notebooks/Clustering/results/sto"
    os.makedirs(output_path, exist_ok=True)

    all_iterations = []

    for run in range(30):  # Run the STO algorithm 30 times
        try:
            _, _, fitness_values_per_run = sto_algorithm(N, max_evaluations, lb, ub, data)

            # Store iteration-wise fitness for this run
            for iteration, fitness in enumerate(fitness_values_per_run, start=1):
                all_iterations.append({"Iteration": iteration, "Fitness": fitness})

        except ValueError as e:
            print(f"Run {run + 1} failed with error: {e}")
            continue

    # Aggregate iteration-wise fitness across all runs
    iteration_df = pd.DataFrame(all_iterations).groupby("Iteration", as_index=False)["Fitness"].mean()

    # Save to CSV
    csv_path = os.path.join(output_path, "liver_iter_sto.csv")
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