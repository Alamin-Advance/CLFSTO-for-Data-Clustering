

# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Wine dataset"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import os
import math

# PSO Algorithm
def pso_algorithm(N, max_evaluations, lb, ub, data):
    dim = len(lb)
    #w = 2  # Inertia weight
    w_max, w_min = 0.9, 0.4  # Inertia weight bounds
    c1 = 2.0  # Cognitive parameter
    c2 = 2.0  # Social parameter
    T = int(np.ceil((max_evaluations - N) // (4 * N)))

    # Initialize particle positions and velocities
    positions = np.random.uniform(lb, ub, size=(N, dim))
    velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), size=(N, dim))
    personal_best_positions = positions.copy()
    personal_best_scores = np.full(N, float("-inf"))

    # Evaluate initial fitness and set personal bests
    fitness = np.zeros(N)
    for i in range(N):
        try:
            labels = np.random.randint(0, int(positions[i][0]) + 1, len(data))
            fitness[i] = silhouette_score(data, labels) + 1  # Shift range to [0, 2]
        except ValueError:
            fitness[i] = float("-inf")
        personal_best_positions[i] = positions[i].copy()
        personal_best_scores[i] = fitness[i]

    # Global best
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = np.max(personal_best_scores)

    fitness_values_per_run = []

# Iterative optimization
    for t in range(T):
        w = w_max - ((w_max - w_min) * t / T)  # Linearly decrease inertia weight

        for i in range(N):
            # Update velocity
            r1, r2 = np.random.random(dim), np.random.random(dim)
            cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
            social = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Clamp velocity to a maximum value
            max_velocity = np.abs(ub - lb) * 0.1
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Update position
            positions[i] += velocities[i]

            # Clamp position to bounds
            positions[i] = np.clip(positions[i], lb, ub)

            # Evaluate fitness
            try:
                labels = np.random.randint(0, int(positions[i][0]) + 1, len(data))
                fitness[i] = silhouette_score(data, labels) + 1  # Shift range to [0, 2]
            except ValueError:
                fitness[i] = float("-inf")

            # Update personal best
            if fitness[i] > personal_best_scores[i]:
                personal_best_positions[i] = positions[i].copy()
                personal_best_scores[i] = fitness[i]

        # Update global best
        if np.max(personal_best_scores) > global_best_score:
            global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
            global_best_score = np.max(personal_best_scores)

        fitness_values_per_run.append(global_best_score)

    return global_best_position, global_best_score, fitness_values_per_run


# Main function
def main():
    # Path to the dataset in Google Drive
    dataset_path = '/content/drive/My Drive/Colab Notebooks/Clustering/Data_set/wine.csv'
    dataset = pd.read_csv(dataset_path)

    # Preprocess the dataset
    features = dataset
    #features = dataset.iloc[:, :-1]  # Assume the last column is the target
    data = StandardScaler().fit_transform(features)

    N = 30  # Population size
    dim = data.shape[1]
    lb = np.array([-100.0] * dim)
    ub = np.array([100.0] * dim)
    #max_iterations = 500
    #max_evaluations = 1000
    num_variables = len(lb)
    #T = (max_evaluations - N) // (4 * N)
    max_evaluations = 10000 * num_variables


    output_path = "/content/drive/My Drive/Colab Notebooks/Clustering/results/pso"
    os.makedirs(output_path, exist_ok=True)

    all_iterations = []

    for run in range(30):  # Run the PSO algorithm 30 times
        try:
            _, _, fitness_values_per_run = pso_algorithm(N, max_evaluations, lb, ub, data)

            # Store iteration-wise fitness for this run
            for iteration, fitness in enumerate(fitness_values_per_run, start=1):
                all_iterations.append({"Iteration": iteration, "Fitness": fitness})

        except ValueError as e:
            print(f"Run {run + 1} failed with error: {e}")
            continue

    # Aggregate iteration-wise fitness across all runs
    iteration_df = pd.DataFrame(all_iterations).groupby("Iteration", as_index=False)["Fitness"].mean()

    # Save to CSV
    csv_path = os.path.join(output_path, "wine_iter_pso.csv")
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
