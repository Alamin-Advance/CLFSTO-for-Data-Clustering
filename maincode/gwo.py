

# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# balance dataset"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import os

# GWO Algorithm
def gwo_algorithm(N, max_evaluations, lb, ub, data):
    dim = len(lb)
    alpha_pos = np.zeros(dim)
    alpha_score = float("-inf")
    beta_pos = np.zeros(dim)
    beta_score = float("-inf")
    delta_pos = np.zeros(dim)
    delta_score = float("-inf")
    T = int(np.ceil((max_evaluations - N) // (4 * N)))

    # Initialize the wolf population
    wolves = np.random.uniform(lb, ub, size=(N, dim))
    fitness = np.zeros(N)

    # Evaluate fitness of initial population
    for i in range(N):
        try:
            labels = np.random.randint(0, int(wolves[i][0]) + 1, len(data))
            fitness[i] = silhouette_score(data, labels) + 1  # Shift range to [0, 2]
        except ValueError:
            fitness[i] = float("-inf")

    # Identify alpha, beta, and delta wolves
    for i in range(N):
        if fitness[i] > alpha_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()
            beta_score = alpha_score
            beta_pos = alpha_pos.copy()
            alpha_score = fitness[i]
            alpha_pos = wolves[i].copy()
        elif fitness[i] > beta_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()
            beta_score = fitness[i]
            beta_pos = wolves[i].copy()
        elif fitness[i] > delta_score:
            delta_score = fitness[i]
            delta_pos = wolves[i].copy()

    fitness_values_per_run = []

    # Iterative optimization
    for t in range(T):
        a = 2 - t * (2 / T)  # Decreasing a linearly
        for i in range(N):
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3

            # Clamp wolves to the bounds
            wolves[i] = np.clip(wolves[i], lb, ub)

            # Update fitness
            try:
                labels = np.random.randint(0, int(wolves[i][0]) + 1, len(data))
                fitness[i] = silhouette_score(data, labels) + 1  # Shift range to [0, 2]
            except ValueError:
                fitness[i] = float("-inf")

            # Update alpha, beta, and delta wolves
            if fitness[i] > alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness[i]
                alpha_pos = wolves[i].copy()
            elif fitness[i] > beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness[i]
                beta_pos = wolves[i].copy()
            elif fitness[i] > delta_score:
                delta_score = fitness[i]
                delta_pos = wolves[i].copy()

        fitness_values_per_run.append(alpha_score)

    return alpha_pos, alpha_score, fitness_values_per_run

# Main function
def main():
    # Path to the dataset in Google Drive
    dataset_path = '/content/drive/My Drive/Colab Notebooks/Clustering/Data_set/balance-scale4.csv'
    dataset = pd.read_csv(dataset_path)

    # Preprocess the dataset
    features = dataset
    #features = dataset.iloc[:, :-1]  # Assume the last column is the target
    data = StandardScaler().fit_transform(features)

    N = 30  # Population size
    dim = data.shape[1]
    lb = np.array([-100.0] * dim)
    ub = np.array([100.0] * dim)
    #max_evaluations = 100
    num_variables = len(lb)
    #T = (max_evaluations - N) // (4 * N)
    max_evaluations = 10000 * num_variables

    output_path = "/content/drive/My Drive/Colab Notebooks/Clustering/results/gwo"
    os.makedirs(output_path, exist_ok=True)

    all_iterations = []

    for run in range(30):  # Run the GWO algorithm 30 times
        try:
            _, _, fitness_values_per_run = gwo_algorithm(N, max_evaluations, lb, ub, data)

            # Store iteration-wise fitness for this run
            for iteration, fitness in enumerate(fitness_values_per_run, start=1):
                all_iterations.append({"Iteration": iteration, "Fitness": fitness})

        except ValueError as e:
            print(f"Run {run + 1} failed with error: {e}")
            continue

    # Aggregate iteration-wise fitness across all runs
    iteration_df = pd.DataFrame(all_iterations).groupby("Iteration", as_index=False)["Fitness"].mean()

    # Save to CSV
    csv_path = os.path.join(output_path, "balance_iter_gwo.csv")
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