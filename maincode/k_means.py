

# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Wine dataset"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
import math

# K-Means Algorithm
def kmeans_algorithm(N, max_evaluations, lb, ub, data):
    dim = len(lb)
    T = int(np.ceil((max_evaluations - N) // (4 * N)))  # Number of iterations
    max_clusters = 10  # Maximum number of clusters
    fitness_values_per_run = []

    for t in range(T):
        # Randomly choose the number of clusters
        num_clusters = np.random.randint(2, max_clusters + 1)

        try:
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=T)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Evaluate silhouette score
            silhouette = silhouette_score(data, labels)
            fitness = silhouette + 1  # Shift range from [-1, 1] to [0, 2]
        except ValueError:
            fitness = float("-inf")

        fitness_values_per_run.append(fitness)

    return fitness_values_per_run

# Main function
def main():
    # Path to the dataset in Google Drive
    dataset_path = '/content/drive/My Drive/Colab Notebooks/Clustering/Data_set/wine.csv'
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

    output_path = "/content/drive/My Drive/Colab Notebooks/Clustering/results/kmeans"
    os.makedirs(output_path, exist_ok=True)

    all_iterations = []

    for run in range(30):  # Run the K-Means algorithm 30 times
        try:
            fitness_values_per_run = kmeans_algorithm(N, max_evaluations, lb, ub, data)

            # Store iteration-wise fitness for this run
            for iteration, fitness in enumerate(fitness_values_per_run, start=1):
                all_iterations.append({"Iteration": iteration, "Fitness": fitness})

        except ValueError as e:
            print(f"Run {run + 1} failed with error: {e}")
            continue

    # Aggregate iteration-wise fitness across all runs
    iteration_df = pd.DataFrame(all_iterations).groupby("Iteration", as_index=False)["Fitness"].mean()

    # Save to CSV
    csv_path = os.path.join(output_path, "wine_iter_kmeans1.csv")
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
