import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin

# ----------------------------
# K-means iterative plotting with explicit steps
# ----------------------------
def plot_kmeans(X, n_clusters, max_iter=300):
    
    # Create folder for saving figures
    save_folder = "kmeans_in_action_figures"
    os.makedirs(save_folder, exist_ok=True)

    # Randomly initialize centroids from the data points
    rng = np.random.RandomState(40)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centroids = X[i]

    for iteration in range(max_iter):

        # -------------------------------
        # Assignment Step
        # -------------------------------
        labels = pairwise_distances_argmin(X, centroids)

        plt.figure(figsize=(6,5))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                    c='red', marker='x', s=200, label="Centroids")
        plt.title(f"K-means Assignment Step (Iteration {iteration+1})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_folder}/kmeans_assignment_iter_{iteration+1}.png")
        plt.close()

        # -------------------------------
        # Update Step
        # -------------------------------
        new_centroids = np.vstack([
            X[labels == j].mean(axis=0) for j in range(n_clusters)
        ])

        plt.figure(figsize=(6,5))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1], 
                    c='blue', marker='x', s=200, label="Updated Centroids")
        plt.title(f"K-means Update Step (Iteration {iteration+1})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_folder}/kmeans_update_iter_{iteration+1}.png")
        plt.close()

        # -------------------------------
        # Convergence check
        # -------------------------------
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration+1}")
            break

        centroids = new_centroids

# ----------------------------
# Main execution
# ----------------------------
def run():
    # Generate dataset
    X, y = make_blobs(n_samples=450, centers=3, random_state=42, cluster_std=2.0)

    # Save initial data plot (same size, theme, style)
    save_folder = "kmeans_in_action_figures"
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=30, cmap='viridis')
    plt.title("Initial Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/kmeans_initial_data.png")
    plt.close()

    # Run K-means iterative plotting
    plot_kmeans(X, n_clusters=3)

# ----------------------------
# Entry point
# ----------------------------
if __name__=="__main__":
    run()
