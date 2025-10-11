import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer


def plot_kmeans(X, n_clusters, max_iter=300):
    
    # Randomly initialize centroids from the data points
    rng = np.random.RandomState(40)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centroids = X[i]

    for i in range(max_iter):
        # Step 1: Assign points to the nearest centroid
        labels = pairwise_distances_argmin(X, centroids)

        # Step 2: Plot the points with their current cluster assignments
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
        plt.title(f"Iteration {i+1}")
        
        # Save the plot as an image
        plt.savefig(f"kmeans_iter_{i+1}.png")
        plt.show()

        # Step 3: Update centroids
        new_centroids = np.array([X[labels == j].mean(0) for j in range(n_clusters)])

        # Break if the centroids have stopped moving
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids



def run_visualizer():
    # Generate synthetic data
    X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.0)

    # KMeans model
    model = KMeans(n_clusters=3)

    # Visualize KMeans with Yellowbrick
    visualizer = SilhouetteVisualizer(model)
    visualizer.fit(X)
    visualizer.show()

    plt.show()


def run():
    # Generate a dataset with 3 clusters
    X, y = make_blobs(n_samples=450, centers=3, random_state=42, cluster_std=2.0)

    # Plot the dataset without clustering information
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30)
    plt.title("Initial Data")
    plt.show()

    command = input("enter y to start, n to recreate and anything else to exit")

    if command == 'n':
        run()
    elif command == 'y':
        # Run the function with the generated data
        plot_kmeans(X, n_clusters=3)

if __name__=="__main__":
    run_visualizer()