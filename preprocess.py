import numpy as np
from numpy.linalg import norm


def standardize(tx):
    """
    Standardize features by mean and standard deviation and return the standardized feature matrix.
    
    Args:
    tx (feature matrix): The matrix containing features.

    Returns:
    The standardized feature matrix.
    """
    tx_transposed = np.transpose(tx)
    standardized_matrix = np.zeros((tx.shape[1], tx.shape[0]))

    for i in range(tx.shape[1]):
        standardized_matrix[i] = (tx_transposed[i] - np.mean(tx_transposed[i])) / np.std(tx_transposed[i])

    return np.transpose(standardized_matrix)

def normalize(tx):
    """
    Normalize features to the range [0, 1] and return the normalized feature matrix.
    
    Args:
    tx (feature matrix): The matrix containing features.

    Returns:
    The normalized feature matrix.
    """
    tx_transposed = np.transpose(tx)
    normalized_matrix = np.zeros((tx.shape[1], tx.shape[0]))
    epsilon = 1e-10 

    for i in range(tx.shape[1]):
        tx_range = np.max(tx_transposed[i]) - np.min(tx_transposed[i])
        normalized_matrix[i] = (np.max(tx_transposed[i]) - tx_transposed[i]) / (tx_range + epsilon)

    return np.transpose(normalized_matrix)




def perform_pca(X, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Args:
    X (numpy.ndarray): The input data matrix where each row represents a sample, and each column represents a feature.
    n_components (int): The number of principal components to retain.

    Returns:
    X_reduced (numpy.ndarray): The reduced-dimensional data.
    """
    # Center the data (subtract the mean)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Calculate the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Select the top n eigenvectors (principal components)
    top_eigenvectors = eigenvectors[:, :n_components]

    # Project the data into the new feature space
    X_reduced = np.dot(X_centered, top_eigenvectors)

    return X_reduced


# def k_means_clustering(X_reduced, n_clusters, max_iterations=100):
#     """
#     Perform K-Means clustering on reduced-dimensional data.
    
#     Args:
#     X_reduced (numpy.ndarray): The reduced-dimensional data, where each row represents a sample, and each column represents a reduced feature.
#     n_clusters (int): The number of clusters.
#     max_iterations (int): Maximum number of iterations.

#     Returns:
#     cluster_assignment (numpy.ndarray): An array containing cluster labels for each sample.
#     """
#     np.random.seed(0)
#     initial_centers = X_reduced[np.random.choice(X_reduced.shape[0], n_clusters, replace=False)
                                
                            
#     for iteration in range(max_iterations):
#         distances = np.linalg.norm(X_reduced[:, np.newaxis] - initial_centers, axis=2)
#         cluster_assignment = np.argmin(distances, axis=1)
        
#         for cluster in range(n_clusters):
#             points_in_cluster = X_reduced[cluster_assignment == cluster]
#             if len(points_in_cluster) > 0:
#                 initial_centers[cluster] = np.mean(points_in_cluster, axis=0)
    
#     return cluster_assignment


                                
                                
# def random_forest():
#     """
#     Args:
#     - X (feature matrix): Feature matrix, where 
#     each row represents a data point, and each column represents a feature.
#     - y: Target variable, typically class labels for classification.
#     - n_trees: Number of trees in the forest.
#     - max_depth: Maximum depth for each tree.
#     - n_features: Number of randomly selected features for each tree.
#     - n_samples: Number of randomly selected samples for each tree.
    
#     Returns:
#     predicted data
#     """

#     forest = []

#     # Build multiple decision trees
#     for i in range(n_trees):
#         # Randomly select n_samples data points
#         sampled_indices = np.random.choice(X.shape[0], n_samples, replace=False)

#         X_sampled = X[sampled_indices]
#         y_sampled = y[sampled_indices]

#         # Randomly select n_features features
#         selected_features = np.random.choice(range(X.shape[1]), n_features)
#         X_selected = X_sampled[:, selected_features]

#         # Create a decision tree
#         tree = DecisionTree()
#         tree.max_depth = max_depth
#         tree.split(X_selected, y_sampled)

#         # Add the decision tree to the random forest
#         forest.append(tree)

#     # Prediction
#     predictions = []
#     for tree in forest:
#         # Make predictions using each tree
#         prediction = tree.predict(x[selected_features])
#         predictions.append(prediction)

#     # Combine the predictions from multiple trees, e.g., using majority voting
#     final_prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions_matrix)

    
    
#     return final_prediction



