import numpy as np
import math
import itertools
from numpy.linalg import norm
def sigmoid(t):
    """
    Compute the sigmoid function.

    Args:
        t: A numpy array or scalar.

    Returns:
        The sigmoid of t.

    The sigmoid function maps any value to a value between 0 and 1 and is commonly used in logistic regression.
    """
    return 1 / (1 + np.exp(-t))

def compute_lr_gradient(y, tx, w):
    """
    Compute the gradient for logistic regression.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        w: The current weights, numpy array of shape=(D,)

    Returns:
        The gradient as a numpy array of shape=(D,)

    This function calculates the gradient of the logistic regression loss with respect to the weights.
    """
    return (tx.T.dot(sigmoid(tx.dot(w)) - y))/(y.shape[0])

def compute_lr_loss(y, tx, w):
    """
    Compute the logistic regression loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        w: The current weights, numpy array of shape=(D,)

    Returns:
        A scalar, representing the logistic regression loss.

    This function calculates the logistic regression loss, which measures the difference between the predicted and actual values.
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    y_star = sigmoid(tx.dot(w))
    _sum = 0
    for i in range(y.shape[0]):
        _sum += (y[i] * math.log(y_star[i]) + (1 - y[i]) * np.log(1 - y_star[i]))
    return - _sum/(y.shape[0])

def compute_mse(y, tx, w):
    """
    Compute the Mean Squared Error (MSE) loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        w: The current weights, numpy array of shape=(D,)

    Returns:
        A scalar, representing the MSE loss.

    This function calculates the mean squared error, a common loss function for regression problems.
    """
    e = y - tx.dot(w)
    return (e.T.dot(e))/(2*y.shape[0])


def compute_mae(y, tx, w):
    """
    Compute the Mean Absolute Error (MAE) loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        w: The current weights, numpy array of shape=(D,)

    Returns:
        A scalar, representing the MAE loss.

    The MAE loss function calculates the average absolute differences between predicted and actual values.
    """
    _sum = 0
    for i in range(y.shape[0]):
        if y[i] - tx[i, :].dot(w) >= 0:
            _sum += y[i] - tx[i, :].dot(w)

        else:
            _sum += tx[i, :].dot(w) - y[i]
    return _sum / (y.shape[0])

def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient for Mean Squared Error (MSE) loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        w: The current weights, numpy array of shape=(D,)

    Returns:
        The gradient as a numpy array of shape=(D,)

    This function calculates the gradient of the MSE loss with respect to the weights.
    """
    e = y - tx.dot(w)
    return -tx.dot(e)/(y.shape[0])


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Apply gradient descent to minimize MSE loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        initial_w: Initial weights, numpy array of shape=(D,)
        max_iters: The maximum number of iterations.
        gamma: The learning rate.

    Returns:
        w: The optimized weights.
        loss: The final loss value.

    This function applies gradient descent to minimize the MSE loss.
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        loss = compute_mse(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform stochastic gradient descent to minimize the Mean Squared Error loss.

    Args:
        y (numpy array): The output values, with shape (N,).
        tx (numpy array): The input data, with shape (N, D).
        initial_w (numpy array): Initial weights, with shape (D,).
        max_iters (int): Maximum number of iterations.
        gamma (float): Learning rate.

    Returns:
        w (numpy array): Optimized weights, with shape (D,).
        loss (float): Final MSE loss value.

    In each iteration, a single data point is randomly selected to compute the gradient
    and update the weights, aiming to minimize the MSE loss. This is repeated for max_iters
    iterations.

    Example:
        w, loss = mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        loss = compute_mse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """
    Solve the least squares problem.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)

    Returns:
        w: The solution as a numpy array of shape=(D,)
        mse: The MSE loss value.

    This function finds the optimal weights that minimize the mean squared error using the normal equations.
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y-tx.dot(w)
    mse = e.T.dot(e)/(2*y.shape[0])
    return w, mse

def ridge_regression(y, tx, lambda_):
    """
    Solve the ridge regression problem.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        lambda_: The regularization parameter.

    Returns:
        w: The solution as a numpy array of shape=(D,)
        mse: The MSE loss value with regularization.

    This function finds the optimal weights that minimize the mean squared error with L2 regularization using the normal equations.
    """
    i = np.identity(tx.shape[1])
    a = tx.T.dot(tx)+2*y.shape[0]*lambda_*i
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Apply gradient descent to minimize logistic regression loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        initial_w: Initial weights, numpy array of shape=(D,)
        max_iters: The maximum number of iterations.
        gamma: The learning rate.

    Returns:
        w: The optimized weights.
        loss: The final logistic regression loss value.

    This function applies gradient descent to minimize the logistic regression loss.
    """
    w = initial_w
    for i in range(max_iters):
        gradient = compute_lr_gradient(y, tx, w)

        w = w - gamma * gradient
        loss = compute_lr_loss(y, tx, w)
        print(f"Iteration {i}: Loss = {loss}") 
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, gamma, max_iters=1):
    """
    Apply gradient descent to minimize regularized logistic regression loss.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        lambda_: The regularization parameter.
        initial_w: Initial weights, numpy array of shape=(D,)
        max_iters: The maximum number of iterations.
        gamma: The learning rate.

    Returns:
        w: The optimized weights.
        loss: The final regularized logistic regression loss value.

    This function applies gradient descent to minimize the regularized logistic regression loss.
    """
    w = initial_w
    for i in range(max_iters):
        gradient = compute_lr_gradient(y, tx, w)
        gradient = gradient + 2 * lambda_ * w
        w = w - gamma * gradient
        loss = compute_lr_loss(y, tx, w)
    return w, loss

def cross_validation(y, x, w, k_indices, k, lambda_, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    initial_w = w
    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]
    y_tr = y[np.setdiff1d(k_indices.reshape(-1), k_indices[k])]
    x_tr = x[np.setdiff1d(k_indices.reshape(-1), k_indices[k])]
    w_tr, loss = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, gamma, max_iters=1)
    
    return w_tr, loss

def best_selection(y, x, w, k_fold, lambdas, gammas):
    # Build k indices for k-fold cross-validation using the target variable y and number of folds k_fold.
    k_indices = build_k_indices(y, k_fold)
    
    # Generate all combinations of gammas and lambdas using the itertools.product function.
    combinations = list(itertools.product(gammas, lambdas))
    
    # Initialize a list to store the average loss for each combination of parameters.
    losses = []
    
    # Loop through all combinations of gamma and lambda.
    for inx, combo in enumerate(combinations):
        gamma, lambda_ = combo
        
        # Initialize a list to store the cross-validation losses for the current combination of parameters.
        loss_cv = []
        
        # Perform k-fold cross-validation.
        for k in range(k_fold):
            # Get the training weights and loss for the current fold and parameter combination.
            w_tr, loss = cross_validation(y, x, w, k_indices, k, lambda_, gamma)
            
            # Append the loss to the loss_cv list.
            loss_cv.append(loss)
        
        # Calculate the average loss over all folds for the current parameter combination.
        losses.append(np.mean(loss_cv))
        
        # Print the progress.
        print(inx, 'done')
    
    # Find the combination of parameters that resulted in the minimum average cross-validation loss.
    gamma, lambda_ = combinations[np.argmin(losses)]
    
    # Return the best combination of gamma and lambda.
    return gamma, lambda_

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def check_array_shapes(*arrays):
    for arr in arrays:
        print(arr.shape)


def analyze_missing_values(tX, y):
    for col in range(tX.shape[1]):
        tX_T = np.transpose(tX)

        # Find the positions of missing values in that column
        null = np.isnan(tX_T[col])

        # Based on the classification result 'y', find two subsets of missing values
        null_p = np.logical_and(y >= 0, null)  # Subset for positive-classified results
        null_n = np.logical_and(y < 0, null)  # Subset for negative-classified results

        # Extract the corresponding subsets
        tX_null = tX[null]  # All rows containing missing values
        tX_null_p = tX[null_p]  # Rows with positive-classified results and missing values
        tX_null_n = tX[null_n]  # Rows with negative-classified results and missing values

        # If there are rows containing missing values
        if (tX_null.shape[0] > 0):
            # Print the percentage of missing values in that column
            print('Column', col, 'has {}% missing values'.format(tX_null.shape[0] * 100 / tX.shape[0]))
            print('P(y = 1|x contains NaN) = {:.3f}%'.format(tX_null_p.shape[0] * 100 / tX_null.shape[0]))
            print('P(y = -1|x contains NaN) = {:.3f}% \n'.format(tX_null_n.shape[0] * 100 / tX_null.shape[0]))


def replace_values(arr, values_to_replace_with_nan, values_to_replace_with_0_001):
    for value in values_to_replace_with_nan:
        arr[arr == value] = np.nan

    for value in values_to_replace_with_0_001:
        arr[arr == value] = 0.001


def remove_columns_with_nan(arr):
    nan_columns = np.any(np.isnan(arr), axis=0)
    tX_NoNaN = arr[:, ~nan_columns]
    return tX_NoNaN


def perform_pca(X, n_components):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Args:
    X (numpy.ndarray)
    n_components (int): The number of principal components to retain.

    Returns:
    X_reduced (numpy.ndarray): The reduced-dimensional data.
    """
    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Select the top n eigenvectors (principal components)
    top_eigenvectors = eigenvectors[:, :n_components]
    # Project the data into the new feature space
    X_reduced = np.dot(X_centered, top_eigenvectors)

    return X_reduced


def k_means_clustering(X_reduced, n_clusters, max_iterations=100):
    """
    Perform K-Means clustering on reduced-dimensional data.

    Args:
    X_reduced (numpy.ndarray):
    n_clusters (int): The number of clusters.
    max_iterations (int): Maximum number of iterations.

    Returns:
    cluster_assignment (numpy.ndarray): An array containing cluster labels for each sample.
    """
    np.random.seed(0)
    initial_centers = X_reduced[np.random.choice(X_reduced.shape[0], n_clusters, replace=False)]

    for iteration in range(max_iterations):
        # Calculate distances and cluster assignment
        distances = np.linalg.norm(X_reduced[:, np.newaxis] - initial_centers, axis=2)
        cluster_assignment = np.argmin(distances, axis=1)

        # Update cluster centers
        for cluster in range(n_clusters):
            points_in_cluster = X_reduced[cluster_assignment == cluster]
            if len(points_in_cluster) > 0:
                initial_centers[cluster] = np.mean(points_in_cluster, axis=0)

    return cluster_assignment


def clean_dataset(tX, missing_value_threshold):
    column_count = tX.shape[1]
    missing_value_count = np.sum(np.isnan(tX), axis=0)
    missing_value_ratio = missing_value_count / tX.shape[0]
    columns_to_remove = np.where(missing_value_ratio > missing_value_threshold)[0]
    columns_to_keep = np.where(missing_value_ratio <= missing_value_threshold)[0]
    return columns_to_keep


def fill_missing_with_median(matrix, cluster_assignment):
    filled_matrix = matrix.copy()
    unique_clusters = np.unique(cluster_assignment)
    column_medians = []

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_assignment == cluster)[0]
        cluster_data = matrix[cluster_indices]

        # Calculate the median for each column in the cluster
        median = np.nanmedian(cluster_data, axis=0)
        column_medians.append(median)

        # Find the positions of missing values within the cluster
        missing_rows, missing_cols = np.where(np.isnan(cluster_data))

        # Fill missing values with the corresponding cluster median
        for row, col in zip(missing_rows, missing_cols):
            filled_matrix[cluster_indices[row], col] = median[col]

    return filled_matrix


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


def expand_features(x):
    n, d = x.shape
    x_expanded = np.c_[np.ones((n, 1)), x]
    return x_expanded


def initialize_w(x_expanded, w_value):
    num_features = x_expanded.shape[1]
    initial_w = np.ones(num_features) * w_value
    return initial_w


def setup_features_and_weights(x, w_value):
    x_expanded = expand_features(x)
    initial_w = initialize_w(x_expanded, w_value)
    return x_expanded, initial_w

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