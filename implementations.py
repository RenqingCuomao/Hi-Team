import numpy as np
import math
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Args:
        y: The output desired values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        batch_size: The number of samples per batch.
        num_batches: The total number of batches to generate.
        shuffle: If True, random shuffle the data before creating batches.

    Yields:
        minibatch_y: A minibatch of the output values.
        minibatch_tx: A minibatch of the input data.

    This function generates minibatches of the dataset to be used in mini-batch gradient descent algorithms. It ensures
    that data is optionally shuffled and that batches may overlap if a random offset is applied.
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

def calculate_hessian(y, tx, w):
     """
    Compute the Hessian matrix for logistic regression.

    Args:
        y: The output values, numpy array of shape=(N,)
        tx: The input data, numpy array of shape=(N,D)
        w: The current weights, numpy array of shape=(D,)

    Returns:
        A numpy array of shape=(D,D), representing the Hessian matrix.

    The Hessian matrix is a second-order partial derivative matrix used in optimization problems.
    """
    _sum = np.zeros((w.shape[0], w.shape[0]))
    for i in range(y.shape[0]):
        _sum = _sum + sigmoid(tx[i, :].dot(w))*(1 - sigmoid(tx[i, :].dot(w)))*tx[i, :].reshape(-1, 1).dot(tx[i, :].reshape(1, -1))
    return _sum/y.shape[0]

def sigmoid(t):
     """
    Compute the sigmoid function.

    Args:
        t: A numpy array or scalar.

    Returns:
        The sigmoid of t.

    The sigmoid function maps any value to a value between 0 and 1 and is commonly used in logistic regression.
    """
    return 1/(1 + np.exp(-t))

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
    print(tx.dot(w))
    y_star = sigmoid(tx.dot(w))
    _sum = 0
    for i in range(y.shape[0]):
        print(y_star)
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
        print(w)
        print(gradient)
        print(gamma)
        loss = compute_lr_loss(y, tx, w)
        print(f"Iteration {i}: Loss = {loss}") 
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
        print(w)
        print(gradient)
        print(gamma)
        loss = compute_lr_loss(y, tx, w)
        print(f"Iteration {i}: Loss = {loss}") 
    return w, loss
