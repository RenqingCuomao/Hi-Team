import numpy as np
import math
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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
    _sum = np.zeros((w.shape[0], w.shape[0]))
    for i in range(y.shape[0]):
        _sum = _sum + sigmoid(tx[i, :].dot(w))*(1 - sigmoid(tx[i, :].dot(w)))*tx[i, :].reshape(-1, 1).dot(tx[i, :].reshape(1, -1))
    return _sum/y.shape[0]

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def compute_lr_gradient(y, tx, w):
    return (tx.T.dot(sigmoid(tx.dot(w)) - y))/(y.shape[0])

def compute_lr_loss(y, tx, w):
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
    e = y - tx.dot(w)
    return (e.T.dot(e))/(2*y.shape[0])


def compute_mae(y, tx, w):
    sum = 0
    for i in range(y.shape[0]):
        if y[i] - tx[i, :].dot(w) >= 0:
            sum += y[i] - tx[i, :].dot(w)

        else:
            sum += tx[i, :].dot(w) - y[i]
    return sum / (y.shape[0])

def compute_gradient_mse(y, tx, w):
    e = y - tx.dot(w)
    return -tx.dot(e)/(y.shape[0])


def mse_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        loss = compute_mse(y, tx, w)
    return w, loss

def mse_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        #get minibatch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stochastic_grad = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * stochastic_grad
            loss= compute_loss(y,tx,w)
    return w, loss

def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y-tx.dot(w)
    mse = e.T.dot(e)/(2*y.shape[0])
    return w, mse

def ridge_regression(y, tx, lambda_):
    i = np.identity(tx.shape[1])
    a = tx.T.dot(tx)+2*y.shape[0]*lambda_*i
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression_gd(y, tx, initial_w, max_iters, gamma, reg=False):
    # if reg is set True, penalty will be applied, which means 'Regularized'
    w = initial_w
    for i in range(max_iters):
        gradient = compute_lr_gradient(y, tx, w)

        if reg:
            gradient = gradient + 2 * lambda_ * w

        w = w - gamma * gradient
        print(w)
        print(gradient)
        print(gamma)
        loss = compute_lr_loss(y, tx, w)
        print(f"Iteration {i}: Loss = {loss}") 
    return w, loss

