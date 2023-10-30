# Machine Learning Project 1 (2023): Machine learning application for coronary heart disease (MICHD)

In this project, we collected some characteristics of individual lifestyle factors and the incidence of coronary heart disease. We used regularized logistic regression to conduct classification training and predict the incidence of different populations.
Detailed project outlines and requirements can be found in the [project description](./project1_description.pdf). 

## Getting Started
Provided scripts and notebook files were built and tested with a conda environment with python version 3.8.8. 
The following external libraries are used within the scripts:

```bash
numpy (as np)
math
time
itertools
```

## Running Prerequisites
Before running the scripts and notebook files, you should keep the folder structure under folder **scripts** as follows:

```bash
  .
  ├── dataset_to_release
  │   ├── x_test.csv               # test set, extract first
  │   ├── x_train.csv              # train set, extract first
  │   └── y_train.csv              # train set, extract first
  ├── helpers.py
  ├── implementations.py
  ├── project1.ipynb
  └── run.py
```

All scripts are placed under the **scripts** folder, and you can find the code that generates our prediction file y_test.csv in `run.py`.


## Implementation Details

#### `'run.py'`
Script that contains the best algorithm implemented, with the generation of corresponding prediction file under `./y_test.csv`. It executes the following steps to get the test set prediction:

* Load the training dataset into feature matrix(x_withNaN), class labels(Y, -1 or 1), and event ids
* Data preprocessing
     
        - Delete the feature columns with NaN values, create x_NoNaN
        - Reduce the dimension of x_NoNaN with Principal Component Analysis
        - Cluster the sample with k-means clustering
        - Impute missing values of X_withNaN with medians of the cluster
        - Normalized the sub-datasets by column means and standard deviations
        - Replace -1 with 0 in Y

* Train a regularized logistic regression model with 10-fold cross validation, with an automatic process of finding the best number of learning rate (gamma), and lambda
* Load the test dasaset into feature matrix(X) 
* Compute and generate a prediction csv file `./scripts/data/pred.csv`




#### `'implementations.py'`
Script that contains the preprocess functions and the implementation of machine learning algorithms according to the following table:

| Function            | Parameters | Details |
|-------------------- |-----------|---------|
| `least_squares_GD`  | `y, tx, initial_w, max_iters, gamma`  | Linear Regression by Gradient Descent |
| `least_squares_SGD` | `y, tx, initial_w, max_iters, gamma`  | Linear Regression by Stochastic Gradient Descent |
| `least_squares`     | `y, tx` | Linear Regression by Solving Normal Equation |
| `ridge_regression`  | `y, tx, lambda_` | Ridge Regression by Soving Normal Equation |
| `logistic_regression`| `y, tx, initial_w, max_iters, gamma, threshold, batch_size` | Logistic Regression by Stochastic Gradient Descent |
| `reg_logistic_regression` | `y, tx, lambda_, initial_w, max_iters, gamma, threshold, batch_size` | Regularized Logistic Regression by Stochastic Gradient Descent |

All functions returns a set of two key values: `w, loss`, where `w` indicates the last weight vector of the algorithm, and `loss` corresponds to this weight `w`.



#### `'helpers.py'`
Script that contains the method to load csv data.

#### `'project1.ipynb'`
Notebook file contains code demonstrating our training and validation process of implemented machine learning algorithms (stated in `implementation.py`). 



## Best Performance
Our best model: Regularized Logistic Regression with imputation through k-means clustering, test accuracy: 0.911, F1 score: 0.108


## Authors
* *Shiyi Huang*
* *Jingren Tang*
* *Renqing Cuomao*