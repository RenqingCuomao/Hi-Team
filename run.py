import numpy as np
from helpers import *

from implementations import *
import time
%load_ext autoreload
%autoreload 2

zip_path = 'dataset_to_release.zip'
save_path = 'project1/'
file = zipfile.ZipFile(zip_path)
file.extractall(save_path)
file.close()
data_path = 'project1/dataset_to_release'

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path, sub_sample=False)
check_array_shapes(x_train, x_test, y_train, train_ids, test_ids)

tX = x_train
y = y_train
ids = train_ids
tX = tX[:, 10:]


values_to_replace_with_nan = [77, 99, 7, 9, 777, 999, 777777, 999999, 98,7777,9999,89]
values_to_replace_with_0_001 = [888,88, 8]

replace_values(tX, values_to_replace_with_nan, values_to_replace_with_0_001)
tX_NoNaN = remove_columns_with_nan(tX)

n_components = 15  # Number of principal components to retain
X_reduced_feature = perform_pca(tX_NoNaN, n_components)


n_clusters = 10  # Number of clusters
cluster_assignment = k_means_clustering(X_reduced_feature, n_clusters)
print(cluster_assignment.shape)

missing_value_threshold = 0.2
remained_columns = clean_dataset(tX, missing_value_threshold)
cleaned_tX = tX[:, remained_columns]
print(cleaned_tX.shape)

tX_filled = fill_missing_with_median(cleaned_tX, cluster_assignment)
train_medians = np.nanmedian(tX_filled, axis=0)

normalized_x = normalize(tX_filled)

w_value = 1e-1
tx_expanded, initial_w = setup_features_and_weights(normalized_x, w_value)

y_new = np.where(y == -1, 0, y)


start_time = time.time()
# Set the hyperparameters
gammas = np.logspace(-0.75, -0.67, 3)
lambdas = np.logspace(-1, 0, 3)# Regularization parameter
k_fold = 10

# Call logistic_regression_gd function for logistic regression
gamma, lambda_= best_selection(y_new, tx_expanded, initial_w, k_fold, lambdas, gammas)

end_time = time.time()

print(gamma, lambda_)
final_w, final_loss = reg_logistic_regression(y=y_new, tx=tx_expanded, initial_w=initial_w, gamma=gamma, lambda_=lambda_)
print("Final Loss:", final_loss)
print('run time:', end_time-start_time)

raw_predictions = np.dot(tx_test, final_w)
predicted_probabilities = 1 / (1 + np.exp(-raw_predictions))
tx_test = preprocess_data(x_test,values_to_replace_with_nan, values_to_replace_with_0_001, n_components, n_clusters, missing_value_threshold, remained_columns, train_medians)

test_predictions = predict(tx_test, final_w)
y_pred = np.where(test_predictions == 0, -1, test_predictions)

create_csv_submission(test_ids, y_pred, name='y_test.csv')