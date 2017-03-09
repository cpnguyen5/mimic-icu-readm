import db
import psycopg2
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn import metrics, grid_search
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def readDataFrame():
    """
    Function returns the contents of the pandas DataFrame.

    :return: pandas DataFrame
    """
    df = db.feat_iculos()
    return df


def data_partition(data):
    """
    Function takes a pandas DataFrame, converts it into a NumPy array, and partitions the data
    into training-testing sets.

    :param data: pandas DataFrame
    :return: tuple of partitioned dataset (X_train, X_test, y_train, y_test)
    """
    data_array = data.as_matrix()

    # Partition into 70/30 Train/Test sets
    data_train, data_test = train_test_split(data_array, random_state=2, test_size=0.30)

    n_col = data.shape[1] - 1 # last index position
    # Isolate features from labels
    X_train = data_train[:, 0:n_col] #training features
    y_train = data_train[:, n_col] #training labels
    X_test = data_test[:, 0:n_col] #testing features
    y_test = data_test[:, n_col] #testing labels
    return (X_train, X_test, y_train, y_test)


def scale(X_train, X_test):
    """
    Function takes the training & testing sets of samples/features and returns the scaled versions.

    :param X_train: Training set features
    :param X_test:  Testing set features
    :return: tuple (normalized X_train, normalized X_test)
    """
    # Fit scaler
    scaler = MinMaxScaler().fit(X_train) # scaler object fitted to training set

    # Transform
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return (scaled_X_train, scaled_X_test)


def feature_select(X_train, X_test, y_train, n_feat='all'):
    """
    Function performs univariate feature selection using sklearn.feature_selection.SelectKBest and a score function.
    SelectKBest removes all but the "k" highest scoring features. The function will return a tuple of the reduced
    features and their respective scores and p-values.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :param n_feat:  Number of features to select
    :return: tuple (selected X_train, selected X_test, scores, p-values)
    """
    # Univariate Feature Selection - chi2 (score_function)
    score_func = SelectKBest(chi2, k=n_feat).fit(X_train, y_train) #k = # features
    select_X_train = score_func.transform(X_train)
    select_X_test = score_func.transform(X_test)

    # Score Function Attributes
    scores =  score_func.scores_
    pvals = score_func.pvalues_
    return (select_X_train, select_X_test, scores, pvals)


def gridsearch(X_train, X_test, y_train):
    """
    Function determines the optimal parameters of the best classifier model/estimator by performing a grid search.
    The best model will be fitted with the Training set and subsequently used to predict the classification/labels
    of the Testing set. The function returns the "best" classifier instance, classifier predictions, best parameters,
    and grid score.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :return: tuple of (best classifier instance, clf predictions, dict of best parameters, grid score)
    """
    # Parameter Grid - dictionary of parameters (map parameter names to values to be searched)
    # param_grid = [
    #     {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
    #     # {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']},
    #     # {'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'degree': [2], 'kernel': ['poly']}
    # ]

    param_grid = [
        {'fit_intercept': [True, False], 'normalize': [True, False]}]

    # Blank clf instance
    # blank_clf = SVR()
    blank_clf = LinearRegression()

    # Grid Search - Hyperparameters Optimization
    clf = grid_search.GridSearchCV(blank_clf, param_grid, n_jobs=-1)  # classifier + optimal parameters
    clf = clf.fit(X_train, y_train)  # fitted classifier
    best_est = clf.best_estimator_
    clf_pred = best_est.predict(X_test)

    best_params = clf.best_params_  # best parameters identified by grid search
    score = clf.best_score_  # best grid score
    return (best_est, clf_pred, best_params, score)


def clf(X_train, X_test, y_train):
    """
    Function instantiates the classifier model with parameters, fits the model to the Training set, and applies it to
    the Testing set for classification predictions.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :return: tuple of (fitted classifier instance, classifier predictions).
    """
    model = LinearRegression(fit_intercept='False', normalize=True)
    model = model.fit(X_train, y_train) #Fit classifier to Training set
    y_pred = model.predict(X_test) # Test classifier on Testing set
    return (model, y_pred)


def RMSE(model_pred, target):
    """
    Function calculates the root mean square error of the regression model's prediction.
    rmse =  np.sqrt(np.mean((y_test-y_pred)**2))

    :param model_pred: model's predictions
    :param target: labels (y_test)
    :return: RMSE value
    """
    y_pred = model_pred # prediction
    y_true = target # true labels
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def MAE(model_pred, target):
    """
    Function calculates the mean absolute error of the regression model's prediction.

    :param model_pred: model's predictions
    :param target: labels (y_test)
    :return: mean absolute error
    """
    y_pred = model_pred # prediction
    y_true = target # true labels
    return metrics.median_absolute_error(y_true, y_pred)


def r2(model_pred, target):
    """
    Function calculates the R-squared score (coefficient of determination) of the regression model's prediction.

    :param model_pred: model's prediction
    :param target: labels (y_tesT)
    :return: coefficient of determination
    """
    y_pred = model_pred # prediction
    y_true = target # true labels
    return metrics.r2_score(y_true, y_pred)


# def plot_reg(X_test, y_test, y_pred):
#     """
#     Plot regression.
#
#     :param X_test: Testing set features
#     :param y_test: Testing set labels
#     :param y_pred: model's predictions
#     :return: plot
#     """
#     plt.scatter(X_test, y_test, color='black')
#     plt.plot(X_test, y_pred, color='blue', linewidth=3)
#     plt.show()
#     plt.close()


#main
if __name__ == '__main__':
    np.random.seed(2)

    # Obtain Data
    data = readDataFrame()

    # Data Partition
    X_train, X_test, y_train, y_test = data_partition(data.iloc[:500])

    # Data Normalization
    scaled_X_train, scaled_X_test = scale(X_train, X_test)

    # Feature Selection
    # # n_features = 7
    # # select_X_train, select_X_test, score, pval = feature_select(scaled_X_train, scaled_X_test, y_train, n_feat=n_features)

    # feat_names = list(data.columns)[:-1]
    # for i in range(len(feat_names)):
    #     print feat_names[i], "\n \t", "score: ", score[i], "\n \t", "p-value: ", pval[i]
    # print

    # Model
    # clf_model, y_pred, best_p, best_score = gridsearch(X_train, X_test, y_train)
    # print "Best Parameters: ", best_p
    # print "Best Grid Search Score: ", best_score
    # print "Best Estimator: ", clf_model, "\n"
    clf_model, y_pred = clf(scaled_X_train, scaled_X_test, y_train) #alternative clf function [no grid search]

    print "RMSE: ", np.around(RMSE(y_pred, y_test), 5)
    print "Mean Absolute Error: ", np.around(MAE(y_pred, y_test), 5)
    print "R2: ", np.around(r2(y_pred, y_test), 5)

