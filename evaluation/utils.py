import itertools
from typing import Iterable
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import time
from evaluation import metrics

def load_Xy(dataset : pd.DataFrame, target: str | list[str], test_size: float | int = None, seed: int = 42):
    """
    This function takes in the dataset as a pandas DataFrame and returns the X and y values from this dataset.
    Optionally, a user can give a test_size and the dataset can be split. This function uses the sklearn
    train_test_split function, see the sklearn documentation for more info.
    
    Input:
    dataset: Pandas DataFrame - the dataset to be split
    target: String OR List of Strings - the column(s) name(s) containing the y values of the dataset
    test_size (optional): float OR int - test_size parameter in train_test_split function from sklearn
    seed (optional): int - the input to random_state parameter in train_test_split function from sklearn

    Output:
    X, y: tuple containing two numpy ndarrays containing the X and y values of the dataset
    if test_size != Null:
        X_train, X_test, y_train, y_test: tuple containing four numpy ndarrays w/ X and y values split
        into train and test
    """

    # Getting X and y as numpy ndarrays
    y = dataset[target].values
    X = dataset.drop(target, axis=1).values
    if test_size:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
    return X, y


reg_metrics = [
    metrics.mse(),
    metrics.r2(),
]

def regression_train_and_test(model: sklearn.base.RegressorMixin, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, progress: bool = False):
    """
    This function takes in a regression model with base class sklearn.base.RegressorMixin and the train and test data
    of a dataset. This function then trains and tests the model given and returns the results.
    """
    assert sklearn.base.is_regressor(model) == True, "Model should be a regressor"
    scores = {}

    # Training the model
    if progress:
        print(f'Beginning training for {type(model).__name__}')
    st = time.time()
    model.fit(X_train, y_train)
    scores['train_time'] = time.time() - st
    if progress:
        print(f'Training for {type(model).__name__} completed in {scores["train_time"]} seconds')


    # Testing the model and reporting MSE and R2
    if progress:
        print(f'Beginning testing for {type(model).__name__}')
    st = time.time()
    y_pred = model.predict(X_test)
    scores['test_time'] = time.time() - st
    scores['name'] = type(model).__name__
    for metric in reg_metrics:
        scores[type(metric).__name__] = metric.score(y_test, y_pred)

    if progress:
        print(f'Testing for {type(model).__name__} completed in {scores["test_time"]} seconds')
    
    return scores

def regression_parametric_study(
        model: sklearn.base.RegressorMixin, 
        dataset: pd.DataFrame, 
        target: str | list[str] | None = None,
        num_vals: int = 100,
        **kwargs):
    """
    For this version of the parametric study function the function will receive the dataset
    and the trained model. The function will task the model with predicting the results for the target
    variable for each predictor X with all other predictors kept at their mean value. As a result, for k
    predictors in the dataset, there will be k sets of values/predictions of varying sizes. A dictionary
    with the key being the name of the column and the values being the varied values and the results. There
    will also be a dictionary with the name of the columns and their means.

    Users can enter key word arguments to specify the range of the values for any number of columns. The argument
    should match the name of the column. For the sake of columns with names that can't be used as keyword arguments,
    users can use ci where i is the index of the column (0 indexing). The values must be entered in list format. 
    """

    if target:
        dataset = dataset.drop(target, axis=1)
    
    column_stats = pd.DataFrame({'Mean': dataset.mean(), 'Max': dataset.max(), 'Min': dataset.min()})

    base_data = dataset.mean().values
    results = {}
    for i, column in enumerate(dataset.columns):
        if column in kwargs.keys():
            var = kwargs[column]
        elif f"c{i}" in kwargs.keys():
            var = kwargs[f"c{i}"]
        else:
            var = np.linspace(column_stats['Min'].loc[column], column_stats['Max'].loc[column], num_vals)
        data = [[*base_data[:i], x, *base_data[i+1:]] for x in var]
        pred = model.predict(data)
        results[column] = (var, pred)
    return column_stats, results

def custom_parametric(model: sklearn.base.RegressorMixin,
                      dataset: pd.DataFrame,
                      values: dict,
                      target: str | list[str] | None = None):
    """
    Function designed to handle parametric studies where the values of the columns are to be defined
    by the user.  The model and dataset to be used are passed into the function alongside a dictionary
    that states the values to be used.

    If a column is not featured in the values dictionary, its mean will be used.
    If a column is featured in the values dictionary with a single value, that will be used as a base
    value.
    If a column is featured in the values dictionary with a list of values, those values will be used
    for the study.
    """
    if target:
        dataset = dataset.drop(target, axis=1)


    for column in dataset.columns:
        if column not in values.keys():
            values[column] = dataset[column].mean()
        if not isinstance(values[column], Iterable):
            values[column] = [values[column]]

    all_combinations = list(itertools.product(*values.values()))

    results = model.predict(all_combinations)
    
    return results, all_combinations