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