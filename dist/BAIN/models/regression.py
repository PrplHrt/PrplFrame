from sklearn.linear_model import Ridge, Lasso
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import RegressorMixin
import numpy as np

class PolynomialRegression(RegressorMixin):
    """
    PolynomialRegression class used as a wrapper for a Pipeline object that represents
    the polynomial regression.
    """
    def __init__(self, degree: int = 3, fit_intercept: bool = False):
        self.model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression(fit_intercept=fit_intercept))])
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class SVR(RegressorMixin):
    """
    Wrapper class for svm.SVR() contained within the MultiOutputRegressor object from 
    sklearn to support multi-target regression.
    """
    def __init__(self):
        self.model = svm.SVR()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim > 1:
            self.model = MultiOutputRegressor(svm.SVR())
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)