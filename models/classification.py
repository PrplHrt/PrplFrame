from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class SVC(ClassifierMixin):
    """
    Wrapper class for svm.SVC() contained within the MultiOutputClassifier object from 
    sklearn to support multi-target classification.
    """
    def __init__(self, *args, **kwargs):
        """
        Takes in same arguments as sklearn.svm.SVC().
        """
        self.model = svm.SVC(*args, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim > 1:
            self.model = MultiOutputClassifier(self.model)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)