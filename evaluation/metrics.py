import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

class r2():
    def __init__(self):
        pass

    def score(self, true, pred,  *args, **kwargs):
        return sklearn.metrics.r2_score(true, pred, *args, **kwargs)
    

class mse():
    def __init__(self):
        pass

    def score(self, true, pred,  *args, **kwargs):
        return sklearn.metrics.mean_squared_error(true, pred,  *args, **kwargs)
    
class f1():
    def __init__(self):
        pass

    def score(self, true, pred, *args, **kwargs):
        return sklearn.metrics.f1_score(true, pred, *args, **kwargs)
    
        
class accuracy():
    def __init__(self):
        pass

    def score(self, true, pred, *args, **kwargs):
        return sklearn.metrics.accuracy_score(true, pred, *args, **kwargs)