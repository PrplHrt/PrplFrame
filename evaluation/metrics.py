import sklearn.metrics

class r2():
    def __init__(self):
        pass

    def score(self, true, pred):
        return sklearn.metrics.r2_score(true, pred)
    

class mse():
    def __init__(self):
        pass

    def score(self, true, pred):
        return sklearn.metrics.mean_squared_error(true, pred)