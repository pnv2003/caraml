class Model:

    def __init__(self):
        self.xlabels = None
        self.ylabel = None

    def fit(self, X, y):
        raise NotImplementedError()
    
    def predict(self, X):
        raise NotImplementedError()
    
    def get_xlabel(self, index):

        if self.xlabels is None:
            return "Feature " + str(index)
        return self.xlabels[index]
    
    def get_ylabel(self):

        if self.ylabel is None:
            return "Target"
        return self.y_label