import numpy as np

class L1:

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, W):
        return self.alpha * np.sum(np.abs(W))
    
    def gradient(self, W):
        return self.alpha * np.sign(W)
    
class L2:

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, W):
        return 0.5 * self.alpha * np.sum(W ** 2)
    
    def gradient(self, W):
        return self.alpha * W
        
        