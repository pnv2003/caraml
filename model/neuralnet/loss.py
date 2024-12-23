from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):

    @abstractmethod
    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        pass

class MSE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((preds - targets) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets) / len(preds)
    
class CrossEntropy(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return -np.sum(targets * np.log(preds)) / len(preds)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (preds - targets) / len(preds)
    
class BinaryCrossEntropy(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return -np.sum(targets * np.log(preds) + (1 - targets) * np.log(1 - preds)) / len(preds)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (preds - targets) / len(preds)
    
class KLDivergence(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(targets * np.log(targets / preds))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return -targets / preds
    
class Huber(LossFunction):

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        diff = np.abs(preds - targets)
        return np.sum(np.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta)))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        diff = preds - targets
        return np.where(np.abs(diff) < self.delta, diff, np.sign(diff) * self.delta)
    
class Hinge(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.maximum(0, 1 - targets * preds))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return -targets * (preds < 1)
    
class SquaredHinge(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.maximum(0, 1 - targets * preds) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return -2 * targets * np.maximum(0, 1 - targets * preds)
    
class LogCosh(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.log(np.cosh(preds - targets)))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.tanh(preds - targets)
    
class Poisson(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(preds - targets * np.log(preds))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return preds - targets
    
class KMeans(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum((preds - targets) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets)
    
class L1(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.abs(preds - targets))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.sign(preds - targets)
    
class L2(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum((preds - targets) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets)
    
class ElasticNet(LossFunction):

    def __init__(self, alpha: float = 0.5, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum((preds - targets) ** 2) + self.alpha * (
            self.l1_ratio * np.sum(np.abs(preds - targets)) + (1 - self.l1_ratio) * np.sum((preds - targets) ** 2)
        )

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets) + self.alpha * (
            self.l1_ratio * np.sign(preds - targets) + (1 - self.l1_ratio) * 2 * (preds - targets)
        )
    
class LogLoss(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return -np.sum(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (preds - targets) / (preds * (1 - preds))
    
class MAE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.abs(preds - targets))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.sign(preds - targets)
    
class MAPE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.abs((preds - targets) / targets))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.sign(preds - targets) / targets
    
class MASE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(np.abs(preds - targets)) / np.sum(np.abs(np.diff(targets)))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.sign(preds - targets) / np.sum(np.abs(np.diff(targets)))
    
class MSLE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum((np.log(preds + 1) - np.log(targets + 1)) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (np.log(preds + 1) - np.log(targets + 1)) / (preds + 1)
    
class RMSLE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sqrt(np.sum((np.log(preds + 1) - np.log(targets + 1)) ** 2))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (np.log(preds + 1) - np.log(targets + 1)) / (np.sqrt(preds + 1) * np.sqrt(targets + 1))
    
class MSPE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sum(((preds - targets) / targets) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets) / (targets ** 2)
    
class RMSE(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.sqrt(np.sum((preds - targets) ** 2))

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets)
    
class R2(LossFunction):

    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return 1 - np.sum((preds - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

    def backward(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (preds - targets) / np.sum((targets - np.mean(targets)) ** 2)