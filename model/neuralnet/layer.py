from abc import ABC, abstractmethod
import numpy as np
from model.neuralnet.activation import Activation, Linear
from model.neuralnet.initializer import xavier_init

class Layer(ABC):

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

class Dense(Layer):

    def __init__(self, input_size: int, output_size: int, activation: Activation=Linear(), initializer=xavier_init):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.W = initializer(input_size, output_size)
        self.b = initializer(1, output_size)
        
        self.dW = None
        self.db = None
        self.X = None

    @property
    def weights(self):
        return [self.W, self.b]

    @property
    def gradients(self):
        return [self.dW, self.db]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return self.activation.forward(X @ self.W + self.b)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.activation.backward(self.X @ self.W + self.b)
        self.dW = self.X.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def __repr__(self) -> str:
        return f"Dense Layer: {self.input_size} -> {self.output_size}"

class Dropout(Layer):

    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self.mask = None

    @property
    def weights(self):
        return []
    
    @property
    def gradients(self):
        return []

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.mask = np.random.binomial(1, self.rate, size=X.shape) / self.rate
        return X * self.mask
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask

    def __repr__(self) -> str:
        return f"Dropout Layer: {self.rate}"