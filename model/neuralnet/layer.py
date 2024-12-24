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

    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation: Activation=Linear(), 
        initializer=xavier_init,
        regularizer=None
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.regularizer = regularizer

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
        self.dW = self.X.T @ grad + (self.regularizer.gradient(self.W) if self.regularizer else 0)
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def __repr__(self) -> str:
        return f"Dense Layer: {self.input_size} -> {self.output_size}"

class Dropout(Layer):

    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            # X is a matrix of shape (batch_size, n)
            # mask is a matrix of shape (batch_size, n)
            # each element of mask is 1 with probability p and 0 with probability 1-p
            # scaled by p to maintain expected value
            self.mask = np.random.binomial(1, self.p, size=X.shape) / self.p
            return X * self.mask
        return X
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask
    
    def __repr__(self) -> str:
        return f"Dropout Layer: {self.p}"
        