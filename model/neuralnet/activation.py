from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):

    @abstractmethod
    def forward(self, input):
        pass
    
    @abstractmethod
    def backward(self, grad):
        pass

class Linear(Activation):

    def forward(self, input):
        return input
    
    def backward(self, grad):
        return grad    

class Sigmoid(Activation):

    def forward(self, input):
        return 1 / (1 + np.exp(-input))
    
    def backward(self, grad):
        return grad * (1 - grad)
    
class ReLU(Activation):

    def forward(self, input):
        return np.maximum(0, input)
    
    def backward(self, grad):
        return grad * (grad > 0)
    
class Softmax(Activation):

    def forward(self, input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def backward(self, grad):
        return grad
    
class Tanh(Activation):

    def forward(self, input):
        return np.tanh(input)
    
    def backward(self, grad):
        return 1 - grad ** 2
    
class LeakyReLU(Activation):

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, input):
        return np.where(input > 0, input, self.alpha * input)
    
    def backward(self, grad):
        return np.where(grad > 0, 1, self.alpha)
    
# Clevert et al. 2015 https://arxiv.org/pdf/1511.07289.pdfs
class ELU(Activation):

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, input):
        return np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
    
    def backward(self, grad):
        return np.where(grad > 0, 1, self.alpha * np.exp(grad))
    
# Ramachandran et al. 2017 https://arxiv.org/pdf/1702.03118.pdf
class Swish(Activation):

    def forward(self, input):
        return input / (1 + np.exp(-input))
    
    def backward(self, grad):
        return grad + (1 - grad) / (1 + np.exp(-grad))
    
# Misra et al. 2019 https://arxiv.org/pdf/1908.08681.pdf
class Mish(Activation):

    def forward(self, input):
        return input * np.tanh(np.log(1 + np.exp(input)))
    
    def backward(self, grad):
        return grad * np.exp(grad) * 4 / (np.exp(2 * grad) + 2 * np.exp(grad) + 2) ** 2
    
# He et al. 2015 https://arxiv.org/pdf/1502.01852.pdf
class PReLU(Activation):

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, input):
        return np.where(input > 0, input, self.alpha * input)
    
    def backward(self, grad):
        return np.where(grad > 0, 1, self.alpha)
    
# Hendrycks et al. 2016 https://arxiv.org/pdf/1606.08415.pdf
class GELU(Activation):

    def forward(self, input):
        return 0.5 * input * (1 + np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * input ** 3)))
    
    def backward(self, grad):
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (grad + 0.044715 * grad ** 3))) * (1 + 0.044715 * grad ** 2 + 0.5 * grad) * (1 / np.cosh(np.sqrt(2 / np.pi) * (grad + 0.044715 * grad ** 3))) ** 2
    
class Softplus(Activation):

    def forward(self, input):
        return np.log(1 + np.exp(input))
    
    def backward(self, grad):
        return 1 / (1 + np.exp(-grad))
    
class Softsign(Activation):

    def forward(self, input):
        return input / (1 + np.abs(input))
    
    def backward(self, grad):
        return 1 / (1 + np.abs(grad)) ** 2
  
class BentIdentity(Activation):

    def forward(self, input):
        return 0.5 * (np.sqrt(input ** 2 + 1) - 1) + input
    
    def backward(self, grad):
        return grad / (2 * np.sqrt(grad ** 2 + 1)) + 1
    
class Sinusoid(Activation):

    def forward(self, input):
        return np.sin(input)
    
    def backward(self, grad):
        return np.cos(grad)
    
class Sinc(Activation):

    def forward(self, input):
        return np.where(input == 0, 1, np.sin(input) / input)
    
    def backward(self, grad):
        return np.where(grad == 0, 0, np.cos(grad) / grad - np.sin(grad) / grad ** 2)
    
class Gaussian(Activation):

    def forward(self, input):
        return np.exp(-input ** 2)
    
    def backward(self, grad):
        return -2 * grad * np.exp(-grad ** 2)