from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):

    def __init__(self, lr=0.01):
        self.lr = lr

    @abstractmethod
    def update(self, params, grads):
        pass

class SGD(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class Momentum(Optimizer):

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grad
            param -= self.lr * self.v[i]

# Duchi et al., 2011 https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
class AdaGrad(Optimizer):

    def __init__(self, lr=0.01, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.h[i] += grad ** 2
            param -= self.lr * grad / (np.sqrt(self.h[i]) + self.epsilon)

# Tieleman and Hinton, 2012 
class RMSprop(Optimizer):

    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def update(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * grad ** 2
            param -= self.lr * grad / (np.sqrt(self.s[i]) + self.epsilon)

# Zeiler, 2012 https://arxiv.org/abs/1212.5701
class AdaDelta(Optimizer):

    def __init__(self, lr=0.01, rho=0.95, epsilon=1e-6):
        super().__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.h = None
        self.s = None

    def update(self, params, grads):
        if self.h is None:
            self.h = [np.zeros_like(param) for param in params]
            self.s = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.h[i] = self.rho * self.h[i] + (1 - self.rho) * grad ** 2
            update = grad * np.sqrt(self.s[i] + self.epsilon) / np.sqrt(self.h[i] + self.epsilon)
            param -= update
            self.s[i] = self.rho * self.s[i] + (1 - self.rho) * update ** 2

# Kingma and Ba, 2015 https://arxiv.org/abs/1412.6980
class Adam(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            param -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)

class Nadam(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Liu et al., 2019 https://arxiv.org/abs/1908.03265
class RAdam(Optimizer):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.9, warmup=100):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.warmup = warmup
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        beta1_t = self.beta1 * (1 - self.decay ** self.t) / (1 - self.beta1 ** self.t)
        beta2_t = self.beta2 ** self.t

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[i] / (1 - beta1_t)
            v_hat = self.v[i] / (1 - beta2_t)
            rho = np.sqrt((2 - beta2_t) / (v_hat + self.epsilon))
            rho = np.minimum(rho, 4)
            rho = np.maximum(rho, 1e-8)
            param -= self.lr * rho * m_hat

# Zhang et al., 2019 https://arxiv.org/abs/1907.08610
class Lookahead(Optimizer):

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = []
        self.state = {}
        
    def add_param_group(self, param_group):
        self.param_groups.append(param_group)
        self.state[param_group] = {}
        
        for param in param_group:
            self.state[param_group][param] = param.clone()
        
    def update(self, params, grads):
        self.base_optimizer.update(params, grads)
        
        for group in self.param_groups:
            for param in group:
                param.data = self.alpha * param.data + (1 - self.alpha) * self.state[group][param].data
                