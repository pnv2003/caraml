import numpy as np
from model import Model
from model.neuralnet.layer import Layer
from model.neuralnet.loss import LossFunction
from model.neuralnet.optimizer import Optimizer

class NeuralNetwork(Model):

    def __init__(self) -> None:
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def compile(self, loss_function: LossFunction, optimizer: Optimizer) -> None:
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, X: np.ndarray) -> np.ndarray:

        for layer in self.layers:
            X = layer.forward(X)

        return X
    
    def backward(self, grad: np.ndarray) -> None:

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:

        preds = self.forward(X)
        loss = self.loss_function.forward(preds, y)
        grad = self.loss_function.backward(preds, y)

        self.backward(grad)
        for layer in self.layers:
            self.optimizer.update(layer.weights, layer.gradients)

        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:

        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.train_step(X_batch, y_batch)

            loss = self.evaluate(X, y)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        print("Training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:

        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:

        preds = self.predict(X)
        return self.loss_function.forward(preds, y)

    def summary(self) -> None:

        print("Model Summary")
        print("--------------")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer}")
        print(f"Loss Function: {self.loss_function}")
        print(f"Optimizer: {self.optimizer}")
        print("--------------")

    def save(self, path: str) -> None:

        model = {
            "layers": self.layers,
            "loss_function": self.loss_function,
            "optimizer": self.optimizer
        }

        np.save(path, model, allow_pickle=True)

    @staticmethod
    def load(path: str) -> 'NeuralNetwork':

        model = np.load(path, allow_pickle=True).item()

        nn = NeuralNetwork()
        nn.layers = model["layers"]
        nn.loss_function = model["loss_function"]
        nn.optimizer = model["optimizer"]

        return nn
    
    def __repr__(self) -> str:
        return f"NeuralNetwork({self.layers}, {self.loss_function}, {self.optimizer})"

                                                                     
                                                            

