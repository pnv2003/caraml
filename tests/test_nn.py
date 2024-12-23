import numpy as np
from model.neuralnet import NeuralNetwork
from model.neuralnet.activation import Sigmoid
from model.neuralnet.layer import Dense
from model.neuralnet.loss import MSE
from model.neuralnet.optimizer import SGD

def test():
    nn = NeuralNetwork()
    nn.add(Dense(1, 1, activation=Sigmoid()))
    nn.add(Dense(1, 1, activation=Sigmoid()))
    nn.compile(loss_function=MSE(), optimizer=SGD(lr=0.1))

    # custom weights initialization
    nn.layers[0].W = np.array([[0.5]])
    nn.layers[0].b = np.array([[0.3]])
    nn.layers[1].W = np.array([[0.7]])
    nn.layers[1].b = np.array([[0.4]])

    X = np.array([2])
    y = np.array([1])

    loss = nn.train_step(X, y)

    print("Loss:", loss)

    print('Hidden Layer Weights and Biases:') 
    print(nn.layers[0].W)
    print(nn.layers[0].b)

    print('Output Layer Weights and Biases:')
    print(nn.layers[1].W)
    print(nn.layers[1].b)