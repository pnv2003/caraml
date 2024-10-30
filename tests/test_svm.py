import numpy as np
from models.svm import SupportVectorMachine

classA = np.array([
    [-5, 5],
    [0, 1],
    [2, 0]
])

classB = np.array([
    [-1, -1],
    [2, -3]
])

data = np.concatenate((classA, classB))
labels = np.concatenate((
    np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])
))

model = SupportVectorMachine()
model.fit(data, labels)

print("Support vectors:\n", model.sv)
print("Alpha:", model.alpha)
print("w: ", model.w)
print("b: ", model.b)
