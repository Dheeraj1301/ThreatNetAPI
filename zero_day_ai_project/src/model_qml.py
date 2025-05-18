import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    qml.AngleEmbedding(x1, wires=[0, 1])
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.AngleEmbedding(x2, wires=[0, 1])
    return qml.probs(wires=[0, 1])
