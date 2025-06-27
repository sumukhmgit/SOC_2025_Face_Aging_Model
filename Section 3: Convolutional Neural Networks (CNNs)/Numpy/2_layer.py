import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 1: Create input
x = np.array([0.5, -1.2, 2.0])

# Step 2: Initialize hidden layer weights and bias
W1 = np.random.randn(4, 3)
b1 = np.random.randn(4)

# Step 3: Compute hidden layer output
# TODO: Write the formula for z1 and a1
z1 = np.matmul(W1, x) + b1
a1 = relu(z1)

# Step 4: Output layer (1 neuron)
W2 = np.random.randn(1, 4)
b2 = np.random.randn(1)

# Step 5: Final output
# TODO: Write z2 and output using sigmoid
z2 = np.matmul(W2, a1) + b2
output = sigmoid(z2)

print("Network output:", output)
