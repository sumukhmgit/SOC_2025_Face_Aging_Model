import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 1: Create 3 sample inputs (3 features each)
X = np.random.randn(3, 3)

# Step 2: Shared hidden layer weights and biases
W = np.random.randn(3, 3)
b = np.random.randn(3,)

# TODO: Complete forward pass logic using dot product
def forward(x):
    z = np.matmul(W, x) + b
    return relu(z)

# Step 3: Run the forward pass on each input
for i, x in enumerate(X):
    print(f"\nInput {i+1}:", x)
    print("Hidden Output:", forward(x))