import numpy as np

arr = np.array([4, 8, 15, 16, 23, 42])

# TODO: Calculate mean and std
mean = np.mean(arr)
std = np.std(arr)

# TODO: Normalize the array
normalized = (arr - mean)/std
print("Normalized array:", normalized)