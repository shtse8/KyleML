import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6]])
i = np.array([[1], [0]])
# print(a[i])
print(np.take_along_axis(a, i, 1))