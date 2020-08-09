# import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout
# import tensorflow as tf

# model = Sequential()
# # model.add(Dense(8, input_shape=(16,)))
# # Afterwards, we do automatic shape inference:  
# # model.add(Dense(4))

# model.add(tf.keras.Input(shape=(16,)))
# model.add(Dense(8))

# print(len(model.weights))
# import numpy as np
# from keras.utils import to_categorical
# actions = [1,2,0,2,0,1,2]
# output = np.logical_not(np.array([to_categorical(action, num_classes=3) for action in actions])).astype(int)
# print(output)

# import collections
# import random
# memory = collections.deque(maxlen=10000)

# while True:
    # memory.append(random.random())
    # print(len(memory), memory[0])
    
import numpy as np
import collections
# a = collections.deque(maxlen=4)
# a.extend([1] * 4)
# print(a)
# print(np.zeros((12,)+(3,4)))

# a = np.array([[1, 2, 7], [3, 4, 8], [5, 6, 9]])

# print(a[a[:,0] == 1])

a = (22, 22) + (1,)
print(a)