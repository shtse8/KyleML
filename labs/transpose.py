import numpy as np
from collections import namedtuple
import timeit

Sample = namedtuple('Sample', ['feature', 'label'])


def generate():
    # return [Sample(*x) for x in np.random.rand(10, 2) * 10]
    return [tuple(x) for x in np.random.rand(10, 2) * 10]

def test1():
    samples = generate()
    samples = np.asarray(samples, dtype=[('feature', float), ('label', np.int64)])
    return samples["feature"], samples["label"]


def test2():
    samples = generate()
    samples = np.asarray(samples)
    features, labels = samples.transpose()
    return features, np.asarray(labels, dtype=np.int64)


def test3():
    samples = generate()
    samples = np.asarray(samples)
    return samples[:, 0], np.asarray(samples[:, 1], dtype=np.int64)


def test4():
    samples = generate()
    samples = np.asarray(samples)
    return np.asarray([x[0] for x in samples], dtype=float), \
           np.asarray([x[1] for x in samples], dtype=np.int64)


def test5():
    samples = generate()
    return np.asarray([x[0] for x in samples], dtype=float), \
           np.asarray([x[1] for x in samples], dtype=np.int64)


print(test1())
print(test2())
print(test3())
print(test4())
print(test5())
print(timeit.timeit(test1, number=10000))
print(timeit.timeit(test2, number=10000))
print(timeit.timeit(test3, number=10000))
print(timeit.timeit(test4, number=10000))
print(timeit.timeit(test5, number=10000))
