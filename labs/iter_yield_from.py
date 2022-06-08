import timeit
import numpy as np


class Foo:
    def __init__(self):
        self.items = np.random.randn(10000, 10)

    def __iter__(self):
        yield from iter([self.items[x] for x in range(len(self.items))])


class Foo2:
    def __init__(self):
        self.items = np.random.randn(10000, 10)

    def __iter__(self):
        for x in range(len(self.items)):
            yield self.items[x]


class Foo3:
    def __init__(self):
        self.items = np.random.randn(10000, 10)

    def __iter__(self):
        yield from map(self.items.__getitem__, range(len(self.items)))


class Foo4:
    def __init__(self):
        self.items = np.random.randn(10000, 10)

    def __iter__(self):
        yield from map(lambda x: self.items[x], range(len(self.items)))


def test(x):
    return next(iter(x))


print(timeit.timeit(lambda: test(Foo()), number=1000))
print(timeit.timeit(lambda: test(Foo2()), number=1000))
print(timeit.timeit(lambda: test(Foo3()), number=1000))
print(timeit.timeit(lambda: test(Foo4()), number=1000))

