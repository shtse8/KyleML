from __future__ import annotations
import numpy as np
from typing import List, Callable, TypeVar, Generic, Tuple, Any
import torch

T = TypeVar('T')
S = TypeVar('S')

class KyleList(Generic[T]):
    def __init__(self, iter: List[T]) -> None:
        self.array = np.array(iter)

    def select(self, property: Callable[[T], S]) -> KyleList[S]:
        return KyleList([property(x) for x in self.array])

    def toArray(self) -> np.array:
        return self.array

    def toTensor(self, dtype: torch.dtype, device: torch.device) -> torch.tensor:
        return torch.tensor(self.array, dtype=dtype, device=device).detach()

    def get(self, fromPos: int, num: int) -> KyleList[T]:
        return KyleList(self.array[fromPos:fromPos+num])

    def sum(self):
        return self.array.sum()

    def mean(self, axis=None):
        return self.array.mean(axis=axis)

    def var(self, axis=None):
        return self.array.var(axis=axis)

    def std(self) -> float:
        return self.array.std()

    def size(self) -> int:
        return len(self.array)

    def __sub__(self, other):
        if isinstance(other, KyleList):
            other = other.array
        return KyleList(self.array - other)

    def __truediv__(self, other):
        if isinstance(other, KyleList):
            other = other.array
        return KyleList(self.array / other)

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, i: int) -> T:
        return self.array[i]

    def __iter__(self) -> Iterator:
        return Iterator(self)

    def __str__(self):
        return f"KyleList({str(self.array)})"


class Iterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.current = 0

    def __iter__(self) -> Iterator:
        return Iterator(self.iterable)

    def __next__(self):
        if self.current < len(self.iterable):
            result = self.iterable[self.current]
            self.current += 1
            return result
        self.current = 0
        raise StopIteration

