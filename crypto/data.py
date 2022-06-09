from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy import datetime64

from crypto.utils import Cache


@dataclass
class Sample:
    feature: [float]
    label: int


@dataclass
class KLine:
    open_time: datetime64
    close_time: datetime64
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_asset_volume: float
    number_of_trades: int


# class KLines:
#     def __init__(self, data_frame: pd.DataFrame, interval: str = '15min'):
#         self.data_frame = data_frame.sort_index().groupby(pd.Grouper(freq=interval), as_index=False).agg(
#             {
#                 'open_time': 'first',
#                 'close_time': 'last',
#                 'open_price': 'first',
#                 'close_price': 'last',
#                 'high_price': 'max',
#                 'low_price': 'min',
#                 'volume': 'sum',
#                 'quote_asset_volume': 'sum',
#                 'number_of_trades': 'sum'
#             }).set_index('open_time', drop=False)[lambda x: (x.close_time - x.open_time) < pd.to_timedelta(interval)]
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __iter__(self):
#         yield from self.data_frame.iterrows()
#
#     def __getitem__(self, item):
#         return KLines(self.data_frame[item])

@dataclass
class Token:
    id: int
    symbol: str

    def __eq__(self, other):
        return self.symbol == other.symbol


class Market:
    _klines: KLines = None

    def __init__(self, agent, token1: Token, token2: Token):
        self.token1 = token1
        self.token2 = token2
        self.agent = agent
        self.cache = Cache(f'data/market.{self.token1.symbol}{self.token2.symbol}.pickle')

    @property
    def klines(self, force_update: bool = False) -> KLines:
        if self._klines is None:
            self._klines = self.cache.load_or_update(lambda: self.agent.get_frames(self.token1, self.token2),
                                                     force_update=force_update)
        return self._klines


class Samples:
    def __init__(self, *args):
        if len(args) == 1:
            samples, = args
            features = [x.feature for x in samples]
            labels = [x.label for x in samples]
        elif len(args) == 2:
            features, labels = args
        else:
            raise TypeError("Expected Sample Sequence or Features-Labels Sequence")
        self._features = np.asarray(features, np.float)
        self._labels = np.asarray(labels, np.long)
        if len(self._features) != len(self._labels):
            raise Exception(f"features and labels must have same number of elements. "
                            f"num_features={len(self.features)}, "
                            f"num_labels={len(self._labels)}")

    @property
    def features(self):
        return self._features

    @property
    def labels(self) -> Sequence[int]:
        return self._labels

    def __len__(self):
        return len(self._features)

    def __getitem__(self, item) -> Sample | Samples:
        if isinstance(item, int):
            return Sample(self._features[item], self._labels[item])
        else:
            return Samples(self._features[item], self._labels[item])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
