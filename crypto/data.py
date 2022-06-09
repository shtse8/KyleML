from __future__ import annotations

from collections import namedtuple
from datetime import datetime
from typing import Sequence, Iterator

import numpy as np
from crypto.utils import Cache

Sample = namedtuple('Sample', ['feature', 'label'])


class Token:
    id: int
    symbol: str

    def __init__(self, id: int, symbol: str):
        self.id = id
        self.symbol = symbol

    def __eq__(self, other):
        return self.symbol == other.symbol


class DataFrame:
    def __init__(self,
                 open_time: int,
                 open_price: float = 0.,
                 high_price: float = 0.,
                 low_price: float = 0.,
                 close_price: float = 0.,
                 volume: float = 0.,
                 close_time: int = 0,
                 quote_asset_volume: float = 0.,
                 number_of_trades: int = 0):
        self.open_time = open_time
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.close_time = close_time
        self.quote_asset_volume = quote_asset_volume
        self.number_of_trades = number_of_trades

    def __repr__(self):
        return f"{type(self).__name__}({str(self)})"

    def __str__(self) -> str:
        return f"{self.open_time}, " \
               f"{self.open_price:.4}, " \
               f"{self.high_price:.4}, " \
               f"{self.low_price:.4}, " \
               f"{self.close_price:.4}, " \
               f"{self.volume:.4}, " \
               f"{self.close_time}, " \
               f"{self.quote_asset_volume:.4}, " \
               f"{self.number_of_trades}"


class DataFrames:
    def __init__(self, interval: int = 15 * 60):
        self.frame_dict = {}
        self.interval = interval
        self.start_time: int = 0
        self.end_time: int = 0

    def add(self, frame: DataFrame):
        # if self.start_time != 0 and (frame.open_time - self.start_time) % self.interval != 0:
        #     warnings.warn(f"the open time is not fit for this frame set. "
        #                   f"{frame.open_time} - {self.start_time} = "
        #                   f"{frame.open_time - self.start_time}", UserWarning)
        #     raise ValueError(f"the open time is not fit for this frame set. "
        #                      f"{frame.open_time} - {self.start_time} = "
        #                      f"{frame.open_time - self.start_time}")

        time = self.get_adjust_time(frame.open_time)
        self.frame_dict[time] = frame

        # update start time and end time
        if self.start_time == 0:
            self.start_time = time
            self.end_time = time
        else:
            if time < self.start_time:
                self.start_time = time
            if time > self.end_time:
                self.end_time = time

    def ensure_valid_timestamp(self, timestamp: int):
        if (timestamp - self.start_time) % self.interval != 0:
            raise Exception("Timestamp is not match for this data frames.")

    def __len__(self) -> int:
        if self.end_time < self.start_time:
            return 0
        return (self.end_time - self.start_time) // self.interval + 1

    def __contains__(self, item) -> bool:
        if isinstance(item, int):
            return 0 <= item < 100
        elif isinstance(item, datetime):
            return self.has_timestamp(int(item.timestamp()))
        else:
            raise TypeError("Invalid argument type.")

    def get_adjust_time(self, time: int):
        return (time // self.interval) * self.interval

    def get_offset(self, frame: DataFrame, offset):
        if offset == 0:
            return frame
        return self.from_timestamp(((frame.open_time // self.interval) + offset) * self.interval)

    def get_next(self, frame: DataFrame):
        return self.get_offset(frame, 1)

    def get_previous(self, frame: DataFrame):
        return self.get_offset(frame, -1)

    def __getitem__(self, key) -> DataFrame | DataFrames:
        # return multiple frames
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            new_data_frames = DataFrames()
            for i in range(*key.indices(len(self))):
                new_data_frames.add(self[i])
            return new_data_frames
        # return multiple frames
        elif isinstance(key, Sequence):
            new_data_frames = DataFrames()
            for i in key:
                new_data_frames.add(self[i])
            return new_data_frames
        # return single frame
        elif isinstance(key, int):
            # elif isinstance(key, int | float):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)

            open_time = self.start_time + key * self.interval
            return self.from_timestamp(open_time)
        # return single frame
        elif isinstance(key, datetime):
            return self.from_timestamp(int(key.timestamp()))  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self) -> Iterator[DataFrame]:
        yield from iter([self[i] for i in range(len(self))])

    def has_timestamp(self, timestamp: int):
        return timestamp in self.frame_dict

    def from_timestamp(self, timestamp: int) -> DataFrame:
        timestamp = self.get_adjust_time(timestamp)
        return self.frame_dict[timestamp] if timestamp in self.frame_dict else DataFrame(timestamp)


class Market:
    _data_frames: DataFrames = None

    def __init__(self, agent, token1: Token, token2: Token):
        self.token1 = token1
        self.token2 = token2
        self.agent = agent
        self.cache = Cache(f'data/market.{self.token1.symbol}{self.token2.symbol}.pickle')

    @property
    def data_frames(self, force_update: bool = False) -> DataFrames:
        if self._data_frames is None:
            self._data_frames = self.cache.load_or_update(lambda: self.agent.get_frames(self.token1, self.token2),
                                                          force_update=force_update)
        return self._data_frames


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
