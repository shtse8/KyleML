from abc import abstractmethod
from time import perf_counter
from typing import Iterator, Hashable

import numpy as np
import pandas as pd

from crypto.data import Sample
from crypto.utils import Cache


class Converter:
    @abstractmethod
    def get_samples(self, frames: pd.DataFrame) -> Iterator[Sample]:
        raise NotImplementedError


class DataFrameSampleConverter(Converter):
    def __init__(self, data_frames: pd.DataFrame):
        self._data_frames = data_frames
        self._frame_features = {}
        sample_cache = Cache(f"data/converter_cache.dat")
        self.sample_cache = sample_cache.load_or_update(self._process)

    def _process(self):
        print("Processing samples...")
        samples = {}
        processed = 0
        start = perf_counter()
        for index, frame in self._data_frames.iterrows():
            try:
                samples[index] = Sample(self._get_seq_feature(frame, 100), self._get_label(frame))
            except IndexError:
                pass
            finally:
                processed += 1
                progress = processed / len(self._data_frames)
                time_to_end = (perf_counter() - start) / progress * (1 - progress)
                print(
                    f"\r[{processed / len(self._data_frames):.2%}] {processed}/{len(self._data_frames)} {time_to_end:.0f}s",
                    end="")
        print()
        return samples

    def get_feature(self, frame: pd.Series) -> np.array:
        return np.array([
            frame.open_time.year,
            frame.open_time.month,
            frame.open_time.day,
            frame.open_time.hour,
            frame.open_time.minute,
            frame.open_time.weekday(),
            frame.open_price,
            frame.high_price,
            frame.low_price,
            frame.close_price,
            frame.volume,
            frame.close_time.year,
            frame.close_time.month,
            frame.close_time.day,
            frame.close_time.hour,
            frame.close_time.minute,
            frame.close_time.weekday(),
            frame.quote_asset_volume,
            frame.number_of_trades
        ], dtype=float)

    def _get_cached_feature(self, index: Hashable) -> [float]:
        if index in self._frame_features:
            return self._frame_features[index]
        frame = self._data_frames.loc[index]
        self._frame_features[index] = self.get_feature(frame)
        return self._frame_features[index]

    def _get_seq_feature(self, frame: pd.Series, seq_len: int = 1) -> [float]:
        # return [self._get_cached_frame_feature(x) for i, x in
        #         self._data_frames[lambda x: x.open_time <= frame.open_time].head(
        #             100).iterrows()]
        end_index = self._data_frames.index.get_loc(frame.name) + 1
        start_index = end_index - seq_len
        if start_index < 0:
            raise IndexError
        return np.array([self._get_cached_feature(i) for i in self._data_frames.index[start_index:end_index]])

    def _get_label(self, frame: pd.Series) -> int:
        if frame.close_price == 0:
            return 0
        # next_frame = self._data_frames[lambda x: x.open_time <= frame.open_time].iloc[0]

        index = self._data_frames.index.get_loc(frame.name)
        next_frame_index = self._data_frames.index[index + 1]
        next_frame = self._data_frames.loc[next_frame_index]
        change_rate = (next_frame.close_price - frame.close_price) / frame.close_price
        if change_rate <= -0.003:
            return 2
        elif change_rate >= 0.003:
            return 1
        return 0

    def get_sample(self, frame: pd.Series):
        time = self._data_frames.get_adjust_time(frame.open_time)
        if time not in self.sample_cache:
            raise IndexError(f"couldn't find {time}")
        return self.sample_cache[time]

    def get_samples(self, frames: pd.DataFrame = None):
        if frames is None:
            frames = self._data_frames
        return [self.sample_cache[i] for i in filter(lambda x: x in self.sample_cache, frames.index)]

    def get_labels(self):
        return [self._get_label(x) for x in self._data_frames]
