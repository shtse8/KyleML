from abc import abstractmethod
from datetime import datetime
from typing import Iterator

import pandas as pd

from crypto.data import Sample
from crypto.utils import Cache, PerformanceTimer


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
        frame_dict = {}
        i = 0
        timer = PerformanceTimer()
        timer.start()
        for index, frame in self._data_frames.iterrows():
            try:
                frame_dict[index] = Sample(self._get_seq_feature(frame, 100), self._get_label(frame))
                i += 1
                if i % 100 == 0:
                    timer.stop()
                    print(timer.elapsed())
                    timer = PerformanceTimer()
                    timer.start()
            except IndexError:
                pass
        return frame_dict

    def get_feature(self, frame: pd.Series) -> [float]:
        return [
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
        ]

    def _get_cached_feature(self, frame: pd.Series) -> [float]:
        if frame.open_time in self._frame_features:
            return self._frame_features[frame.open_time]
        self._frame_features[frame.open_time] = self.get_feature(frame)
        return self._frame_features[frame.open_time]

    def _get_seq_feature(self, frame: pd.Series, seq_len: int = 1) -> [float]:
        # return [self._get_cached_frame_feature(x) for i, x in
        #         self._data_frames[lambda x: x.open_time <= frame.open_time].head(
        #             100).iterrows()]
        seq = self._data_frames.loc[:frame.name].tail(seq_len)
        if len(seq) != seq_len:
            raise IndexError
        return [self._get_cached_feature(x) for i, x in seq.iterrows()]

    def _get_label(self, frame: pd.Series) -> int:
        if frame.close_price == 0:
            return 0
        # next_frame = self._data_frames[lambda x: x.open_time <= frame.open_time].iloc[0]
        next_frame = self._data_frames.loc[frame.name:].iloc[1:2].iloc[0]
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
        return [self.sample_cache[self._data_frames.get_adjust_time(x.open_time)] for x in frames]

    def get_labels(self):
        return [self._get_label(x) for x in self._data_frames]
