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
        self._data_frames = data_frames.assign(open_time_year=lambda x: x.open_time.dt.year,
                   open_time_month=lambda x: x.open_time.dt.month,
                   open_time_day=lambda x: x.open_time.dt.month,
                   open_time_minute=lambda x: x.open_time.dt.month,
                   open_time_weekday=lambda x: x.open_time.dt.weekday,
                   close_time_year=lambda x: x.close_time.dt.year,
                   close_time_month=lambda x: x.close_time.dt.month,
                   close_time_day=lambda x: x.close_time.dt.month,
                   close_time_minute=lambda x: x.close_time.dt.month,
                   close_time_weekday=lambda x: x.close_time.dt.weekday).filter([
            'open_time_year',
            'open_time_month',
            'open_time_day',
            'open_time_minute',
            'open_time_weekday',
            'close_time_year',
            'close_time_month',
            'close_time_day',
            'close_time_minute',
            'close_time_weekday'
            'open_price',
            'high_price',
            'low_price',
            'close_price',
            'volume'
        ])

        self._frame_features = {}
        sample_cache = Cache(f"data/converter_cache.dat")
        self.sample_cache = sample_cache.load_or_update(self._process)

    def _process(self):
        print("Processing samples...")
        frame_dict = {}
        # i = 0
        # timer = PerformanceTimer()
        # timer.start()
        for frames in filter(lambda x: len(x) == 100, self._data_frames.rolling(window=100)):
            try:
                frame = frames.iloc[-1]
                feature = frames.to_numpy(dtype=float)
                frame_dict[frame.name] = Sample(feature, self._get_label(frame))
                # i += 1
                # if i % 100 == 0:
                #     timer.stop()
                #     print(timer.elapsed())
                #     timer = PerformanceTimer()
                #     timer.start()
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
        # return seq.assign(open_time_year=lambda x: x.open_time.dt.year,
        #            open_time_month=lambda x: x.open_time.dt.month,
        #            open_time_day=lambda x: x.open_time.dt.month,
        #            open_time_minute=lambda x: x.open_time.dt.month,
        #            open_time_weekday=lambda x: x.open_time.dt.weekday,
        #            close_time_year=lambda x: x.close_time.dt.year,
        #            close_time_month=lambda x: x.close_time.dt.month,
        #            close_time_day=lambda x: x.close_time.dt.month,
        #            close_time_minute=lambda x: x.close_time.dt.month,
        #            close_time_weekday=lambda x: x.close_time.dt.weekday).filter([
        #     'open_time_year',
        #     'open_time_month',
        #     'open_time_day',
        #     'open_time_minute',
        #     'open_time_weekday',
        #     'close_time_year',
        #     'close_time_month',
        #     'close_time_day',
        #     'close_time_minute',
        #     'close_time_weekday'
        #     'open_price',
        #     'high_price',
        #     'low_price',
        #     'close_price',
        #     'volume'
        # ]).to_numpy(dtype=float)

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
