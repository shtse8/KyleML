from abc import abstractmethod
from datetime import datetime
from typing import Iterator

from crypto.data import DataFrames, Sample, DataFrame
from crypto.utils import Cache


class Converter:
    @abstractmethod
    def get_samples(self, frames: DataFrames) -> Iterator[Sample]:
        raise NotImplementedError


class DataFrameSampleConverter(Converter):
    def __init__(self, data_frames: DataFrames):
        self._data_frames = data_frames
        self._frame_features = {}
        sample_cache = Cache(f"data/converter_cache.dat")
        self.sample_cache = sample_cache.load_or_update(self._process)

    def _process(self):
        print("Processing samples...")
        frame_dict = {}
        for frame in self._data_frames:
            time = self._data_frames.get_adjust_time(frame.open_time)
            frame_dict[time] = Sample(self._get_feature(frame), self._get_label(frame))
        return frame_dict

    def get_frame_feature(self, frame: DataFrame) -> [float]:
        data_open_time = datetime.fromtimestamp(frame.open_time)
        data_close_time = datetime.fromtimestamp(frame.close_time)
        return [
            data_open_time.year,
            data_open_time.month,
            data_open_time.day,
            data_open_time.hour,
            data_open_time.minute,
            data_open_time.weekday(),
            frame.open_price,
            frame.high_price,
            frame.low_price,
            frame.close_price,
            frame.volume,
            data_close_time.year,
            data_close_time.month,
            data_close_time.day,
            data_close_time.hour,
            data_close_time.minute,
            data_close_time.weekday(),
            frame.quote_asset_volume,
            frame.number_of_trades
        ]

    def _get_cached_frame_feature(self, frame: DataFrame) -> [float]:
        if frame.open_time in self._frame_features:
            return self._frame_features[frame.open_time]
        self._frame_features[frame.open_time] = self.get_frame_feature(frame)
        return self._frame_features[frame.open_time]

    def _get_feature(self, frame: DataFrame) -> [float]:
        return [self._get_cached_frame_feature(self._data_frames.get_offset(frame, offset=-i)) for i in
                reversed(range(100))]

    def _get_label(self, frame: DataFrame) -> int:
        if frame.close_price == 0:
            return 0
        next_frame = self._data_frames.get_next(frame)
        change_rate = (next_frame.close_price - frame.close_price) / frame.close_price
        if change_rate <= -0.003:
            return 2
        elif change_rate >= 0.003:
            return 1
        return 0

    def get_sample(self, frame: DataFrame):
        time = self._data_frames.get_adjust_time(frame.open_time)
        if time not in self.sample_cache:
            raise IndexError(f"couldn't find {time}")
        return self.sample_cache[time]

    def get_samples(self, frames: DataFrames = None):
        if frames is None:
            frames = self._data_frames
        return [self.sample_cache[self._data_frames.get_adjust_time(x.open_time)] for x in frames]

    def get_labels(self):
        return [self._get_label(x) for x in self._data_frames]
