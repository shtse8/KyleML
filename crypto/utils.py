import pickle
import signal
import sys
from os.path import exists
from time import perf_counter
from typing import Callable, IO, NoReturn


class PerformanceTimer:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = perf_counter()
        return self

    def stop(self):
        self.stop_time = perf_counter()
        return self

    def elapsed(self):
        return self.stop_time - self.start_time

    def __repr__(self):
        return f"{type(self).__name__}({str(self)})"

    def __str__(self):
        return f"{self.elapsed():.4}s"

    def __format__(self, format_spec):
        return format(self.elapsed(), format_spec)


class Cache:
    def __init__(self,
                 file_path: str,
                 load_fn: Callable[[IO], any] = pickle.load,
                 save_fn: Callable[[any, IO], None] = pickle.dump):
        self.file_path = file_path
        self.load_fn = load_fn
        self.save_fn = save_fn

    def load(self) -> any:
        if not exists(self.file_path):
            raise FileNotFoundError

        with open(self.file_path, 'rb') as file:
            data = self.load_fn(file)
            if data is None:
                raise ValueError
            return data

    def update(self, data: any, save_fn: Callable[[any, IO], None] = pickle.dump) -> None:
        with open(self.file_path, 'wb') as file:
            save_fn(data, file)

    def load_or_update(self,
                       on_failed: Callable[[], any],
                       force_update: bool = False) -> any:
        data = None
        try:
            if not force_update:
                data = self.load()
        except (FileNotFoundError, ValueError, EOFError) as _:
            force_update = True

        if force_update:
            data = on_failed()
            self.update(data)

        return data


def _signal_handler(sig, frame) -> NoReturn:
    print()
    print()
    print()
    print('You pressed Ctrl+C!')
    sys.exit(0)


def register_signals():
    signal.signal(signal.SIGINT, _signal_handler)

