import math
import pickle
import sys
import torch


#   數組為正態分佈
#   計算出最後每個數值的標準差
def normalize(value):
    value = (value - value.mean()) / (value.std() + 1e-8)
    return value


def humanize(value):
    if value in [None, math.nan, math.inf, -math.inf]:
        return str(value)

    sci = ["K", "M", "B", "T"]
    for i in reversed(range(len(sci))):
        base = 1000 ** (i + 1)
        if value >= base:
            return f"{value / base:.2f}{sci[i]}"
    return f"{value:.0f}" if isinstance(value, int) else f"{value:.2f}"


def humanize_time(value):
    if value in [None, math.nan, math.inf, -math.inf]:
        return str(value)

    time_str = ""
    if value > 86400 * 7:
        time_str += f"{value // (86400 * 7):.0f}w"
    if value > 86400:
        time_str += f"{(value % (86400 * 7)) // 86400:>1.0f}d"
    if value > 3600:
        time_str += f"{(value % 86400) // 3600:>2.0f}h"
    if value > 60:
        time_str += f"{(value % 3600) // 60:>2.0f}m"
    time_str += f"{value % 60 // 1:>2.0f}s"
    return time_str.strip()


def get_size(obj):
    return len(pickle.dumps(obj))
