import math
from datetime import datetime
from binance.spot import Spot
import pandas as pd
from dataclasses import dataclass, make_dataclass

from numpy import datetime64
from pandas import DataFrame


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
#
# KLine = make_dataclass("KLine", [
#     ("open_time", datetime64),
#     ("open_price", float),
#     ("high_price", float),
#     ("low_price", float),
#     ("close_price", float),
#     ("volume", float),
#     ("close_time", datetime64),
#     ("quote_asset_volume", float),
#     ("number_of_trades", int),
# ])
#

client = Spot()
start_time_datetime = datetime.strptime("01/01/2010", '%d/%m/%Y')
market_id = "BTCUSDT"
print("Fetching ", market_id, " market")
time = int(start_time_datetime.timestamp() * 1000)

klines = []
while datetime.now().timestamp() > time / 1000:
    lines = client.klines(market_id, "15m", startTime=time, limit=1000)
    for line in lines:
        open_time = pd.to_datetime(line[0], unit='ms')
        close_time = pd.to_datetime(line[6], unit='ms')
        open_price = float(line[1])
        high_price = float(line[2])
        low_price = float(line[3])
        close_price = float(line[4])
        volume = float(line[5])
        quote_asset_volume = float(line[7])
        number_of_trades = int(line[8])
        kline = KLine(open_time,
                      close_time,
                      open_price,
                      high_price,
                      low_price,
                      close_price,
                      volume,
                      quote_asset_volume,
                      number_of_trades)
        klines.append(kline)
    # last close time + 1s
    time = lines[-1][6] + 1
    break

data_frame: DataFrame = DataFrame(klines).set_index('open_time', drop=False)
print(time, len(data_frame))

print(data_frame.dtypes)
interval = '15min'
result = data_frame.sort_index().groupby(pd.Grouper(freq=interval), as_index=False).agg(
    {
        'open_time': 'first',
        'close_time': 'last',
        'open_price': 'first',
        'close_price': 'last',
        'high_price': 'max',
        'low_price': 'min',
        'volume': 'sum',
        'quote_asset_volume': 'sum',
        'number_of_trades': 'sum'
    }).set_index('open_time', drop=False)[lambda x: (x.close_time - x.open_time) < pd.to_timedelta(interval)]
