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

c = 0
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

    c += 1
    if c > 30:
        break

data_frame: DataFrame = DataFrame(klines).set_index('open_time', drop=False)
print(time, len(data_frame))

print(data_frame.dtypes)
interval = pd.Timedelta(minutes=15)
# .groupby(pd.Grouper(freq=interval), as_index=False)
result = data_frame.sort_index().resample(interval).agg(
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
    })[lambda x: (x.open_time >= x.index) & (x.close_time < x.index + interval)].assign(
    duration=lambda x: x.close_time - x.open_time)[lambda x: x.duration < pd.Timedelta(minutes=14)].filter(
    ['open_time', 'close_time'])

samples = pd.DataFrame({"features": list(data_frame.assign(open_time_year=lambda x: x.open_time.dt.year,
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
]).rolling(window=100))})

