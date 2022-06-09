import math
from abc import ABC, abstractmethod
from datetime import datetime

from binance.spot import Spot

from crypto.data import Token, DataFrame, DataFrames


class Client(ABC):
    name: str = ""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_frames(self, token1: Token, token2: Token) -> DataFrames:
        raise NotImplementedError


class BinanceMarketAgent(Client):
    _client: Spot = Spot()

    def __init__(self):
        super().__init__("Binance")

    def get_frames(self, token1: Token, token2: Token) -> DataFrames:
        data_frames = DataFrames()
        start_time_datetime = datetime.strptime("01/01/2010", '%d/%m/%Y')
        market_id = token1.symbol + token2.symbol
        print("Fetching ", market_id, " market")
        time = math.floor(start_time_datetime.timestamp() * 1000)
        while datetime.now().timestamp() > time / 1000:
            lines = self._client.klines(market_id, "15m", startTime=time, limit=1000)
            for line in lines:
                open_time = line[0] // 1000
                open_price = float(line[1])
                high_price = float(line[2])
                low_price = float(line[3])
                close_price = float(line[4])
                volume = float(line[5])
                close_time = line[6] // 1000
                quote_asset_volume = float(line[7])
                number_of_trades = int(line[8])
                frame = DataFrame(open_time,
                                  open_price,
                                  high_price,
                                  low_price,
                                  close_price,
                                  volume,
                                  close_time,
                                  quote_asset_volume,
                                  number_of_trades)
                data_frames.add(frame)
            print(time, len(data_frames))

            # last close time + 1s
            time = lines[-1][6] + 1

        print("Returning")
        return data_frames
