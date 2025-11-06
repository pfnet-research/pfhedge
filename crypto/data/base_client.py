from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class MarketDataClient(ABC):

    @abstractmethod
    def get_instruments(
        self, currency: str = "BTC", kind: str = "option", **kwargs
    ) -> List[Dict]:
        pass

    @abstractmethod
    def get_historical_trades(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        count: int = 1000,
    ) -> List[Dict]:
        pass

    @abstractmethod
    def get_ticker(self, instrument_name: str, timestamp: Optional[int] = None) -> Dict:
        pass

    @abstractmethod
    def get_funding_rate_history(
        self,
        instrument_name: str = "BTC-PERPETUAL",
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> List[Dict]:
        pass

    @abstractmethod
    def get_recent_trades(self, instrument_name: str, count: int = 10) -> List[Dict]:
        pass

    # Optional helper methods that subclasses can override
    def get_order_book(self, instrument_name: str, depth: int = 5) -> Dict:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support order book data"
        )

    def get_historical_volatility(self, currency: str = "BTC") -> List[Dict]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support historical volatility data"
        )
