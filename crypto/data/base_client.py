"""
Abstract base class for market data clients.

This module defines the common interface that all market data providers
(Deribit, Tardis, etc.) must implement to ensure interoperability.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class MarketDataClient(ABC):
    """Abstract base class for cryptocurrency market data clients.

    All market data providers must implement this interface to ensure
    they can be used interchangeably across the codebase.

    The interface is designed to match Deribit's API structure, as it's
    the primary data source, but can be adapted for other providers.
    """

    @abstractmethod
    def get_instruments(
        self, currency: str = "BTC", kind: str = "option", **kwargs
    ) -> List[Dict]:
        """Get available instruments.

        Args:
            currency: Currency (BTC, ETH, etc.)
            kind: Instrument kind (option, future, spot)
            **kwargs: Provider-specific parameters

        Returns:
            List of instrument data dictionaries with fields:
            - instrument_name: str
            - strike: float (for options)
            - expiration_timestamp: int (milliseconds)
            - option_type: str ("call" or "put" for options)
            - kind: str
            - Other provider-specific fields
        """
        pass

    @abstractmethod
    def get_historical_trades(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        count: int = 1000,
    ) -> List[Dict]:
        """Get historical trades for an instrument.

        Args:
            instrument_name: Name of instrument (e.g., "BTC-PERPETUAL")
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            count: Maximum number of trades to fetch

        Returns:
            List of trade data dictionaries with fields:
            - timestamp: int (milliseconds)
            - price: float
            - amount: float
            - direction: str ("buy" or "sell")
            - trade_id: str
            - Other provider-specific fields
        """
        pass

    @abstractmethod
    def get_ticker(self, instrument_name: str, timestamp: Optional[int] = None) -> Dict:
        """Get ticker data for an instrument.

        Args:
            instrument_name: Name of instrument
            timestamp: Optional timestamp in milliseconds (for historical data)
                      If None, returns current/latest ticker

        Returns:
            Ticker data dictionary with fields:
            - instrument_name: str
            - last_price: float
            - best_bid_price: float
            - best_ask_price: float
            - mark_price: float
            - index_price: float
            - timestamp: int (milliseconds)
            - Other provider-specific fields
        """
        pass

    @abstractmethod
    def get_funding_rate_history(
        self,
        instrument_name: str = "BTC-PERPETUAL",
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> List[Dict]:
        """Get funding rate history for perpetual contract.

        Args:
            instrument_name: Name of perpetual instrument
            start_timestamp: Start time in milliseconds (optional)
            end_timestamp: End time in milliseconds (optional)

        Returns:
            List of funding rate data dictionaries with fields:
            - timestamp: int (milliseconds)
            - instrument_name: str
            - interest_8h: float (or similar rate field)
            - Other provider-specific fields
        """
        pass

    @abstractmethod
    def get_recent_trades(self, instrument_name: str, count: int = 10) -> List[Dict]:
        """Get recent trades for an instrument.

        Args:
            instrument_name: Name of instrument
            count: Number of recent trades to fetch

        Returns:
            List of trade data (same format as get_historical_trades)
        """
        pass

    # Optional helper methods that subclasses can override
    def get_order_book(self, instrument_name: str, depth: int = 5) -> Dict:
        """Get order book for an instrument.

        This is optional and may not be supported by all providers.

        Args:
            instrument_name: Name of instrument
            depth: Order book depth

        Returns:
            Order book data dictionary

        Raises:
            NotImplementedError: If provider doesn't support order book data
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support order book data"
        )

    def get_historical_volatility(self, currency: str = "BTC") -> List[Dict]:
        """Get historical volatility data.

        This is optional and may not be supported by all providers.

        Args:
            currency: Currency to get volatility for

        Returns:
            List of volatility data points

        Raises:
            NotImplementedError: If provider doesn't support volatility data
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support historical volatility data"
        )
