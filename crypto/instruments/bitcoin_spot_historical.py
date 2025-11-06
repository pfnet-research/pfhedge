from .bitcoin_perpetual_historical import BitcoinPerpetualHistorical


class BitcoinSpotHistorical(BitcoinPerpetualHistorical):

    @property
    def has_funding(self) -> bool:
        return False

    @property
    def max_leverage(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"volatility_window={self.volatility_window}",
        ]
        if self.constant_volatility is not None:
            params.append(f"constant_volatility={self.constant_volatility}")
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinSpotHistorical({', '.join(params)})"
