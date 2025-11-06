import os
import logging
from typing import Optional

from .base_client import MarketDataClient
from .deribit_client import DeribitClient

logger = logging.getLogger(__name__)


def create_client(
    data_source: str = "deribit",
    testnet: bool = True,
    tardis_api_key: Optional[str] = None,
    **kwargs,
) -> MarketDataClient:
    if data_source == "tardis":
        # Get API key from parameter or environment
        api_key = tardis_api_key or os.getenv("TARDIS_API_KEY")

        if not api_key:
            logger.warning(
                "No Tardis API key provided. Using free tier (first day of each month only). "
                "For full access, provide API key via tardis_api_key parameter or TARDIS_API_KEY env var."
            )

        try:
            from .tardis_client import TardisClient

            access_mode = (
                "full historical access"
                if api_key
                else "free tier (first day of month)"
            )
            logger.info(f"Using Tardis.dev as data source ({access_mode})")
            return TardisClient(api_key=api_key, testnet=testnet, **kwargs)
        except ImportError as e:
            raise ImportError(
                "tardis-client is required for Tardis data source. "
                "Install with: pip install tardis-client"
            ) from e

    elif data_source == "deribit":
        network = "testnet" if testnet else "mainnet"
        logger.info(f"Using Deribit API as data source ({network})")
        return DeribitClient(testnet=testnet, **kwargs)

    else:
        raise ValueError(
            f"Unknown data source: {data_source}. " f"Must be 'deribit' or 'tardis'"
        )


def add_client_args(parser):
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["deribit", "tardis"],
        default="deribit",
        help="Data source: 'deribit' (recent, ~24h history) or 'tardis' (full history since 2019-03-30)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet instead of mainnet (Deribit only; Tardis uses mainnet historical data)",
    )
    parser.add_argument(
        "--tardis-api-key",
        type=str,
        default=os.getenv("TARDIS_API_KEY"),
        help="Tardis.dev API key (or set TARDIS_API_KEY env var)",
    )

    return parser
