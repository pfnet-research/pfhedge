"""
Factory function for creating market data clients.

This module provides a unified way to create market data clients across
different scripts in the crypto module.
"""

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
    """Create market data client based on data source.

    Args:
        data_source: "deribit" or "tardis"
        testnet: Use testnet (for Deribit only, Tardis uses mainnet historical data)
        tardis_api_key: Tardis.dev API key (optional; None uses free tier - first day of month only)
        **kwargs: Additional client-specific parameters

    Returns:
        MarketDataClient instance (DeribitClient or TardisClient)

    Raises:
        ValueError: If data_source is invalid
        ImportError: If required package is not installed (tardis-client for Tardis)

    Notes:
        - Tardis without API key: Free tier access (first day of each month only)
        - Tardis with API key: Full historical access since 2019-03-30
        - API key can be provided via parameter or TARDIS_API_KEY environment variable

    Examples:
        >>> # Create Deribit client (testnet)
        >>> client = create_client("deribit", testnet=True)

        >>> # Create Tardis client with API key (full access)
        >>> client = create_client("tardis", tardis_api_key="your_key")

        >>> # Create Tardis client without API key (free tier)
        >>> client = create_client("tardis")  # First day of month only

        >>> # Use environment variable for API key
        >>> os.environ["TARDIS_API_KEY"] = "your_key"
        >>> client = create_client("tardis")
    """
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
    """Add common data source arguments to argparse parser.

    This is a convenience function to add standardized client selection
    arguments to any script that needs market data access.

    Args:
        parser: argparse.ArgumentParser instance

    Returns:
        parser (for chaining)

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> add_client_args(parser)
        >>> args = parser.parse_args()
        >>> client = create_client(args.data_source, args.testnet, args.tardis_api_key)
    """
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
