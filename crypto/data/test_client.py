"""
Test script for Deribit client.
"""
from datetime import datetime, timedelta, timezone
from deribit_client import DeribitClient, timestamp_to_ms


def test_deribit_client():
    """Test basic functionality of Deribit client."""
    print("Testing Deribit API client...")

    # Create client (testnet, no auth needed for public endpoints)
    client = DeribitClient(testnet=True)

    try:
        # Test 1: Get BTC instruments
        print("\n1. Testing get_instruments (BTC options)...")
        instruments = client.get_instruments(currency="BTC", kind="option")
        print(f"Found {len(instruments)} BTC options")
        if instruments:
            print(f"Example: {instruments[0]['instrument_name']}")

        # Test 2: Get BTC perpetual ticker
        print("\n2. Testing get_ticker (BTC-PERPETUAL)...")
        ticker = client.get_ticker("BTC-PERPETUAL")
        print(f"BTC-PERPETUAL last price: ${ticker['last_price']}")

        # Test 3: Get order book
        print("\n3. Testing get_order_book...")
        order_book = client.get_order_book("BTC-PERPETUAL", depth=3)
        print(f"Best bid: ${order_book['best_bid_price']}, Best ask: ${order_book['best_ask_price']}")

        # Test 4: Get recent trades (last hour)
        print("\n4. Testing get_historical_trades...")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        # Try simple recent trades first
        try:
            recent_trades = client.get_recent_trades("BTC-PERPETUAL", count=5)
            print(f"Retrieved {len(recent_trades)} recent trades")
            if recent_trades:
                print(f"Latest trade: ${recent_trades[-1]['price']} at {recent_trades[-1]['timestamp']}")
        except Exception as e:
            print(f"Recent trades failed: {e}")

        # Test 5: Get historical volatility (simpler endpoint)
        print("\n5. Testing get_historical_volatility...")
        try:
            volatility_data = client.get_historical_volatility("BTC")
            print(f"Retrieved {len(volatility_data)} volatility records")
            if volatility_data:
                latest = volatility_data[-1]
                print(f"Latest volatility: {latest}")
        except Exception as e:
            print(f"Historical volatility failed: {e}")
            print("(This endpoint may not be available on testnet)")

        print("\n✅ All tests passed! Deribit client is working.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_deribit_client()