"""Entry point for running backtest as a module.

Allows running backtest with:
    python -m crypto.backtest [args]

This is equivalent to:
    python -m crypto.backtest.run [args]
"""

from crypto.backtest.run import main

if __name__ == "__main__":
    main()
