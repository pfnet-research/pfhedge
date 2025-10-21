"""Entry point for running training as a module.

Allows running training with:
    python -m crypto.training [args]

This is equivalent to:
    python -m crypto.training.train_model [args]
"""

from crypto.training.train_model import main

if __name__ == "__main__":
    exit(main())
