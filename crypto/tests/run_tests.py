"""
Test runner for crypto module unit tests.
"""
import unittest
import sys
import os

# Add the crypto directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_all_tests():
    """Run all unit tests."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
