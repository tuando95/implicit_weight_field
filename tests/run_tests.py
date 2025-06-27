"""Script to run all unit tests."""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Discover and run all tests."""
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run a specific test module."""
    # Import the test module
    module = __import__(f'tests.{test_module}', fromlist=[''])
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unit tests')
    parser.add_argument('--module', type=str, help='Specific test module to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if args.module:
        # Run specific module
        print(f"Running tests from {args.module}...")
        success = run_specific_test(args.module)
    else:
        # Run all tests
        print("Running all tests...")
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()