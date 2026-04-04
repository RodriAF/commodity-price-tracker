# conftest.py
# pytest configuration for the commodity tracker test suite.
# Place this file in the project root or the tests/ directory.

import pytest
import warnings

# Suppress noisy warnings from statsmodels and arch during test runs.
# These are expected convergence warnings from statistical models running
# on synthetic data with limited observations — not bugs.
def pytest_configure(config):
    warnings.filterwarnings('ignore', category=UserWarning,    module='statsmodels')
    warnings.filterwarnings('ignore', category=FutureWarning,  module='statsmodels')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='arch')
    warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
