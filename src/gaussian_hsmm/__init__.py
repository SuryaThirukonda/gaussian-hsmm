"""Gaussian hidden semi-Markov models with explicit duration distributions."""

from .model import ConvergenceMonitor, GaussianHSMM

__version__ = "0.1.0"
__all__ = ["ConvergenceMonitor", "GaussianHSMM", "__version__"]
