# training/__init__.py
"""
Training package for orbital AI models.
Provides entry points for training the predictor (AI1) and corrector (AI2).
"""

from .predictor import train_predictor
#from .corrector import train_corrector

__all__ = ["train_predictor"]

# , "train_corrector"