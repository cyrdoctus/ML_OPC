# ML_Orbital_Predictor_Corrector/data_generation/__init__.py

"""
Data Generation Package

This package is responsible for:
- Defining orbital scenarios (scenarios.py).
- Saving raw and normalized trajectory data (data_saver.py).
- Normalizing raw data and engineering features (normalizer.py).
- Orchestrating the entire data generation pipeline (run_generation.py).
"""

# From scenarios.py:
try:
    from .scenarios import LEO_ORBIT_ELEMENT_RANGES
except ImportError:
    print("Warning: Could not import LEO_ORBIT_ELEMENT_RANGES from data_generation.scenarios")
    LEO_ORBIT_ELEMENT_RANGES = {} # Fallback

# From data_saver.py:
# Expose the saving functions and the output directory paths.
try:
    from .data_saver import (
        save_raw_trajectory,
        save_norm_trajectory,
        RAW_TRAJECTORIES_OUTPUT_DIR,
        NORMALIZED_TRAJECTORIES_OUTPUT_DIR
    )
except ImportError:
    print("Warning: Could not import from data_generation.data_saver")
    # Define fallbacks or leave them undefined if critical
    def save_raw_trajectory(*args, **kwargs):
        print("Fallback save_raw_trajectory called. Please check data_saver.py.")
    def save_norm_trajectory(*args, **kwargs):
        print("Fallback save_norm_trajectory called. Please check data_saver.py.")
    RAW_TRAJECTORIES_OUTPUT_DIR = "data_training/raw_trajectories_fallback_init"
    NORMALIZED_TRAJECTORIES_OUTPUT_DIR = "data_training/normalized_trajectories_fallback_init"


# From normalizer.py
try:
    from .normalizer import (
        normalize_single_trajectory,
        process_and_save_all  # this is your batch‐runner
    )
except ImportError:
    print("Warning: Could not import from data_generation.normalizer")
    def normalize_single_trajectory(*args, **kwargs):
        print("Fallback normalize_single_trajectory called. Please check normalizer.py.")
    def process_and_save_all(*args, **kwargs):
        print("Fallback process_and_save_all called. Please check normalizer.py.")


__all__ = [
    'LEO_ORBIT_ELEMENT_RANGES',
    'save_raw_trajectory',
    'save_norm_trajectory',
    'RAW_TRAJECTORIES_OUTPUT_DIR',
    'NORMALIZED_TRAJECTORIES_OUTPUT_DIR',
    'normalize_single_trajectory',
    'process_and_save_all',
]

print("Data Generation package initialized.")
