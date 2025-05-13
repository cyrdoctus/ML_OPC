# my_ai_orbital_project/data_generation/data_saver.py

import os
import numpy as np

# --- Configuration and Path Management ---
try:
    from config.app_config import (
        RAW_GENERATED_DATA_DIR,
        NORMALIZED_GENERATED_DATA_DIR,
        check_path
    )
    print("data_saver.py: Successfully imported configuration from config.app_config.")
except ImportError as e:
    print(f"data_saver.py: CRITICAL Error importing from config.app_config: {e}")
    # Fallback—fix PYTHONPATH so you don't hit this in prod.
    FALLBACK_BASE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data_training")
    )
    RAW_GENERATED_DATA_DIR = os.path.join(FALLBACK_BASE, "raw_trajectories_fallback")
    NORMALIZED_GENERATED_DATA_DIR = os.path.join(FALLBACK_BASE, "training_samples_ai1_fallback")

    def check_path(name, path, create_if_not_exists=False, is_critical=True):
        print(f"Fallback check_path: {name} = {path}")
        if not os.path.exists(path):
            if create_if_not_exists:
                os.makedirs(path, exist_ok=True)
                print(f"Created fallback directory: {path}")
                return True
            if is_critical:
                raise FileNotFoundError(f"Fallback: critical path not found: {path}")
            return False
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Fallback: path is not a directory: {path}")
        return True

# Ensure output directories exist
check_path("RAW_GENERATED_DATA_DIR", RAW_GENERATED_DATA_DIR, create_if_not_exists=True)
check_path("NORMALIZED_GENERATED_DATA_DIR", NORMALIZED_GENERATED_DATA_DIR, create_if_not_exists=True)

RAW_TRAJECTORIES_OUTPUT_DIR = RAW_GENERATED_DATA_DIR
NORMALIZED_TRAJECTORIES_OUTPUT_DIR = NORMALIZED_GENERATED_DATA_DIR


def save_raw_trajectory(
    trajectory_id: int,
    initial_coes: dict,
    initial_rv: np.ndarray,
    times: np.ndarray,
    states: np.ndarray,
    output_dir: str = None
):
    """
    Saves the raw propagated trajectory data to a compressed .npz archive.

    Files will be named `raw_trajectory_0000001.npz`, etc.
    """
    dirpath = output_dir or RAW_TRAJECTORIES_OUTPUT_DIR
    os.makedirs(dirpath, exist_ok=True)

    # flatten everything into arrays/scalars
    npz_dict = {
        "trajectory_id":       np.array(trajectory_id, dtype=int),
        "initial_state_vector_km_kms": initial_rv,
        "times_sec":           times,
        "states_km_kms":       states,
    }
    # embed each COE as its own entry (they’re scalars)
    for key, val in initial_coes.items():
        # ensure it's an array or scalar
        npz_dict[key] = np.array(val)

    filename = f"raw_trajectory_{trajectory_id:07d}.npz"
    filepath = os.path.join(dirpath, filename)

    try:
        np.savez_compressed(filepath, **npz_dict)
        print(f"Saved raw trajectory {trajectory_id} → {filepath}")
    except Exception as e:
        print(f"ERROR: Could not save raw trajectory {trajectory_id} to {filepath}: {e}")


def save_norm_trajectory(
    trajectory_id: int,
    normalized_data: dict,
    original_raw_filename: str = None,
    output_dir: str = None
):
    """
    Saves normalized trajectory data for AI training as .npz.
    """
    dirpath = output_dir or NORMALIZED_TRAJECTORIES_OUTPUT_DIR
    os.makedirs(dirpath, exist_ok=True)

    npz_dict = {
        "trajectory_id": np.array(trajectory_id, dtype=int),
    }
    # copy all normalized arrays in
    for key, val in normalized_data.items():
        npz_dict[key] = np.array(val)

    if original_raw_filename:
        # store as a small byte-string attribute
        npz_dict["original_raw_filename"] = np.array(
            original_raw_filename, dtype='S'
        )

    filename = f"norm_trajectory_{trajectory_id:07d}.npz"
    filepath = os.path.join(dirpath, filename)

    try:
        np.savez_compressed(filepath, **npz_dict)
        print(f"Saved normalized trajectory {trajectory_id} → {filepath}")
    except Exception as e:
        print(f"ERROR: Could not save normalized trajectory {trajectory_id} to {filepath}: {e}")


if __name__ == '__main__':
    print("RAW output dir:", RAW_TRAJECTORIES_OUTPUT_DIR)
    print("NORM output dir:", NORMALIZED_TRAJECTORIES_OUTPUT_DIR)
