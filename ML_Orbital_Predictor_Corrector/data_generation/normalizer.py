# ML_Orbital_Predictor_Corrector/data_generation/normalizer.py

import os
import glob
import numpy as np

from config.app_config import (
    RAW_GENERATED_DATA_DIR,
    NORMALIZED_GENERATED_DATA_DIR,
)
from config.sim_params import (
    R_EARTH_KM,
    MU_EARTH_KM3_S2
)
from .data_saver import save_norm_trajectory

# Default altitude bounds for normalization (in km)
MIN_ALTITUDE_KM = 300.0
MAX_ALTITUDE_KM = 2000.0

def normalize_trajectory_data(raw_filepath: str) -> tuple[dict, str, int]:
    """
    Load one raw .npz file, normalize its positions and velocities,
    and return a dict of normalized arrays, the original filename,
    and the trajectory_id.
    """
    arr = np.load(raw_filepath)
    times = arr['times_sec']                         # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    states = arr['states_km_kms']                   # shape (N,6) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    traj_id = int(arr['trajectory_id'].item())

    # compute normalization factors
    r_max = R_EARTH_KM + MAX_ALTITUDE_KM             # :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    r_min = R_EARTH_KM + MIN_ALTITUDE_KM
    v_max = np.sqrt(MU_EARTH_KM3_S2 / r_min)

    # normalize
    positions_norm = states[:, :3] / r_max
    velocities_norm = states[:, 3:] / v_max

    normalized = {
        'times_sec': times,
        'positions_norm': positions_norm,
        'velocities_norm': velocities_norm
    }
    original_filename = os.path.basename(raw_filepath)
    return normalized, original_filename, traj_id

def process_and_normalize_all_raw_data():
    """
    Finds every raw_trajectory_*.npz in RAW_GENERATED_DATA_DIR,
    normalizes it, and saves out norm_trajectory_*.npz in NORMALIZED_GENERATED_DATA_DIR.
    """
    pattern = os.path.join(RAW_GENERATED_DATA_DIR, "raw_trajectory_*.npz")  # :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
    raw_files = sorted(glob.glob(pattern))
    if not raw_files:
        print(f"No raw files found in {RAW_GENERATED_DATA_DIR}")
        return

    print(f"Found {len(raw_files)} raw files. Starting normalization...")
    for raw_fp in raw_files:
        normalized, orig_name, traj_id = normalize_trajectory_data(raw_fp)
        save_norm_trajectory(
            trajectory_id=traj_id,
            normalized_data=normalized,
            original_raw_filename=orig_name
        )
    print("All trajectories normalized and saved.")
