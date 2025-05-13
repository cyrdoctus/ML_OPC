# my_ai_orbital_project/run_data_gen.py

import numpy as np
import time
import os
import multiprocessing
from functools import partial

# --- Configuration ---
NUM_ORBITS_TO_GENERATE = 1000
PROPAGATION_DURATION = 90.0 * 24 * 3600  # seconds
TIME_STEP = 10.0 * 60                    # seconds
NUM_CORES = None  # None → os.cpu_count()

# --- Project-Specific Imports ---
from data_generation.scenarios import LEO_ORBIT_ELEMENT_RANGES
from orbital_dynamics.coe2rv import coe2rv
from orbital_dynamics.propagator import propagate_orbit
from data_generation.data_saver import save_raw_trajectory
from config import MU_EARTH_KM3_S2, RAW_GENERATED_DATA_DIR

def generate_single_random_coe(ranges):
    # ... (same as before) ...
    sma = np.random.uniform(ranges['sma_km'][0], ranges['sma_km'][1])
    min_perigee = ranges['earth_radius_km'] + ranges['min_perigee_altitude_km']
    ecc_max = max(0.0, 1 - min_perigee / sma)
    ecc = np.random.uniform(ranges['ecc'][0], min(ranges['ecc'][1], ecc_max))
    incl = np.random.uniform(*ranges['incl_rad'])
    raan = np.random.uniform(*ranges['raan_rad'])
    argp = np.random.uniform(*ranges['argp_rad'])
    nu   = np.random.uniform(*ranges['nu_rad'])
    p = sma * (1 - ecc ** 2)
    return dict(p_km=p, ecc=ecc, incl_rad=incl,
                raan_rad=raan, argp_rad=argp, nu_rad=nu,
                sma_km=sma)

def coe_to_rv(coe):
    r, v = coe2rv(
        p=coe['p_km'], ecc=coe['ecc'], incl=coe['incl_rad'],
        raan=coe['raan_rad'], argp=coe['argp_rad'], nu=coe['nu_rad'],
        mu=MU_EARTH_KM3_S2
    )
    return coe, np.concatenate((r, v))

def propagate_and_save(args):
    idx, coe, rv0 = args
    t_end = PROPAGATION_DURATION
    dt    = TIME_STEP
    t_eval = np.arange(0, t_end + dt, dt)
    try:
        times, states, _ = propagate_orbit(
            initial_state_vector=rv0,
            t_span=(0, t_end),
            t_eval=t_eval
        )
        if states is not None and len(states):
            save_raw_trajectory(
                trajectory_id=idx,
                initial_coes=coe,
                initial_rv=rv0,
                times=times,
                states=states,
            )
            return True
    except Exception as e:
        print(f"[Propagate #{idx}] Error: {e}")
    return False

def main():
    np.random.seed(0)
    cpu_count = NUM_CORES or os.cpu_count()

    # 1) Generate COEs
    all_coes = [generate_single_random_coe(LEO_ORBIT_ELEMENT_RANGES)
                for _ in range(NUM_ORBITS_TO_GENERATE)]

    # 2) COE → RV in parallel
    with multiprocessing.Pool(cpu_count) as pool:
        coe_rv_pairs = pool.map(coe_to_rv, all_coes)

    # Filter out any failures
    successful = [(i, coe, rv)
                  for i, (coe, rv) in enumerate(coe_rv_pairs)
                  if rv is not None]

    # 3) Propagate + save in parallel
    #    We pack (index, coe, rv) into args for each worker
    args = successful
    with multiprocessing.Pool(cpu_count) as pool:
        results = pool.map(propagate_and_save, args)

    print(f"Successfully generated {sum(results)}/{len(results)} trajectories.")

if __name__ == "__main__":
    main()
