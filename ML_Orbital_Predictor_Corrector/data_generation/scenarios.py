# my_ai_orbital_project/data_generation/scenarios.py

import numpy as np

EARTH_RADIUS_KM = 6378.1363  # km

# ------------------------------------------------------------------------------
# Define LEO Scenario Parameters
# ------------------------------------------------------------------------------

# 1. Altitude Range for LEO orbits
MIN_ALTITUDE_KM = 300.0
MAX_ALTITUDE_KM = 2000.0

# Derived Semi-Major Axis (SMA) range (a)
# For near-circular orbits, sma is approximately Earth radius + altitude
MIN_SMA_KM = EARTH_RADIUS_KM + MIN_ALTITUDE_KM
MAX_SMA_KM = EARTH_RADIUS_KM + MAX_ALTITUDE_KM

# 2. Eccentricity (ecc)
# LEO orbits are typically near-circular.
MIN_ECCENTRICITY = 0.0
MAX_ECCENTRICITY = 0.5

# Constraint for minimum perigee altitude (can be same as MIN_ALTITUDE_KM)
MIN_PERIGEE_ALTITUDE_KM = MIN_ALTITUDE_KM

# 3. Inclination (incl)
# Range: 0 to 180 degrees (0 to pi radians)
MIN_INCLINATION_RAD = 0.0
MAX_INCLINATION_RAD = np.pi

# 4. Right Ascension of the Ascending Node (RAAN)
# Range: 0 to 360 degrees (0 to 2*pi radians)
MIN_RAAN_RAD = 0.0
MAX_RAAN_RAD = 2 * np.pi

# 5. Argument of Perigee (argp)
# Range: 0 to 360 degrees (0 to 2*pi radians)
MIN_ARGP_RAD = 0.0
MAX_ARGP_RAD = 2 * np.pi

# 6. True Anomaly (nu) - starting true anomaly for the trajectory
# Range: 0 to 360 degrees (0 to 2*pi radians)
MIN_NU_RAD = 0.0
MAX_NU_RAD = 2 * np.pi

# It can be useful to package these into a dictionary for easy import
LEO_ORBIT_ELEMENT_RANGES = {
    'sma_km': (MIN_SMA_KM, MAX_SMA_KM),
    'ecc': (MIN_ECCENTRICITY, MAX_ECCENTRICITY),
    'incl_rad': (MIN_INCLINATION_RAD, MAX_INCLINATION_RAD),
    'raan_rad': (MIN_RAAN_RAD, MAX_RAAN_RAD),
    'argp_rad': (MIN_ARGP_RAD, MAX_ARGP_RAD),
    'nu_rad': (MIN_NU_RAD, MAX_NU_RAD),
    'min_perigee_altitude_km': MIN_PERIGEE_ALTITUDE_KM,
    'earth_radius_km': EARTH_RADIUS_KM # Include for convenience
}

if __name__ == '__main__':
    print("--- LEO Scenario Parameters ---")
    for key, value in LEO_ORBIT_ELEMENT_RANGES.items():
        if isinstance(value, tuple):
            print(f"  Range for {key}: {value[0]:.2f} to {value[1]:.2f}")
        else:
            print(f"  {key}: {value:.2f}")
    print(f"\nNote: Eccentricity will be further constrained by SMA to ensure perigee altitude "
          f">= {LEO_ORBIT_ELEMENT_RANGES['min_perigee_altitude_km']:.2f} km.")

