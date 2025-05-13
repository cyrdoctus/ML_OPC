# my_ai_orbital_project/config/sim_params.py
"""
Simulation Parameters for Orbital Mechanics and AI-Driven Simulation.
"""

# --- General Time Settings ---
TIME_STEP_S = 60.0  # Simulation time step in seconds (for propagation and AI updates)
MAX_SIM_TIME_S = 3 * 24 * 3600.0  # Maximum simulation duration in seconds (e.g., 3 days)

# --- Earth Constants & Perturbations ---
MU_EARTH_KM3_S2 = 398600.4412  # Earth gravitational parameter (km^3/s^2)
R_EARTH_KM = 6378.137         # Earth equatorial radius (km)
J2_EARTH = 0.00108263         # J2 perturbation coefficient for Earth
OMEGA_EARTH = 7.29211576e-5 # rotation tate of earth rad/s

ENABLE_J2_PERTURBATION = True
ENABLE_DRAG_PERTURBATION = False # Not fully implemented

# --- Spacecraft Parameters (example, adjust as needed) ---
SPACECRAFT_MASS_KG = 500.0          # Mass of the spacecraft in kg
SPACECRAFT_DRAG_COEFF = 2.2         # Drag coefficient (Cd)
SPACECRAFT_DRAG_AREA_M2 = 5.0       # Cross-sectional area subject to drag in m^2

# --- Numerical Integration Settings for ODE Solver (e.g., SciPy's solve_ivp) ---
ODE_SOLVER_METHOD = 'RK45'        # 'RK45', 'DOP853', 'LSODA' (good for stiff problems)
ODE_RELATIVE_TOLERANCE = 1e-9     # Relative tolerance for the ODE solver
ODE_ABSOLUTE_TOLERANCE = 1e-12    # Absolute tolerance for the ODE solver

# --- AI-Driven Simulation Specific Parameters ---
DEVIATION_THRESHOLD_KM = 10.0     # Max allowed deviation from reference orbit before AI2 correction (km)
PREDICTION_HORIZON_S = 3600.0     # How far ahead AI1 should predict (in seconds)
                                  # (This should align with how AI1 is trained)
MANEUVER_EXECUTION_ASSUMPTION = "INSTANTANEOUS" # vs "FINITE_BURN" (finite burn is more complex)

# --- Reference Orbit Parameters (example for a LEO) ---
# These might be overridden by specific scenarios in data_generation or simulation_setup
REF_ORBIT_SEMI_MAJOR_AXIS_KM = R_EARTH_KM + 700.0 # Example: 700 km altitude LEO
REF_ORBIT_ECCENTRICITY = 0.001
REF_ORBIT_INCLINATION_DEG = 51.6
REF_ORBIT_RAAN_DEG = 0.0
REF_ORBIT_ARG_PERIGEE_DEG = 0.0
REF_ORBIT_TRUE_ANOMALY_DEG = 0.0

print("Simulation parameters from sim_params.py loaded.")