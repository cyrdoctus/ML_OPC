# ML_Orbital_Predictor_Corrector/orbit_dynamics/__init__.py

"""
Orbit Dynamics Package

This package provides tools for orbit propagation and related coordinate transformations.
"""

from .propagator import propagate_orbit, orbital_ode

# Uncomment and adjust the following lines based on the actual contents
# of your other modules in this package (perturbations.py, coordinate_systems.py)

# from .coordinate_systems import (
#     keplerian_to_cartesian,
#     cartesian_to_keplerian,
#     eci_to_ecef,
#     ecef_to_eci
#     # Add other relevant coordinate functions
# )



# Define __all__ to specify what 'from orbit_dynamics import *' should import.
# It's good practice to define this, even if you don't plan to use 'import *'.
__all__ = [
    'propagate_orbit',
    'orbital_ode', # Exposing the ODE might be useful for analysis or custom integrators

    # Add the names of functions/classes imported from other modules here
    # if you uncommented their imports above.
    # 'keplerian_to_cartesian',
    # 'cartesian_to_keplerian',
    # 'eci_to_ecef',
    # 'ecef_to_eci',
    # 'calculate_j2_perturbation',
    # 'calculate_drag_perturbation',
]