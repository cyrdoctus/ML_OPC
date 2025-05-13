"""
orbital_utils.py

Combined translation of:
  - constmath.m
  - constastro.m
  - coe2rv.m

David Vallado, 2007 / 2025 versions.
"""

import math
import numpy as np

# --- mathematical constants ---
RAD      = 180.0 / math.pi
TWOPI    = 2.0 * math.pi
HALFPI   = 0.5 * math.pi

# --- unit conversions ---
FT2M        = 0.3048
MILE2M      = 1609.344
NM2M        = 1852.0
MILE2FT     = 5280.0
MILEPH2KMPH = 0.44704
NMPH2KMPH   = 0.5144444

# --- Earth physical constants (EGM-08) ---
RE           = 6378.1363             # km
FLAT         = 1.0 / 298.257223563
EARTH_ROT    = 7.292115e-5           # rad/s
MU_EARTH     = 398600.4415           # km^3/s^2
MU_EARTH_M   = 3.986004415e14        # m^3/s^2

# --- Derived Earth constants ---
ECC_EARTH       = math.sqrt(2.0 * FLAT - FLAT**2)
ECC_EARTH_SQ    = ECC_EARTH**2
RENM            = RE / NM2M
RE_FT           = RE * 1000.0 / FT2M
TUSEC           = math.sqrt(RE**3 / MU_EARTH)
TUMIN           = TUSEC / 60.0
TUDAY           = TUSEC / 86400.0
TUDAYSID        = TUSEC / 86164.090524
OMEGA_RAD_PER_TU= EARTH_ROT * TUSEC
OMEGA_RAD_PER_MIN=EARTH_ROT * 60.0
VEL_KM_S        = math.sqrt(MU_EARTH / RE)
VEL_FT_S        = VEL_KM_S * 1000.0 / FT2M
VEL_RAD_PER_MIN = VEL_KM_S * 60.0 / RE
DEG_PER_SEC     = (180.0 / math.pi) / TUSEC
RAD_PER_DAY     = 2.0 * math.pi * 1.002737909350795

# --- Extra astronomical constants ---
SPEED_OF_LIGHT = 299792.458    # km/s
AU             = 149597870.7   # km
EARTH2MOON     = 384400.0      # km
MOON_RADIUS    = 1738.0        # km
SUN_RADIUS     = 696000.0      # km

MASS_SUN       = 1.9891e30     # kg
MASS_EARTH     = 5.9742e24     # kg
MASS_MOON      = 7.3483e22     # kg

MU_SUN         = 1.32712428e11 # km^3/s^2
MU_MOON        = 4902.799      # km^3/s^2


def coe2rv(
    p, ecc, incl, raan, argp, nu,
    arglat: float = 0.0,
    truelon: float = 0.0,
    lonper: float = 0.0,
    mu: float = MU_EARTH
):
    """
    Convert classical orbital elements to ECI position & velocity.

    Parameters
    ----------
    p : float
        Semilatus rectum (km)
    ecc : float
        Eccentricity
    incl : float
        Inclination (rad)
    raan : float
        Right ascension of ascending node (rad)
    argp : float
        Argument of perigee (rad)
    nu : float
        True anomaly (rad)
    arglat : float, optional
        Argument of latitude (rad) for circular inclined orbits
    truelon : float, optional
        True longitude (rad) for circular equatorial orbits
    lonper : float, optional
        Longitude of periapsis (rad) for elliptical equatorial orbits
    mu : float, optional
        Gravitational parameter (km^3/s^2) (default: Earth)

    Returns
    -------
    r : ndarray, shape (3,)
        Position vector in ECI frame (km)
    v : ndarray, shape (3,)
        Velocity vector in ECI frame (km/s)
    """
    small = 1e-8

    # --- handle circular / equatorial special cases ---
    if ecc < small:
        # circular orbit
        if abs(incl) < small or abs(incl - math.pi) < small:
            # circular equatorial
            argp = 0.0
            raan = 0.0
            nu   = truelon
        else:
            # circular inclined
            argp = 0.0
            nu   = arglat
    else:
        # elliptical
        if abs(incl) < small or abs(incl - math.pi) < small:
            # elliptical equatorial
            argp = lonper
            raan = 0.0

    # prevent zero / negative p
    if abs(p) < 1e-4:
        p = 1e-4

    # PQW frame coordinates
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)
    r_pqw = np.array([
        p * cos_nu / (1.0 + ecc * cos_nu),
        p * sin_nu / (1.0 + ecc * cos_nu),
        0.0
    ])
    sqrt_mu = math.sqrt(mu)
    v_pqw = np.array([
        -sin_nu     * sqrt_mu / math.sqrt(p),
         (ecc + cos_nu) * sqrt_mu / math.sqrt(p),
         0.0
    ])

    # --- rotation helpers ---
    def rot1(vec, ang):
        ca, sa = math.cos(ang), math.sin(ang)
        R = np.array([
            [1,  0,  0],
            [0, ca, sa],
            [0,-sa, ca],
        ])
        return R.dot(vec)

    def rot3(vec, ang):
        ca, sa = math.cos(ang), math.sin(ang)
        R = np.array([
            [ ca, sa, 0],
            [-sa, ca, 0],
            [  0,  0, 1],
        ])
        return R.dot(vec)

    # Transform PQW → ECI
    temp = rot3(r_pqw, -argp)
    temp = rot1(temp, -incl)
    r = rot3(temp, -raan)

    temp = rot3(v_pqw, -argp)
    temp = rot1(temp, -incl)
    v = rot3(temp, -raan)

    return r, v
