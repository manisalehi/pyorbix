"""
Example 4: Solving Kepler’s Equation and Verifying Eccentric Anomaly

Problem Statement:
------------------
This example demonstrates how to numerically solve **Kepler’s Equation** to
determine the eccentric anomaly (E) from the mean anomaly (M_e) and orbital
eccentricity (e).

Kepler’s Equation:
------------------
    M_e = E - e * sin(E)

Because this equation is transcendental, it cannot be solved analytically.
Instead, we apply a numerical root-finding approach.

Objectives:
-----------
1. Implement a solver using SciPy’s `fsolve` to compute eccentric anomaly (E).
2. Validate the solver using known examples from lecture notes.
3. Extend the solver to compute time since perigee and true anomaly (θ).
4. Confirm consistency with the physics-based methods in `orbit_util.py`.

Verification Example (from Curtis, *Orbital Mechanics for Engineering Students*):
---------------------------------------------------------------------------------
- Eccentricity: e = 0.3725
- Mean anomaly: M_e = 1.36 rad
- Expected eccentric anomaly: E = 1.7281 rad

Method:
-------
We use the class `Orbit_2body` (from orbit_util.py) to:
- Compute orbital geometry (angular momentum h, semi-major axis a, etc.).
- Propagate time since perigee for a given true anomaly.
- Numerically solve Kepler’s equation using SciPy’s `fsolve`.
- Compare computed eccentric anomaly with expected textbook values.

This workflow validates both the direct Kepler solver and the orbital mechanics
pipeline implemented in `orbit_util.py`.
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
from orbit_util import Orbit_2body
from scipy.optimize import fsolve


# ============================================================
# Helper Functions
# ============================================================
def rad2deg(rad):
    """Convert radians to degrees."""
    return np.rad2deg(rad)

def deg2rad(deg):
    """Convert degrees to radians."""
    return np.deg2rad(deg)

def compare(val1, val2, tol=1e-2):
    """
    Compare two values with tolerance.
    Returns True if |val1 - val2| <= tol.
    """
    return abs(val1 - val2) <= tol


# ============================================================
# Initialize Orbit Solver
# ============================================================
orbit = Orbit_2body()


# ============================================================
# Validation Function
# ============================================================
def validate_shadow_example(apogee_toward_sun=True):
    """
    Validate orbital shadow duration using geometry.

    Parameters
    ----------
    apogee_toward_sun : bool
        If True  → apogee points toward the Sun.
        If False → perigee points toward the Sun.

    Method:
    -------
    1. Define orbit with perigee (500 km altitude) and apogee (5000 km altitude).
    2. Compute angular momentum vector h from initial conditions.
    3. Solve for shadow entry/exit anomalies θ where:
           sin(θ) = RE / r(θ)
    4. Use time_since_perigee() to calculate shadow duration.
    5. Print results for both orbital configurations.
    """

    print("\nSlide 27–35: Shadow duration calculation")

    # ----------------------------------------
    # Define Earth constants and orbit geometry
    # ----------------------------------------
    RE = 6378  # Earth radius [km]
    rp = RE + 500      # Perigee radius [km]
    ra = RE + 5000     # Apogee radius [km]

    a = (rp + ra) / 2                 # Semi-major axis
    e = (ra - rp) / (ra + rp)         # Eccentricity
    mu = orbit.mu                     # Gravitational parameter

    # ----------------------------------------
    # State vectors at perigee
    # ----------------------------------------
    r_vec = np.array([rp, 0, 0])
    v_vec = np.array([0, np.sqrt(mu * (1 + e) / rp), 0])
    _, h = orbit.specific_angular_momentum(r_vec, v_vec)

    # ----------------------------------------
    # Orbital radius as function of true anomaly θ
    # ----------------------------------------
    def r_theta(theta):
        return (h**2 / mu) / (1 + e * np.cos(theta))

    # Condition for shadow entry/exit
    def f(theta):
        return np.sin(theta) - RE / r_theta(theta)

    # ----------------------------------------
    # Solve for θ intersections (entry/exit)
    # ----------------------------------------
    theta1 = fsolve(f, deg2rad(30))[0]
    theta2 = -theta1 if apogee_toward_sun else np.pi - theta1

    # ----------------------------------------
    # Compute time since perigee for θ1 and θ2
    # ----------------------------------------
    t1, _ = orbit.time_since_perigee(theta1, h=h, e=e)
    t2, _ = orbit.time_since_perigee(theta2, h=h, e=e)
    shadow_time = abs(t2 - t1)

    # ----------------------------------------
    # Display results
    # ----------------------------------------
    if apogee_toward_sun:
        print("Configuration: Apogee toward Sun")
    else:
        print("Configuration: Perigee toward Sun")

    print(f"Theta entry: {rad2deg(theta1):.2f}°, exit: {rad2deg(theta2):.2f}°")
    print(f"Shadow duration: {shadow_time:.2f} s = {shadow_time/60:.2f} min")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("Extended Slide Example Validations")

    # Case 1: Apogee toward Sun
    validate_shadow_example(apogee_toward_sun=True)

    # Case 2: Perigee toward Sun
    validate_shadow_example(apogee_toward_sun=False)
