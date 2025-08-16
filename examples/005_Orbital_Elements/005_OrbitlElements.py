"""
Example 5: Orbit Propagation Using Classical Orbital Elements

Problem Statement:
------------------
This example demonstrates how to numerically propagate a spacecraft’s orbit
under the two-body gravitational assumption, using **classical orbital elements (COEs)**
as the initial conditions instead of Cartesian state vectors.

The task is to:
1. Convert the given orbital elements into Earth-Centered Inertial (ECI) position
   and velocity vectors.
2. Propagate the orbit over a 24-hour period using Newtonian gravity.
3. Visualize the resulting trajectory with both static and animated 3D plots.

Initial Conditions (Classical Orbital Elements):
------------------------------------------------
- Semi-major axis: a = R_E + 600 km   (LEO orbit, where R_E = 6371 km)
- Eccentricity: e = 0.001  (nearly circular)
- Inclination: i = 98°  (near-polar orbit)
- Argument of Periapsis: ω = 40°
- Right Ascension of Ascending Node (RAAN): Ω = 120°
- True Anomaly: θ = 50°

Method:
-------
1. Use the custom class `Orbit_2body` (from orbit_util.py) to:
   - Convert orbital elements to Cartesian coordinates (`keplerian_to_cartesian`).
   - Numerically propagate the orbit for 24 hours with a time step of 10 seconds.
2. Use the `OrbitVisualizer` class to generate visualizations:
   - Static 3D orbit in space.
   - Animated orbit evolution in 3D.
   - Static ground-track orbit over Earth map.
   - Detailed animated orbit with Earth borders.

This example validates that the Keplerian-to-Cartesian conversion produces
the same trajectory as the Cartesian-based initialization used in Example 1.
"""

from orbit_util import Orbit_2body, OrbitVisualizer
import numpy as np

# ----------------------------
# Define orbital elements
# ----------------------------
R_E = 6371        # Earth's mean radius [km]
a = R_E + 600     # Semi-major axis [km]
e = 0.001         # Eccentricity (nearly circular)
i = 98            # Inclination [degrees]
w = 40            # Argument of periapsis [degrees]
RAAN = 120        # Right Ascension of Ascending Node [degrees]
theta = 50        # True anomaly [degrees]

# ----------------------------
# Initialize orbit propagator
# ----------------------------
orbit = Orbit_2body()

# Compute specific angular momentum h
# Formula: h = sqrt(mu * a * (1 - e^2))
h = np.sqrt(orbit.mu * a * (1 - e**2))

# Convert orbital elements into Cartesian state (ECI)
r_eci, v_eci = orbit.keplerian_to_cartesian(
    e, h, theta, i, RAAN, w, degree_mode=True
)

# ----------------------------
# Propagate the orbit
# ----------------------------
# 24 hours (86400 s), step size = 10 seconds
r, t = orbit.propagate_init_cond(
    T=24 * 3600,
    time_step=10,
    R0=r_eci,
    V0=v_eci
)

# ----------------------------
# Visualize results
# ----------------------------
ob = OrbitVisualizer()

# Static 3D orbit
ob.SimpleStatic(r, title="3D Orbit from Classical Orbital Elements")

# Animated 3D orbit
ob.SimpleDynamic(r[::10], t[::10], title="Animated Orbital Motion")

# Static orbit over Earth map
ob.EarthStatic(r, title="Orbit Over Earth Map")

# Detailed animation with borders
ob.EarthDynamic(r[::10], t[::10], title="Detailed Animated Orbit with Earth Borders")
