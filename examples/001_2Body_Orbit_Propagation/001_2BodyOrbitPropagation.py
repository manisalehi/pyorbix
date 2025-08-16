"""
Example 1: 2-Body Orbit Propagation Using Initial State Vectors

Problem Statement:
------------------
This example demonstrates how to numerically solve the two-body orbital motion problem
for a spacecraft in Earth orbit using its initial position and velocity vectors.

The goal is to propagate the orbit over a 24-hour period using Newtonian gravitational
physics, assuming:
- Earth is a point mass at the origin.
- No perturbations other than gravity (ideal 2-body scenario).
- Initial state is given in the Earth-Centered Inertial (ECI) coordinate frame.

Initial Conditions:
-------------------
Position vector (km):
    r0 = [840.5, 485.3, 6905.8]

Velocity vector (km/s):
    v0 = [3.7821, -6.5491, 0.0057]

Method:
-------
We use a custom class `Orbit_2body` (from orbit_util.py) to integrate the equations
of motion using SciPy's `odeint` method with a time step of 10 seconds.
Visualization is done using the `OrbitVisualizer` class, which produces both static
and animated 3D plots.

This simulation demonstrates classical orbital motion and helps validate
numerical methods against analytical expectations.
"""

from orbit_util import Orbit_2body, OrbitVisualizer

# -------------------------
# Define initial conditions
# -------------------------
# Position components (km)
X_0 = 840.5
Y_0 = 485.3
Z_0 = 6905.8

# Velocity components (km/s)
VX_0 = 3.7821
VY_0 = -6.5491
VZ_0 = 0.0057

# Combined initial state vector
state_0 = [X_0, Y_0, Z_0, VX_0, VY_0, VZ_0]

# -----------------------------------------
# Instantiate the orbit propagator class
# -----------------------------------------
orbit = Orbit_2body()

# -----------------------------------------
# Propagate orbit for 24 hours (86400 s)
# Time step = 10 seconds
# -----------------------------------------
r, t = orbit.propagate_init_cond(
    T=24 * 3600,
    time_step=10,
    R0=state_0[0:3],  # Initial position vector
    V0=state_0[3:6]   # Initial velocity vector
)

# ----------------------------
# Initialize the visualizer
# ----------------------------
ob = OrbitVisualizer()

# ----------------------------
# Create visualizations
# ----------------------------

# Static 3D view of orbit around Earth
ob.SimpleStatic(r, title="3D orbit around Earth")

# Animated 3D orbit (downsampled for performance)
ob.SimpleDynamic(r[::10], t[::10], title="Orbital motion around Earth")

# Static orbit with Earth map and borders
ob.EarthStatic(r, title="Orbit over Earth with country borders")

# Detailed animation with borders and Earth model
ob.EarthDynamic(r[::10], t[::10], title="Detailed orbital motion over Earth")
