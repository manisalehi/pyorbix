"""
Example 3: Orbital State Propagation in the Perifocal Coordinate System

Problem Statement:
------------------
This example demonstrates the analytical propagation of a spacecraft's
state vector within the perifocal coordinate system (or a z-rotated
version of it), given an initial position and velocity vector, and a
specified change in true anomaly.

The goal is to determine:
1. The spacecraft’s new position and velocity vectors after advancing
   by the specified true anomaly change.
2. The rotation angle between the input frame and the standard perifocal frame.
3. The conservation of specific angular momentum and mechanical energy
   before and after propagation to validate numerical accuracy.

Initial Conditions:
---------
We are given:
- Initial position vector in (rotated) perifocal frame (km):
    r0 = [3450, -1700, 7750]
- Initial velocity vector in (rotated) perifocal frame (km/s):
    v0 = [5.4, -5.4, 1.0]
- Change in true anomaly:
    Δθ = 82°

Method:
-------
We use the `perifocal_calculator()` method from `Orbit_2body` (in orbit_util.py)
to propagate the orbit analytically using Lagrange coefficients.
This function:
- Computes specific angular momentum.
- Determines the radial velocity and uses classical orbital geometry
  to get the new state.
- Detects if the input frame is rotated about the z-axis relative to
  the pure perifocal frame.

We also verify:
- Conservation of specific angular momentum magnitude.
- Conservation of specific mechanical energy.

Visualization is performed using `OrbitVisualizer` to plot the initial and final
state vectors for qualitative inspection.
"""

import numpy as np
from orbit_util import Orbit_2body, OrbitVisualizer

# -------------------------
# Define initial conditions
# -------------------------
# Position vector in perifocal (or rotated perifocal) frame [km]
r0 = [3450, -1700, 7750]

# Velocity vector in perifocal (or rotated perifocal) frame [km/s]
v0 = [5.4, -5.4, 1.0]

# Change in true anomaly [degrees]
delta_true_anomaly = 82

# -----------------------------------------
# Instantiate the orbit computation class
# -----------------------------------------
orbit = Orbit_2body()

# ------------------------------------------------
# Compute new position & velocity after Δθ change
# ------------------------------------------------
r_new, v_new, rotation_angle = orbit.perifocal_calculator(r0, v0, delta_true_anomaly)

# --------------------
# Display key results
# --------------------
print("Initial Position Vector r0 [km]:", r0)
print("Initial Velocity Vector v0 [km/s]:", v0)
print("Delta True Anomaly [°]:", delta_true_anomaly)
print("→ New Position Vector r [km]:", r_new)
print("→ New Velocity Vector v [km/s]:", v_new)
print(f"→ Rotation angle (Z-axis): {rotation_angle:.2f}°")

# ---------------------------------------------
# Conservation of angular momentum & energy
# ---------------------------------------------
# Angular momentum before and after
h_vec_0, h_mag_0 = orbit.specific_angular_momentum(r0, v0)
h_vec_1, h_mag_1 = orbit.specific_angular_momentum(r_new, v_new)

# Mechanical energy before and after
energy_0 = orbit.energy(r0, v0)
energy_1 = orbit.energy(r_new, v_new)

print("\n=== Conservation Check ===")
print(f"Initial Specific Angular Momentum: {h_mag_0:.4f} km²/s")
print(f"Final   Specific Angular Momentum: {h_mag_1:.4f} km²/s")
print(f"Change: {abs(h_mag_1 - h_mag_0):.6f} km²/s")

print(f"Initial Mechanical Energy: {energy_0:.4f} km²/s²")
print(f"Final   Mechanical Energy: {energy_1:.4f} km²/s²")
print(f"Change: {abs(energy_1 - energy_0):.6f} km²/s²")

# -----------------------------
# Frame alignment check
# -----------------------------
if abs(rotation_angle) < 1.0:
    print("\n→ Frame is aligned with Perifocal coordinate system.")
else:
    print("\n→ Frame is a rotated version of the Perifocal coordinate system.")

# ---------------------------------
# Visualization of initial & final
# ---------------------------------
viz = OrbitVisualizer()
viz.SimpleStatic([r0, r_new], names=["Initial", "Final"], colors=["blue", "orange"])
viz.SimpleDynamic([r0, r_new], [0, 1], names=["Initial", "Final"], colors=["blue", "orange"])
