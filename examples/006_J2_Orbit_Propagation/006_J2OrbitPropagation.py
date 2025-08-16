"""
Example 6: Orbit Propagation with J2 Perturbation

Problem Statement:
------------------
This example demonstrates how to propagate a spacecraft's orbit while accounting
for the perturbing effect of Earth's oblateness, modeled by the J2 zonal harmonic.

Unlike the classical two-body model, which assumes Earth is perfectly spherical,
this extended model incorporates the J2 effect, which causes:
- Nodal regression (westward drift of the orbital plane).
- Apsidal rotation (rotation of the line of apsides).

We propagate the orbit over 10 days using the initial state vectors provided,
numerically integrating the equations of motion with J2 acceleration terms.

Initial Conditions:
-------------------
Position vector (km):
    r0 = [840.5, 485.3, 6905.8]

Velocity vector (km/s):
    v0 = [3.7821, -6.5491, 0.0057]

Method:
-------
We use the `Orbit_2body` class (from orbit_util.py), which now includes
a `propagate_with_J2()` method. This integrates the motion over 10 days
using SciPy’s ODE solver, adding the J2 perturbation to the acceleration.

Visualization is done with the `OrbitVisualizer` class, producing both
static and animated plots of the trajectory around Earth.

Additional Capabilities:
------------------------
The script demonstrates utility methods for:
- Time conversions (UTC → Julian Date, YYDDD, formatted strings).
- Ephemeris export in FreeFlyer/STK (`.e`) and SPICE (`.bsp`) formats,
  enabling interoperability with professional astrodynamics tools.
"""

import numpy as np
from datetime import datetime, timezone
from orbit_util import Orbit_2body, OrbitVisualizer

# -------------------------
# Define initial conditions
# -------------------------
# Position components (km)
X0 = 840.5
Y0 = 485.3
Z0 = 6905.8
R0 = [X0, Y0, Z0]

# Velocity components (km/s)
V0 = [3.7821, -6.5491, 0.0057]

# Combined state vector
state_0 = R0 + V0

# -----------------------------------------
# Instantiate the orbit propagator class
# -----------------------------------------
orbit = Orbit_2body()

# -----------------------------------------
# Propagate orbit with J2 perturbation
# Duration: 10 days (864,000 s)
# Time step: 10 seconds
# -----------------------------------------
T_sim_seconds = 24 * 10 * 3600  # 10 days
time_step = 10
r, t = orbit.propagate_with_J2(
    T=T_sim_seconds,
    time_step=time_step,
    R0=R0,
    V0=V0
)

# ----------------------------
# Initialize the visualizer
# ----------------------------
ob = OrbitVisualizer()

# ----------------------------
# Create visualizations
# ----------------------------
# Static 3D orbit with Earth and borders
ob.EarthStatic(r, title="Orbit with J2 Perturbation (Static View)")

# Animated orbit (downsampled for performance)
ob.EarthDynamic(r[::50], t[::50], title="Orbit with J2 Perturbation (10 days)")

# ----------------------------
# Demonstrate time utilities
# ----------------------------
current_utc_time = datetime.now(timezone.utc)
print("\n--- Time Conversion Outputs ---")
print(f"Current UTC: {current_utc_time}")
print(f"Julian Date: {orbit.UTC_to_julian(current_utc_time)}")
print(f"YYDDD Format: {orbit.UTC_to_YYDDD(current_utc_time)}")
print(f"Formatted UTC (6 decimals): {orbit.format_utc(current_utc_time, decimals=6)}")
print(f"Formatted UTC (12 decimals): {orbit.format_utc(current_utc_time, decimals=12)}")

# ----------------------------
# Export ephemeris files
# ----------------------------
positions = r[:, :3]
velocities = r[:, 3:]

scenario_start_epoch = datetime(
    year=2025, month=5, day=19, hour=8, minute=20, second=0, tzinfo=timezone.utc
)

print("\n--- Ephemeris File Generation ---")

# Export to FreeFlyer/STK format
ff_output_message = orbit.save_ephermeris_freeflyer(
    r=positions,
    v=velocities,
    t=t,
    scenario_epoch=scenario_start_epoch,
    file_name="j2_orbit_ff"
)
print(ff_output_message)

# Export to SPICE kernel format
spk_output_message = orbit.save_to_spk(
    r_vectors=positions,
    v_vectors=velocities,
    time=t,
    scenario_epoch=scenario_start_epoch,
    output_file="j2_orbit_spk",
    kernel_list=["naif0012.tls", "pck00010.tpc"],
    kernel_base_dir="./kernels"
)
print(spk_output_message)

print("\nScript finished. Check your directory for 'j2_orbit_ff.e' and 'j2_orbit_spk.bsp'.")
