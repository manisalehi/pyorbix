
"""
Example 7: Sun-Synchronous Circular Orbit — Multi-Model Propagation

Problem Statement:
------------------
This example designs and simulates a Sun-synchronous orbit (SSO) at an altitude of 500 km.
The goal is to determine the required inclination for Sun-synchronicity, then propagate the orbit
using three models:
- Two-Body (ideal Keplerian motion)
- J2 Perturbation (Earth’s oblateness)
- HPOP (High-Fidelity Orbit Propagator with higher-order forces)

We then compare the trajectories and ground tracks to analyze the differences between the models.

Governing Equations:
--------------------
Sun’s mean motion:
    Ω_req = 2π / (365.26 × 24 × 3600)   [rad/s]

J2-induced nodal regression:
    Ω̇_J2 = -(3/2) * J2 * (RE/a)^2 * sqrt(μ / a^3) * cos(i)

For Sun-synchronicity, require:
    cos(i) = -(2 / (3 J2)) * (a/RE)^2 * Ω_req * sqrt(a^3 / μ)

With altitude h = 500 km (a = RE + h = 6878.137 km), this yields i ≈ 97.4°.

Initial Conditions:
-------------------
- Semi-major axis: a = 6878.137 km (Earth radius + 500 km altitude)
- Eccentricity: e = 0 (circular)
- Inclination: i ≈ 97.4°
- RAAN = 0°, Argument of Perigee = 0°, True Anomaly = 0°

Method:
-------
1. Compute the required inclination for a 500 km altitude Sun-synchronous orbit.
2. Convert Keplerian orbital elements to Cartesian state vectors.
3. Propagate the orbit over 5 days with a 120-second time step using:
   - Two-Body model
   - J2 Perturbation model
   - HPOP (High-Precision Orbit Propagator)
4. Generate 3D visualizations and combined ground-track plots to compare the models.
"""

import numpy as np
from datetime import datetime, timezone
from orbit_util import Orbit_2body, OrbitVisualizer

# -------------------------
# 1. Compute Sun-synchronous inclination
# -------------------------
orb = Orbit_2body()
mu = getattr(orb, "mu", 398600.4418)     # Earth’s GM [km^3/s^2]
R_E = getattr(orb, "R_E", 6378.137)      # Earth radius [km]
J2  = getattr(orb, "J2", 1.08263e-3)     # Earth J2 coefficient

h_alt_km = 500.0                         # altitude [km]
a = R_E + h_alt_km                       # semi-major axis [km]
Omega_dot_req = 2 * np.pi / (365.26 * 24 * 3600.0)  # required regression rate [rad/s]

cos_i = -(2.0 / (3.0 * J2)) * ((a / R_E) ** 2) * Omega_dot_req * np.sqrt(a ** 3 / mu)
cos_i = np.clip(cos_i, -1.0, 1.0)
i_deg = float(np.degrees(np.arccos(cos_i)))  # expected ~97.4°

print(f"Designed SSO inclination ≈ {i_deg:.2f}° at h = {h_alt_km} km")

# -------------------------
# 2. Convert orbital elements to Cartesian state
# -------------------------
e = 0.0
RAAN = 0.0
w = 0.0
theta = 0.0

h_spec = np.sqrt(mu * a)  # specific angular momentum
R0, V0 = orb.keplerian_to_cartesian(
    e=e, h=h_spec, theta=theta, i=i_deg, RAAN=RAAN, w=w, degree_mode=True
)

# -------------------------
# 3. Set simulation parameters
# -------------------------
T_days = 5
T = T_days * 24 * 3600     # 5 days in seconds
time_step = 120            # step size [s]
scenario_epoch = datetime.now(timezone.utc)

print(f"Simulation duration: {T_days} days, time step: {time_step} s")

# -------------------------
# 4. Propagate orbit with three models
# -------------------------
sol_2body, t_2body = orb.propagate_init_cond(T=T, time_step=time_step, R0=R0, V0=V0)
sol_j2,    t_j2    = orb.propagate_with_J2(T=T, time_step=time_step, R0=R0, V0=V0)
sol_hpop,  t_hpop  = orb.HFOP(T=T, time_step=time_step, R0=R0, V0=V0, scenario_epoch=scenario_epoch)

# -------------------------
# 5. Extract positions and ground tracks
# -------------------------
r_2body = sol_2body[:, 0:3]
r_j2    = sol_j2[:, 0:3]
r_hpop  = sol_hpop[:, 0:3]

lat_2body, lon_2body = orb.lat_long_from_ECI(r_eci=r_2body, t=t_2body, scenario_epoch=scenario_epoch)
lat_j2,    lon_j2    = orb.lat_long_from_ECI(r_eci=r_j2,    t=t_j2,    scenario_epoch=scenario_epoch)
lat_hpop,  lon_hpop  = orb.lat_long_from_ECI(r_eci=r_hpop,  t=t_hpop,  scenario_epoch=scenario_epoch)

# -------------------------
# 6. Visualizations
# -------------------------
viz = OrbitVisualizer()

# 3D orbits
viz.EarthStatic(r=r_2body, title="Example 7 — Two-Body Orbit (3D)", names=["Two-Body"])
viz.EarthStatic(r=r_j2,    title="Example 7 — J2 Orbit (3D)",      names=["J2"])
viz.EarthStatic(r=r_hpop,  title="Example 7 — HPOP Orbit (3D)",    names=["HPOP"])

# Combined ground tracks
latitudes  = np.vstack([np.nan_to_num(lat_2body), np.nan_to_num(lat_j2), np.nan_to_num(lat_hpop)])
longitudes = np.vstack([np.nan_to_num(lon_2body), np.nan_to_num(lon_j2), np.nan_to_num(lon_hpop)])
viz.ground_track(latitudes=latitudes, longitudes=longitudes, names=["Two-Body", "J2", "HPOP"])

print("Example 7 complete: Results and plots generated.")
