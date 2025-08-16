"""
Example 2: Injection Velocity Analysis & Orbit Classification

Problem Statement:
------------------
This example investigates how the initial injection velocity of a spacecraft affects
the resulting orbit type under the two-body gravitational model. The spacecraft’s
initial position is fixed, and the magnitude and/or direction of its velocity is varied
to produce different conic trajectories:

- Elliptical orbit: Specific mechanical energy ε < 0
- Parabolic trajectory: Specific mechanical energy ε ≈ 0
- Hyperbolic trajectory: Specific mechanical energy ε > 0

In addition to classifying orbits based on velocity magnitude, this example also
explores how changing the *direction* of the initial velocity (yawed in the local
Radial–Tangential–Normal (RTN) frame) affects the resulting trajectory.

Initial Conditions:
-------------------
Position vector (km):
    r0 = [840.5, 485.3, 6905.8]

Base velocity vector for elliptical orbit (km/s):
    v0_elliptical = [3.7821, -6.5491, 0.0057]

Method:
-------
The specific mechanical energy is computed as:
    ε = v²/2 - μ/|r|

Where:
    - v is the speed magnitude (km/s)
    - r is the position vector magnitude (km)
    - μ = 398600.4418 km³/s² is Earth’s gravitational parameter

We use the `Orbit_2body` class (from orbit_util.py) to:
    1. Compute the specific mechanical energy and classify orbit type
    2. Propagate the orbit numerically using SciPy's ODE solver
    3. Visualize trajectories with the `OrbitVisualizer` class

The simulation is performed for:
    - Elliptical: original velocity vector
    - Parabolic: ~1.41 × elliptical velocity magnitude
    - Hyperbolic: 2.0 × elliptical velocity magnitude

We then perform a direction study:
    - Convert initial position to local RTN frame
    - Generate five yaw angles (0°, 72°, 144°, 216°, 288°) about radial vector
    - Simulate orbits for each yaw direction at fixed speed
    - Visualize all variations in static and dynamic 3D plots
"""

import numpy as np
from orbit_util import Orbit_2body, OrbitVisualizer

# -------------------------
# Define constants
# -------------------------
MU_EARTH = 398600.4418  # km^3/s^2
EARTH_RADIUS = 6371.0   # km

# -------------------------
# Common initial position
# -------------------------
R0 = [840.5, 485.3, 6905.8]  # km

# -------------------------
# 1. Basic orbit type demonstration
# -------------------------
V_ell = [3.7821, -6.5491, 0.0057]               # Elliptical
V_par = [v * 1.41 for v in V_ell]               # Parabolic (approx escape)
V_hyp = [v * 2.0  for v in V_ell]               # Hyperbolic (> escape)

def simulate_orbit(R0, V0, label, sim_time_hours=6):
    """Propagate orbit, print energy/type, and visualize."""
    orbit = Orbit_2body()
    energy = orbit.energy(R0, V0)
    otype = orbit.orbit_type(R0, V0, threshold=0.5)
    print(f"{label} → Energy: {energy:.4f} km²/s² | Type: {otype}")
    r, t = orbit.propagate_init_cond(T=sim_time_hours*3600, time_step=10, R0=R0, V0=V0)

    ob = OrbitVisualizer()
    ob.EarthStatic(r, title=f"{label} - Static View")
    ob.EarthDynamic(r[::5], t[::5], title=f"{label} - Dynamic View")
    return r, t

# Run simulations for the three orbit types
r_ell, t_ell = simulate_orbit(R0, V_ell, "Elliptical Orbit")
r_par, t_par = simulate_orbit(R0, V_par, "Parabolic Orbit")
r_hyp, t_hyp = simulate_orbit(R0, V_hyp, "Hyperbolic Orbit")

# -------------------------
# 2. Combined orbit visualization
# -------------------------
ob = OrbitVisualizer()
ob.EarthStatic(
    [r_ell[::10], r_par[::10], r_hyp[::10]],
    names=["Elliptical", "Parabolic", "Hyperbolic"],
    colors=["yellow", "light blue", "red"]
)
ob.EarthDynamic(
    [r_ell[::10], r_par[::10], r_hyp[::10]],
    t_ell[::10],
    names=["Elliptical", "Parabolic", "Hyperbolic"],
    colors=["yellow", "light blue", "red"]
)

# -------------------------
# 3. Orbit direction variation (RTN frame)
# -------------------------
# Create local RTN frame
R0_vec = np.array(R0)
r_hat = R0_vec / np.linalg.norm(R0_vec)
z_axis = np.array([0, 0, 1])
t_temp = np.cross(z_axis, R0_vec)
t_hat = t_temp / np.linalg.norm(t_temp)
n_hat = np.cross(r_hat, t_hat)

# Escape velocity at initial position
v_escape = np.sqrt(2 * MU_EARTH / np.linalg.norm(R0_vec))

# Orbit configurations: (velocity magnitude, simulation time)
orbit_configs = {
    'Elliptical': (0.9 * v_escape, 36000),  # 10 h
    'Parabolic':  (1.0 * v_escape, 18000),  # 5 h
    'Hyperbolic': (1.2 * v_escape, 18000)   # 5 h
}

# Yaw angles in degrees
yaw_angles = np.linspace(0, 360, 5, endpoint=False)

# Rotation utility
def rotate_vector(v, axis, angle_deg):
    ang = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(ang) +
            np.cross(axis, v) * np.sin(ang) +
            axis * np.dot(axis, v) * (1 - np.cos(ang)))

# Color palette per orbit type
colors = {
    'Elliptical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'Parabolic':  ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'Hyperbolic': ['#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']
}

# Simulate and visualize yaw variations for each orbit type
for orbit_type, (v_mag, Tsim) in orbit_configs.items():
    base_vel = v_mag * t_hat
    trajs, labs, cols = [], [], []
    for idx, yaw in enumerate(yaw_angles):
        v_rot = rotate_vector(base_vel, r_hat, yaw)
        solver = Orbit_2body()
        r, t = solver.propagate_init_cond(
            T=Tsim, time_step=10, R0=R0_vec.tolist(), V0=v_rot.tolist()
        )
        if np.all(np.linalg.norm(r, axis=1) > EARTH_RADIUS):  # Avoid impact
            trajs.append(r[::20])
            labs.append(f"{orbit_type} | yaw={yaw:.0f}°")
            cols.append(colors[orbit_type][idx])
    ob = OrbitVisualizer()
    ob.EarthStatic(trajs, names=labs, colors=cols,
                   title=f"{orbit_type} Orbits - 5 Yaw Directions")
    ob.EarthDynamic(trajs, trajs[0][:, 0] * 0, names=labs, colors=cols,
                    title=f"{orbit_type} Orbits - Dynamic, 5 Directions")
