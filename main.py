from orbit_util import Orbit_2body

# Initial Conditions
X_0 = 840.5  # [km]
Y_0 = 485.3  # [km]
Z_0 = 6905.8  # [km]
VX_0 = 3.7821  # [km/s]
VY_0 = -6.5491 # [km/s]
VZ_0 = 0.0057  # [km/s]
state_0 = [X_0, Y_0, Z_0, VX_0, VY_0, VZ_0]

orbit = Orbit_2body()   #Make an instance of orbit class
r, t =orbit.propagate_init_cond(T = 6*3600, time_step = 10, R0 = state_0[0:3], V0 =state_0[3:6])    #Propagte the orbit

from orbit_util import OrbitVisualizer

ob = OrbitVisualizer()  #Inilizing a visualizer

ob.SimpleStatic(r, title="3D orbit around earth")

ob.SimpleDynamic(r[::10], t[::10], title="Orbital motion of the spacecraft around the earth")

ob.EarthStatic(r, title="Plotting the orbit")

ob.EarthDynamic(r[::10],t[::10],title="Detailed Orbital motion of the spacecraft around the earth")