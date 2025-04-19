#Dependencies
import numpy as np 
from scipy.integrate import odeint
import plotly.graph_objects as go
import math

#The orbit propagetor
class Orbit_2body():
    def __init__(self, R0 = None, V0 = None):
        self.mu = 3.986004418E+05  # Earth's gravitational parameter  
                        # [km^3/s^2]

    #Propagting the orbit from the intial conditons
    def propagate_init_cond(self, T, time_step, R0, V0):
        
        S0 = np.hstack([R0, V0])            #Inital condition state vector
        t = np.arange(0, T, time_step)     #The time step's to solve the equation for

        #Numerically solving the equation 
        sol = odeint(self.dS_dt, S0, t)

        #Saving the propagted orbit
        self.orbit = sol    
        self.time = t

        return sol, t

    #Calculating the dS/dt with the 2 Body differential equation 
    def dS_dt(self, state ,t):  

        x = state[0]
        y = state[1]
        z = state[2]
        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]

        x_ddot = -self.mu * x / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
        y_ddot = -self.mu * y / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
        z_ddot = -self.mu * z / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
        ds_dt = np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])

        return ds_dt


class Orbit_visualizer():

    def static(self, orbit ,show_earth=True):
        # Create figure
        fig = go.Figure()

        # Add map with country borders
        fig.add_trace(go.Scattergeo(
            locationmode="ISO-3",
            lon=[],  # Automatically handled by Plotly
            lat=[],
            mode="lines",
            line=dict(width=1, color="grey")  # Grey borders
        ))

        # Set layout with larger size & custom colors
        fig.update_layout(
            width=1200,  # Increased width
            height=1200,  # Increased height
            geo=dict(
                projection_type="orthographic",  # Sphere-like Earth
                showcoastlines=True,
                coastlinecolor="grey",  # Grey coastlines
                showland=True,
                landcolor="lightblue",  # Light blue Earth
                showcountries=True,
                countrycolor="grey",  # Grey country borders
                bgcolor="black"  # Black background
            ),
            paper_bgcolor="black"  # Fully black outer background
        )

        # Show plot
        fig.show()


    

    