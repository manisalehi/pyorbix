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


class OrbitVisualizer():
    def simpleStatic(self, r, title="3D orbit around earth"):
        
        # Create figure
        fig = go.Figure()

        # Define central sphere (Earth-like representation)
        num_points = 50  # Sphere resolution
        theta, phi = np.meshgrid(np.linspace(0, np.pi, num_points), np.linspace(0, 2*np.pi, num_points))

        # Scale the sphere to a reasonable size
        sphere_radius = 6371  # Example size for visibility

        x_sphere = sphere_radius * np.sin(theta) * np.cos(phi)
        y_sphere = sphere_radius * np.sin(theta) * np.sin(phi)
        z_sphere = sphere_radius * np.cos(theta)

        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, "blue"], [1, "blue"]],  # Blue sphere
            showscale=False
        ))

        # Preserve all existing elements, including the orbit
        fig.add_trace(go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2],
            mode="lines",
            line=dict(color="white", width=2),
            name="Orbit"
        ))

        # Set layout for clean 3D interaction
        fig.update_layout(
            title=title,  # Add title here
            title_font=dict(size=24, color="white"),  # Customize font size and color
            width=1200,
            height=1200,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False),
                yaxis=dict(showbackground=False, showgrid=False),
                zaxis=dict(showbackground=False, showgrid=False),
            )
        )

        # Show plot
        fig.show()



    

    