#Dependencies
import numpy as np 
from scipy.integrate import odeint
import plotly.graph_objects as go
import requests
import random


#The orbit propagetor
class Orbit_2body():
    def __init__(self, R0 = None, V0 = None):
        # Earth's gravitational parameter  
        self.mu = 3.986004418E+05  # [km^3/s^2]
        self.s = np.array([])
        self.t = np.array([])
        
                        
    #Propagting the orbit from the intial conditons
    def propagate_init_cond(self, T, time_step, R0, V0):
        "Propagting the orbit using the inital conditions"
        
        S0 = np.hstack([R0, V0])            #Inital condition state vector
        t = np.arange(0, T, time_step)     #The time step's to solve the equation for

        #Numerically solving the equation 
        sol = odeint(self.dS_dt, S0, t)

        #Saving the propagted orbit
        self.s = sol    
        self.t = t

        return sol, t

    #Calculating the dS/dt with the 2 Body differential equation 
    def dS_dt(self, state ,t):  
        "Returning the time derivative of the state vector"

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

    #✅Calculating the Specific Mechanical Energy of the Orbit_2body
    def energy(self, r, v):
        #Turn the r and v into numpy array
        r = np.array(r)
        v = np.array(v)

        #Magnitude of V and r vector
        v_mag =np.sqrt(v.dot(v))
        r_mag =np.sqrt(r.dot(r))
        

        #Finding the specific energy of the orbit 
        energy = (v_mag ** 2) / 2 - self.mu / r_mag     # epsilon = v^2/2 - mu/r  

        return energy

    #Will indicate the type of the orbit
    def orbit_type(self, r=None, v=None, threshold=0.5):
        #Check if r and v are given 
        if r != None and v != None:
            energy = self.energy(r,v)
        elif self.s != []:
            #The orbit has been propagated
            energy = self.energy(self.s[0,:3],self.s[0,3:])
        else:
            raise Exception("🚀Sorry for the orbit shape to be determined the r and v must be given or the orbit must be first propagated🚀")

        #Rounding the value of energy
        energy = round(energy/(2*threshold))        #If the energy is in the range of energy = [-2.5 , 2.5] round it to zero -> change the 5 if neccesary

        #Determining the orbit type
        if energy < 0:
            return "ellipse"    #The code here can't set apart circular orbits from the elliptical ones -> Working on it⚒️

        elif energy == 0: 
            return "parabola"

        elif energy > 0 :
            return "hyperbolic"

    

class OrbitVisualizer():
    def colorGenerator(self, num):
        chars = '0123456789ABCDEF'
        return ['#'+''.join(random.sample(chars,6)) for i in range(num)]

    def simpleStatic(self, r, title="3D orbit around earth"):
        "Plotting the orbit in static form. No animation"
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

    def EarthStatic(self, r, title="3D earth orbit"):
        "Plotting the orbit and the earth with countries borders"

        # Get country borders from Natural Earth (GeoJSON)
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        geojson_data = requests.get(geojson_url).json()

        # Define Earth radius in kilometers
        earth_radius = 6371

        # Generate Earth's sphere
        theta, phi = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 25)
        theta, phi = np.meshgrid(theta, phi)

        x = earth_radius * np.cos(theta) * np.sin(phi)
        y = earth_radius * np.sin(theta) * np.sin(phi)
        z = earth_radius * np.cos(phi)

        # Generate orbit (e.g., a circular orbit around Earth)
        orbit_theta = np.linspace(0, 2*np.pi, 100)
        orbit_x = 1.5 * earth_radius * np.cos(orbit_theta)
        orbit_y = 1.5 * earth_radius * np.sin(orbit_theta)
        orbit_z = np.zeros_like(orbit_x)  # Keeping orbit in equatorial plane

        # Create figure
        fig = go.Figure()
        fig.update_layout(showlegend=False)


        # Add Earth
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, "lightblue"], [1, "lightblue"]], showscale=False))

        # Add orbit trajectory
        fig.add_trace(go.Scatter3d(x=r[:,0], y=r[:,1], z=r[:,2], mode="lines", 
                                line=dict(color="red", width=3), name="Orbit"))

        # Add country borders (approximation by plotting geojson points)
        for feature in geojson_data["features"]:
            coordinates = feature["geometry"]["coordinates"]
            for polygon in coordinates:
                lon, lat = np.array(polygon).T
                lat, lon = np.radians(lat), np.radians(lon)  # Convert degrees to radians

                # Convert lat/lon to 3D coordinates scaled to Earth's radius
                border_x = earth_radius * np.cos(lon) * np.cos(lat)
                border_y = earth_radius * np.sin(lon) * np.cos(lat)
                border_z = earth_radius * np.sin(lat)

                fig.add_trace(go.Scatter3d(x=border_x, y=border_y, z=border_z, mode="lines",
                                        line=dict(color="black", width=1), name="Borders"))

        # Customize view
        # Customize view
        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            title=title,  # Add title
            title_font=dict(size=20, color="white"),  # Customize title font
            
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),    
            )
        )

        fig.show()


    def SimpleDynamic(self, r, time, title="3D animation of orbit"):
        "Plotting the orbital motion with animation"

        # Create figure
        fig = go.Figure()

        # Define central sphere (Earth representation)
        num_points = 50
        theta, phi = np.meshgrid(np.linspace(0, np.pi, num_points), np.linspace(0, 2*np.pi, num_points))

        # Scale the sphere to Earth's actual radius (6371 km)
        sphere_radius = 6371
        x_sphere = sphere_radius * np.sin(theta) * np.cos(phi)
        y_sphere = sphere_radius * np.sin(theta) * np.sin(phi)
        z_sphere = sphere_radius * np.cos(theta)

        # Add Earth Sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False
        ))

        # Convert orbit to NumPy array for animation steps
        num_orbit_points = len(r)

        # Initialize empty orbit trace (animated later)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],  # Start with an empty orbit
            mode="lines",
            line=dict(color="white", width=2),
            name="Orbit"
        ))

        #The satellite
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],  # Start with an empty orbit
            mode="lines",
            line=dict(color="white", width=2),
            name="Orbit"
        ))

        # Create animation frames (Earth stays constant, orbit updates, time updates)
        frames = [
            go.Frame(
                data=[
                    go.Surface(
                        x=x_sphere, y=y_sphere, z=z_sphere,
                        colorscale=[[0, "blue"], [1, "blue"]],
                        showscale=False
                    ),  # Keep Earth in every frame!
                    go.Scatter3d(
                        x=r[:i+1, 0], y=r[:i+1, 1], z=r[:i+1, 2],  # Add orbit points
                        mode="lines",
                        line=dict(color="white", width=2)
                    ),
                    go.Scatter3d(
                        x=r[i:i+1, 0], y=r[i:i+1, 1], z=r[i:i+1, 2], 
                        mode = "markers",
                        line = dict(color="pink")
                    )
                ],
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Time: {time[i]/3600:10.2f}h",  # **Display current time step**
                        x=0.05, y=0.95,  # Position in top-left corner
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=20, color="white")
                    )]
                )
            ) for i in range(num_orbit_points)
        ]

        # Apply animation settings
        fig.frames = frames

        fig.update_layout(
            title=title,
            title_font=dict(size=24, color="white"),
            width=1200,
            height=1200,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False),
                yaxis=dict(showbackground=False, showgrid=False),
                zaxis=dict(showbackground=False, showgrid=False),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,  # Ensure button remains active
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]  
                )]
            )]
        )

        # Show plot
        fig.show()

    def EarthDynamic(self, r, time, title="3D animation of orbital motion around earth"):
        "Plotting the orbital motion with animation"

        # Get country borders from Natural Earth (GeoJSON)
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        geojson_data = requests.get(geojson_url).json()

        earth_radius = 6371


        # Create figure
        fig = go.Figure()

        # Define central sphere (Earth representation)
        num_points = 50
        theta, phi = np.meshgrid(np.linspace(0, np.pi, num_points), np.linspace(0, 2*np.pi, num_points))

        # Scale the sphere to Earth's actual radius (6371 km)
        sphere_radius = 6371
        x_sphere = sphere_radius * np.sin(theta) * np.cos(phi)
        y_sphere = sphere_radius * np.sin(theta) * np.sin(phi)
        z_sphere = sphere_radius * np.cos(theta)

        fig.update_layout(showlegend=False) #Not showing the legend

        # Add Earth Sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            showscale=False
        ))

        # Convert orbit to NumPy array for animation steps
        num_orbit_points = len(r)

        # Initialize empty orbit trace (animated later)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],  # Start with an empty orbit
            mode="lines",
            line=dict(color="red", width=2),
            name="Orbit"
        ))

        #The sattlite
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],  # Start with an empty orbit
            mode="lines",
            line=dict(color="white", width=2),
            name="Orbit"
        ))

        # Add Earth land and borders (approximate using lat/lon)
        for feature in geojson_data["features"]:
            coordinates = feature["geometry"]["coordinates"]
            for polygon in coordinates:
                lon, lat = np.array(polygon).T
                lat, lon = np.radians(lat), np.radians(lon)  # Convert degrees to radians

                # Convert lat/lon to 3D coordinates scaled to Earth's radius
                border_x = earth_radius * np.cos(lon) * np.cos(lat)
                border_y = earth_radius * np.sin(lon) * np.cos(lat)
                border_z = earth_radius * np.sin(lat)

                fig.add_trace(go.Scatter3d(x=border_x, y=border_y, z=border_z, mode="lines",
                                        line=dict(color="black", width=1), name="Borders"))


        # Create animation frames (Earth stays constant, orbit updates, time updates)
        frames = [
            go.Frame(
                data=[
                    go.Surface(
                        x=x_sphere, y=y_sphere, z=z_sphere,
                        colorscale=[[0, "lightblue"], [1, "lightblue"]],
                        showscale=False
                    ),  # Keep Earth in every frame!
                    go.Scatter3d(
                        x=r[:i+1, 0], y=r[:i+1, 1], z=r[:i+1, 2],  # Add orbit points
                        mode="lines",
                        line=dict(color="red", width=2)
                    ),
                    go.Scatter3d(
                        x=r[i:i+1, 0], y=r[i:i+1, 1], z=r[i:i+1, 2], 
                        mode = "markers",
                        line = dict(color="pink")
                    )
                ],
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Time: {time[i]/3600:10.2f}h",  # **Display current time step**
                        x=0.05, y=0.95,  # Position in top-left corner
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=20, color="white")
                    )]
                )
            ) for i in range(num_orbit_points)
        ]

        # Apply animation settings
        fig.frames = frames

        fig.update_layout(
            title=title,
            title_font=dict(size=24, color="white"),
            width=1200,
            height=1200,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False),
                yaxis=dict(showbackground=False, showgrid=False),
                zaxis=dict(showbackground=False, showgrid=False),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,  # Ensure button remains active
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]  
                )]
            )]
        )

        # Show plot
        fig.show()

    #The multiple visualizer
    def simpleStaticM(self, r, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
        "Plotting the orbit in static form. No animation"
        # Create figure
        fig = go.Figure()

        #Correcting the r format
        r = np.array(r) 

        if r.ndim == 2:                                 #if only a single orbit is provided fix the formatting
            r = r.reshape([1,r.shape[0],r.shape[1]])    #Reshaping the r

        #Number of orbits
        n = len(r)  

        #Check if the colors are provided otherwise generate the color set
        if not colors:
            colors = self.colorGenerator(n)

        #Check if the names(legends) are provided otherwise set all the values equal to "ORBIT" and disable the legend
        if names == []:
            names = ["Orbit" for i in range(n)]
            fig.update_layout(showlegend=False)   #Not showing the legend

        elif type(names) is str:    #If the names has only a single value make i a list
            names = [names for i in range(n)]
        
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
        
        for ind, orbit in enumerate(r):
            # Preserve all existing elements, including the orbit
            fig.add_trace(go.Scatter3d(
                x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2],
                mode="lines",
                line=dict(color=colors[ind], width=2),
                name=names[ind]
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
                xaxis=dict(range=[limits[0,1], limits[0,0]], title='X (km)', showbackground=False, showgrid=False),
                yaxis=dict(range=[limits[1,1], limits[1,0]], title='Y (km)',showbackground=False, showgrid=False),
                zaxis=dict(range=[limits[2,1], limits[2,0]], title='Z (km)',showbackground=False, showgrid=False),
                aspectmode='manual',  # Important for 3D plots
                aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio
            ),
            legend=dict(
                title=dict(text='<i>ORBITAL ELEMENTS</i>', font=dict(size=20)),
                font=dict(
                    size=30,
                    color='white'
                )
            ),
        )

        # Show plot
        fig.show()

    def EarthStaticM(self, r, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
        "Plotting the orbit and the earth with countries borders"
        # Create figure
        fig = go.Figure()

        #Handeling the inputs and their formating
         #Correcting the r format
        r = np.array(r) 

        if r.ndim == 2:                                 #if only a single orbit is provided fix the formatting
            r = r.reshape([1,r.shape[0],r.shape[1]])    #Reshaping the r

        #Number of orbits
        n = len(r)  

        #Check if the colors are provided otherwise generate the color set
        if not colors:
            colors = self.colorGenerator(n)

        #Check if the names(legends) are provided otherwise set all the values equal to "ORBIT" and disable the legend
        if names == []:
            names = ["Orbit" for i in range(n)]
            fig.update_layout(showlegend=False)   #Not showing the legend

        elif type(names) is str:    #If the names has only a single value make i a list
            names = [names for i in range(n)]

        # Get country borders from Natural Earth (GeoJSON)
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        geojson_data = requests.get(geojson_url).json()


        # Define central sphere (Earth representation)
        # Scale the sphere to Earth's actual radius (6371 km)
        num_points = 50
        sphere_radius = 6371
    
        #Generating the sphere's coordinates
        x_sphere, y_sphere, z_sphere = self.__sphereProvider(num_points, sphere_radius)

        # Add Earth
        fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, colorscale=[[0, "lightblue"], [1, "lightblue"]], showscale=False))

        # Add orbits trajectory
        for ind, orbit in enumerate(r):
            # Preserve all existing elements, including the orbit
            fig.add_trace(go.Scatter3d(
                x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2],
                mode="lines",
                line=dict(color=colors[ind], width=2),
                name=names[ind]
            ))


        # Add country borders (approximation by plotting geojson points)
        for feature in geojson_data["features"]:
            coordinates = feature["geometry"]["coordinates"]
            for polygon in coordinates:
                lon, lat = np.array(polygon).T
                lat, lon = np.radians(lat), np.radians(lon)  # Convert degrees to radians

                # Convert lat/lon to 3D coordinates scaled to Earth's radius
                border_x = sphere_radius * np.cos(lon) * np.cos(lat)
                border_y = sphere_radius * np.sin(lon) * np.cos(lat)
                border_z = sphere_radius * np.sin(lat)

                fig.add_trace(go.Scatter3d(x=border_x, y=border_y, z=border_z, mode="lines",
                                        line=dict(color="black", width=1), showlegend=False))


        # Customize view
        fig.update_layout(
            title=title,  # Add title here
            title_font=dict(size=24, color="white"),  # Customize font size and color
            width=1200,
            height=1200,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(range=[limits[0,1], limits[0,0]], title='X (km)', showbackground=False, showgrid=False),
                yaxis=dict(range=[limits[1,1], limits[1,0]], title='Y (km)',showbackground=False, showgrid=False),
                zaxis=dict(range=[limits[2,1], limits[2,0]], title='Z (km)',showbackground=False, showgrid=False),
                aspectmode='manual',  # Important for 3D plots
                aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio
            ),
            legend=dict(
                title=dict(text='<i>ORBITAL ELEMENTS</i>', font=dict(size=20)),
                font=dict(
                    size=30,
                    color='white'
                )
            ),
        )

        fig.show()



    def SimpleDynamicM(self, r, time, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
        "Plotting the orbital motion with animation"

        # Create figure
        fig = go.Figure()

        #Handeling the inputs and their formating
        #Correcting the r format
        r = np.array(r) 

        if r.ndim == 2:                                 #if only a single orbit is provided fix the formatting
            r = r.reshape([1,r.shape[0],r.shape[1]])    #Reshaping the r

        #Number of orbits
        n = len(r)  

        #Check if the colors are provided otherwise generate the color set
        if not colors:
            colors = self.colorGenerator(n)

        #Check if the names(legends) are provided otherwise set all the values equal to "ORBIT" and disable the legend
        if names == []:
            names = ["Orbit" for i in range(n)]
            fig.update_layout(showlegend=False)   #Not showing the legend

        elif type(names) is str:    #If the names has only a single value make i a list
            names = [names for i in range(n)]


        # Define central sphere (Earth representation)
        # Scale the sphere to Earth's actual radius (6371 km)
        num_points = 50
        sphere_radius = 6371
    
        #Generating the sphere's coordinates
        x_sphere, y_sphere, z_sphere = self.__sphereProvider(num_points, sphere_radius)

        # Add Earth Sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False
        ))

        # Convert orbit to NumPy array for animation steps
        num_orbit_points = r.shape[1]

        # Initialize empty orbits trace (animated later)
        for i in range(n):
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],  # Start with an empty orbit
                mode="lines",
                line=dict(color="white", width=2),
                name= names[i]
            ))

        #The sattlite empty trace
        for i in range(n):
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],  # Start with an empty orbit
                mode="lines",
                line=dict(color="white", width=2),
                name= names[i]
            ))


        # Create animation frames (Earth stays constant, orbit updates, time updates)
        frames = [
            go.Frame(
                data=self.__sceneProviderSimple(x_sphere, y_sphere, z_sphere, r, i, n, colors),
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Time: {time[i]/3600:10.2f}h",  # **Display current time step**
                        x=0.05, y=0.95,  # Position in top-left corner
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=20, color="white")
                    )]
                )
            ) for i in range(num_orbit_points)
        ]

        # Apply animation settings
        fig.frames = frames

        fig.update_layout(
            title=title,
            title_font=dict(size=24, color="white"),
            width=1200,
            height=1200,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
            xaxis=dict(range=[limits[0,1], limits[0,0]], title='X (km)', showbackground=False, showgrid=False),
            yaxis=dict(range=[limits[1,1], limits[1,0]], title='Y (km)',showbackground=False, showgrid=False),
            zaxis=dict(range=[limits[2,1], limits[2,0]], title='Z (km)',showbackground=False, showgrid=False),
            aspectmode='manual',  # Important for 3D plots
            aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio
            ),
            legend=dict(
                title=dict(text='<i>ORBITAL ELEMENTS</i>', font=dict(size=20)),
                font=dict(
                    size=30,
                    color='white'
                )
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,  # Ensure button remains active
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]  
                )]
            )]
        )

        # Show plot
        fig.show()


    def EarthDynamicM(self, r, time, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
        "Plotting the orbital motion with animation"

        # Get country borders from Natural Earth (GeoJSON)
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        geojson_data = requests.get(geojson_url).json()

        # Create figure
        fig = go.Figure()

         #Handeling the inputs and their formating
        #Correcting the r format
        r = np.array(r) 

        if r.ndim == 2:                                 #if only a single orbit is provided fix the formatting
            r = r.reshape([1,r.shape[0],r.shape[1]])    #Reshaping the r

        #Number of orbits
        n = len(r)  

        #Check if the colors are provided otherwise generate the color set
        if not colors:
            colors = self.colorGenerator(n)

        #Check if the names(legends) are provided otherwise set all the values equal to "ORBIT" and disable the legend
        if names == []:
            names = ["Orbit" for i in range(n)]
            fig.update_layout(showlegend=False)   #Not showing the legend

        elif type(names) is str:    #If the names has only a single value make i a list
            names = [names for i in range(n)]


        # Define central sphere (Earth representation) with the actual radius (6371km)
        num_points = 50
        sphere_radius = 6371
        
        #Generating the sphere's coordinates
        x_sphere, y_sphere, z_sphere = self.__sphereProvider(num_points, sphere_radius)

        # Add Earth Sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            showscale=False
        ))

        # Convert orbit to NumPy array for animation steps
        num_orbit_points = r.shape[1]

        # Initialize empty orbits trace (animated later)
        for i in range(n):
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],  # Start with an empty orbit
                mode="lines",
                line=dict(color="white", width=2),
                name= names[i]
            ))

        #The sattlite empty trace
        for i in range(n):
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],  # Start with an empty orbit
                mode="lines",
                line=dict(color="white", width=2),
                name= names[i]
            ))

        # Add Earth land and borders (approximate using lat/lon)
        for feature in geojson_data["features"]:
            coordinates = feature["geometry"]["coordinates"]
            for polygon in coordinates:
                lon, lat = np.array(polygon).T
                lat, lon = np.radians(lat), np.radians(lon)  # Convert degrees to radians

                # Convert lat/lon to 3D coordinates scaled to Earth's radius
                border_x = sphere_radius * np.cos(lon) * np.cos(lat)
                border_y = sphere_radius * np.sin(lon) * np.cos(lat)
                border_z = sphere_radius * np.sin(lat)

                fig.add_trace(go.Scatter3d(x=border_x, y=border_y, z=border_z, mode="lines",
                                        line=dict(color="black", width=1), showlegend=False))


        # Create animation frames (Earth stays constant, orbit updates, time updates)
        frames = [
            go.Frame(
                data= self.__sceneProviderSimple(x_sphere, y_sphere, z_sphere, r, i, n, colors, "lightblue"), 
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Time: {time[i]/3600:10.2f}h",  # **Display current time step**
                        x=0.05, y=0.95,  # Position in top-left corner
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=20, color="white")
                    )]
                )
            ) for i in range(num_orbit_points)
        ]

        # Apply animation settings
        fig.frames = frames

        fig.update_layout(
            title=title,
            title_font=dict(size=24, color="white"),
            width=1200,
            height=1200,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(range=[limits[0,1], limits[0,0]], title='X (km)', showbackground=False, showgrid=False),
                yaxis=dict(range=[limits[1,1], limits[1,0]], title='Y (km)',showbackground=False, showgrid=False),
                zaxis=dict(range=[limits[2,1], limits[2,0]], title='Z (km)',showbackground=False, showgrid=False),
                aspectmode='manual',  # Important for 3D plots
                aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio
            ),
            legend=dict(
                title=dict(text='<i>ORBITAL ELEMENTS</i>', font=dict(size=20)),
                font=dict(
                    size=30,
                    color='white'
                )
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,  # Ensure button remains active
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]  
                )]
            )]
        )

        # Show plot
        fig.show()


    #The function will provide the coordinates of sphere 
    def __sphereProvider(self, num_points=50, sphere_radius=6371 ):
        "Simple sphere"

        # Define central sphere (Earth representation)
        theta, phi = np.meshgrid(np.linspace(0, np.pi, num_points), np.linspace(0, 2*np.pi, num_points))

        # Scale the sphere to Earth's actual radius (6371 km)
        x_sphere = sphere_radius * np.sin(theta) * np.cos(phi)
        y_sphere = sphere_radius * np.sin(theta) * np.sin(phi)
        z_sphere = sphere_radius * np.cos(theta)

        return x_sphere, y_sphere, z_sphere

    #i is the time 
    #n is the number of orbits/satellites
    def __sceneProviderSimple(self,x_sphere, y_sphere, z_sphere, r, i, n, colors, earth_color="blue"):
        "Returning the data for each frame"
        data = []

        #Adding the earth
        data.append(
            go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale=[[0, earth_color], [1, earth_color]],
                showscale=False
            ),  # Keep Earth in every frame!    
        )

        #Now adding the trajectories
        for k in range(n):
            data.append(
                go.Scatter3d(
                            x=r[k, :i+1, 0], y=r[k ,:i+1, 1], z=r[k ,:i+1, 2],  # Add orbit points
                            mode="lines",
                            line=dict(color=colors[k], width=2)
                        ),
            )

        #Now adding the satellites 
        for k in range(n):
            data.append(
                go.Scatter3d(
                            x=r[k,i:i+1, 0], y=r[k,i:i+1, 1], z=r[k,i:i+1, 2], 
                            mode = "markers",
                            line = dict(color=colors[k])
                        )
            )

        return data


