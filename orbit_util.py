#Dependencies
import numpy as np 
from scipy.integrate import odeint, quad
from scipy.optimize  import fsolve
import plotly.graph_objects as go
import requests
import random
from math import sin, cos, pi, atan2, sqrt, atan, tan, acos


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

    #delta_true_anomaly is assumed to be in degrees if provided in radians set the is_radians to true
    def perifocal_calculator(self, r0, v0, delta_true_anomaly, is_radians = False):
        "Prediciting the r and v of the sattelite's position vecotr and velocity vector after a specific amount of true anomaly"
        #Converting the delta_true_anomaly to radians to be used with math.sin() and math.cos()
        if not is_radians:
            tr_anomaly = delta_true_anomaly * pi / 180

        #Converting the r0 and v0 to array to perform vector calculus using np
        r0 = np.array(r0)
        v0 = np.array(v0)

        #Finding the angular momentum of the orbit
        _ , h = self.specific_angular_momentum(r0, v0) 

        #Finding the Vro
        v_r0 = np.dot(v0 , r0) / np.linalg.norm(r0)

        #Calculating the r0_mag
        r0_mag = np.linalg.norm(r0)

        #Calculating the r at the delta_true_anomaly
        r = (h ** 2 / self.mu) * 1/(1 + (h**2/(self.mu*r0_mag) - 1 )*cos(tr_anomaly) - (h*v_r0/self.mu) * sin(tr_anomaly))

        #Calculating the lagrange coefficents
        f = 1 - (self.mu * r / h**2) * (1 - cos(tr_anomaly))                                                                            #[Dimensionless]
        g = r * r0_mag * sin(tr_anomaly) / h                                                                                               #[1/s]
        f_dot = self.mu * (1 - cos(tr_anomaly)) / (h * sin(tr_anomaly)) * ( (self.mu / h**2) * (1-cos(tr_anomaly)) - 1 / r0_mag - 1/r )    #[1/s]
        g_dot = 1 - self.mu * r0_mag * (1 - cos(tr_anomaly)) / h**2                                                                     #[Dimensionless]

        #Calculating the postition and the velocity vector 
        r_vec = f * r0 + g * v0 
        v_vec = f_dot * r0 + g_dot * v0

        #---Calculating if the coordinates frame is the actual perifocal or the rotated version of it 
        #---Finding the inital_true_anomaly and the eccentricity form the 2Body orbit equation and Vr formula
        esin = v_r0 * h / self.mu
        ecos = h**2 / (self.mu * r0_mag) -1 

        #Finding the inital_true_anomaly
        inital_true_anomaly = atan2(esin, ecos)

        #Finding the virtual_anomaly
        inital_virtual_anomaly = atan2(r0[1], r0[0]) 

        #Finding the angle which the coordinate system has been rotated about the Z-axis of the perifocal frame
        angle_of_rotation = inital_true_anomaly - inital_virtual_anomaly

        #Converting the angle of rotation to degree
        angle_of_rotation = angle_of_rotation * 180 / pi

        return r_vec, v_vec, angle_of_rotation


    #If the 2Body assumption without disturbance and thruster's interfearance are assume the h = constant
    def specific_angular_momentum(self, r, v):
        "Calculating  the specific angular momentum of the sattlite in a specific location"

        #Convert to np.arr
        r = np.array(r)
        v = np.array(v)

        #Calculating the h vector
        h_vec = np.cross(r, v, axisa=-1, axisb=-1, axisc=-1, axis=None)

        #Magnitude
        h_mag = np.linalg.norm(h_vec)

        return h_vec, h_mag

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
        
    
    #✅Calculating the eccentricity vector and magnitude
    def eccentricity(self, r, v):
        """
        Calculating the e vector and magnitude using r and v
        Parameters:\n
            r: (np.array([rx, ry, rz])) position vector in ECI in [km]
            v: (np.array([vx, vy, vz])) velocity vector in ECI in [km]

        Returns:\n
            e_vec: (np.array([ex, ey, ez])) Eccentricity vector
            e_mag: (float) Magnitude of the eccentricity vector
        """

        #Conversion to array
        r = np.array(r)
        v = np.array(v)

        #Magnitude of r and v
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        #Radial velocity
        v_r = np.dot(r, v) / np.linalg.norm(r)

        #Using the orbit equation(differential equation)
        e_vec = (1/self.mu)*((v_mag**2 - (self.mu/r_mag))*r - r_mag * v_r * v)

        #Magnitude of e
        e_mag = np.linalg.norm(e_vec)

        return e_vec, e_mag

    
    #Finding the change in true anomaly with time
    def time_since_perigee(self ,true_anomaly, r=None, v=None, h=None, e=None, degree_mode=False):
        """
        Calculates the time required to get from the preigee to the specified true anomaly.\n
        Parameters:\n
            If h and e are known provide them otherwise provide r and v (true_anomaly must be in radians)\n
            degree_mode : (bool) Set equal to true and the true_anomaly will be assumed to be in degrees \n
        Returns:\n
            time: Seconds\n 
            error : Estimated absolute error

        """

        #Check to see if the degree mode is beening used
        if degree_mode:
            true_anomaly = true_anomaly * pi / 180

        #Calcualting the neccessary variables(If not provided)
        if h == None:
            h = self.specific_angular_momentum(r,v)
        if e == None:
            _ , e = self.eccentricity(r,v) 

        #Integrating the general formula -> Is valid for all orbit types
        f_t = lambda theta: (h**3 / self.mu ** 2) * 1/(1+e * np.cos(theta))**2

        time, err = quad(f_t, 0, true_anomaly)

        return time ,err
    
    #Calculating the true anomaly of the satellite from the time since preigee
    def true_anomaly_from_time(self, time, h=None ,e=None ,r=None , v=None, degree_mode=False):
        """
        Calculates the true anomaly of satellite from the time since preigee.\n
        Parameters:\n
            If h and e are known provide them otherwise provide r and v (time is the time from preigee in seconds)\n
            degree_mode : (bool) Set equal to true and the true_anomaly, eccentric_anomaly and mean_anomaly will be in degrees \n
        Returns:\n
            true_anomaly: (float) in radians
            eccentric_anomaly: (flot)
            mean_anomaly: (float) in radians
        """

        #Calcualting the neccessary variables(If not provided)
        if h == None:
            h = self.specific_angular_momentum(r,v)
        if e == None:
            _ , e = self.eccentricity(r,v) 
    
        #Calculating the period of the orbit
        T = self.period(h, e)

        #Calculating the mean_anomaly
        M_e = 2 * pi * time / T

        #kepler's equation
        kep_E = lambda E : E - e * sin(E) - M_e

        #Inital guess for the solution of kepler's equation
        inital_guess = M_e + e/2 if M_e < pi else M_e - e/2

        #Solving the kepler's equation and finding the eccentric anomaly
        E = fsolve(kep_E, inital_guess)[0]

        #Finding the true_anomaly
        true_anomaly = 2 * atan( sqrt((1+e)/(1-e)) * tan(E/2))

        #Adding 2pi to the true_anomaly if it is negative
        true_anomaly = true_anomaly if true_anomaly > 0 else true_anomaly + 2 * pi

        #Check if the degree mode is enabled
        if degree_mode:
            true_anomaly = true_anomaly * 180 / pi
            E = E * 180 / pi
            M_e = M_e * 180 / pi

        return true_anomaly, E, M_e


    #✅Calculates the period of an orbit
    def period(self, h, e):
        "Calculates the priod of an orbit from true specific angular momentum and eccentricity"
        
        #For open orbits T=inf
        if e >= 1:
            return float('inf')

        #For closed orbits:
        #Semi-major axis
        a = self.semi_major_axis(h,e)

        #Calculating the period
        T =  (2 * pi/sqrt(self.mu)) * self.semi_major_axis(h,e)**1.5

        return T


    #✅Calculating the semi_major_axis of the orbit "a"
    def semi_major_axis(self, h, e):
        "Calculates the semi major axis of the orbit for any orbit-type"

        #Parabolic orbit -> a:undefined
        if e == 1:
            raise Exception("🚀Sorry for the parabolic orbits the semi-major axis is undefined🚀")
        
        #For any other orbit
        a =  ((h ** 2)/(self.mu)) * (1/abs( 1 - e**2))

        return a 

    #Converting the Cartesian element to classical orbital elements
    def cartesian_to_keplerain(self, r, v, degree_mode=False):
        '''
        Converting the cartesian to classical orbital elements(Position vector and velocity vector) for a single instance
        Parameters:\n
            r: (np.array([rx, ry, rz])) position vector in ECI in [km]
            v: (np.array([vx, vy, vz])) velocity vector in ECI in [km]
            degree_mode: (bool) if equal to true the i, w, RAAN and theta should be given in degrees

        Returns:\n
            e: (float) Eccentricity 
            h: (float) Specific angular momentum
            theta : (float) True anomaly in radians
            i : (float) inclination in radians
            w : (float) Argument of preiapsis in radians 
            RAAN : (float) right ascension of ascending node in radians
            
        '''

        #Converting the r and v to array
        r = np.array(r)
        v = np.array(v)

        #Calculating the magnitude of r and v(speed)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        #Determining the radial vecloicty
        v_r = np.dot(r , v)         

        #💫Determining the specific angular momentum
        h_vector, h_mag = self.specific_angular_momentum(r, v)

        #💫Determining the eccentricity vector and magnitude               
        e_vec, e = self.eccentricity(r,v)  

        return e_vec, e
        #💫Determining the inclination
        # i = acos(h_vector[2]/h_mag)

        # #Determing the node line vector and magnitude
        # N_vec = np.cross([0,0,1] , h_vector)
        # N_mag = np.linalg.norm(N_vec)

        # #💫Determining the RAAN
        # RAAN = acos(N_vec[0]/N_mag) if N_vec[1] >= 0 else 2*pi - acos(N_vec[0]/N_mag) 

        # #💫Determining the argument of preiapsis w
        # w = acos(np.dot(N_vec, e_vec)/(N_mag * e)) if e_vec[2] >=0 else 2*pi - acos(np.dot(N_vec, e_vec)/(N_mag * e))

        # #💫Determining the true anomaly
        # theta = acos(np.dot(e_vec,r)/(e * r_mag)) if v_r >= 0 else 2*pi - acos(np.dot(e_vec,r)/(e * r_mag))

        # if degree_mode:
        #     theta = theta * 180 / pi
        #     i = i * 180 / pi 
        #     RAAN = RAAN * 180 / pi
        #     w = w * 180 / pi

        # #Returning the elements
        # return e , h_mag, theta, i, w, RAAN 


    #Converting the classical orbital element to state sapce 
    def keplerian_to_cartesian(self, e, h, theta, i, RAAN, w, degree_mode=False):
        '''
        Converting the classical orbital elements to cartesian elements(Position vector and velocity vector) for a single instance
        Parameters:\n
            e: (float) Eccentricity 
            h: (float) Specific angular momentum
            theta : (float) True anomaly in radians
            i : (float) inclination in radians
            w : (float) Argument of preiapsis in radians 
            degree_mode: (bool) if equal to true the i, w, RAAN and theta should be given in degrees

        Returns:\n
            r: (np.array([rx, ry, rz])) position vector in ECI in [km]
            v: (np.array([vx, vy, vz])) velocity vector in ECI in [km]
        '''

        #Check if the angles are in radians or not
        if degree_mode:
            theta = theta * pi / 180
            w = w * pi / 180
            i = i * pi / 180
            RAAN = RAAN * pi / 180

        #Calculating the r and v in perifocal frame
        r_x_perifocal = (h**2/self.mu) * (1/(1+e*cos(theta))) * cos(theta)
        r_y_perifocal = (h**2/self.mu) * (1/(1+e*cos(theta))) * sin(theta)
        r_z_perifocal = 0                                                  #Becuase of how the perifocal frame is definded

        v_x_perifocal = (self.mu/h) * (-1) * sin(theta)
        v_y_perifocal = (self.mu/h) * (e + cos(theta))
        v_z_perifocal = 0                                                  #Becuase of how the perifocal frame is definded

        #setting up the vector
        r_perifocal = np.array([r_x_perifocal, r_y_perifocal, r_z_perifocal])
        v_perifocal = np.array([v_x_perifocal, v_y_perifocal, v_z_perifocal])
        

        #Perifcoal --> ECI transfer matric (Direction cosine matrix)
        DCM_perifocal_to_ECI = np.array([
            [-sin(RAAN)*cos(i)*sin(w)+cos(RAAN)*cos(w), -sin(RAAN)*cos(i)*cos(w)-cos(RAAN)*sin(w), sin(RAAN)*sin(i)],
            [cos(RAAN)*cos(i)*sin(w)+sin(RAAN)*cos(w), cos(RAAN)*cos(i)*cos(w)-sin(RAAN)*sin(w), -cos(RAAN)*sin(i)],
            [sin(i)*sin(w) , sin(i)*cos(w), cos(i)]
            ])
        
        #Coordiante transformation for position and velocity vector from perifocal to ECI(earth centerd interia)
        r_ECI = np.matmul(DCM_perifocal_to_ECI, r_perifocal)
        v_ECI = np.matmul(DCM_perifocal_to_ECI, v_perifocal)

        #Returning the r and v vector in the ECI frame (Coordinate)
        return r_ECI , v_ECI
    

    

class OrbitVisualizer():
    def colorGenerator(self, num):
        chars = '0123456789ABCDEF'
        return ['#'+''.join(random.sample(chars,6)) for i in range(num)]

    #The multiple visualizer
    def simpleStatic(self, r, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
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

    def EarthStatic(self, r, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
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
                #Handelling the MultiPolygon type 
                if feature['geometry']['type'] == "MultiPolygon":
                    polygon = np.array(polygon)
                    polygon = polygon.reshape(polygon.shape[1], 2)

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



    def SimpleDynamic(self, r, time, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
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


    def EarthDynamic(self, r, time, colors=False, title="3D orbit around earth", names=[], limits=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
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
                #Handeling MultiPolygon type
                if feature['geometry']['type'] == "MultiPolygon":
                    polygon = np.array(polygon)
                    polygon = polygon.reshape(polygon.shape[1], 2)

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


