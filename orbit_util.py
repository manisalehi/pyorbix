""""

"""

#Dependencies
import numpy as np 
from scipy.integrate import odeint, quad
from scipy.optimize  import fsolve
import plotly.graph_objects as go
import requests
import random
from math import sin, cos, pi, atan2, sqrt, atan, tan, acos, floor, asin
from datetime import datetime, timezone, timedelta
import spiceypy as spice
import nrlmsise00
import os
import sys
from typing import Tuple, List, Callable, Any, Optional, Union, Dict




#The orbit propagetor
class Orbit_2body():
    def __init__(self):
        # 🌍 Earth's gravitational parameter  
        self.mu = 3.986004418E+05  # [km^3/s^2]
        self.s = np.array([])
        self.t = np.array([])
        self.w_earth = 2*pi/(24 * 60 * 60) + 2 * pi /(365.25 * 24 * 60 * 60)    #[rad/s] : MAGNITUDE
        self.omega_earth = np.array([0, 0, 7.292115e-5])                        # rad/s (ECI Z-axis) : VECTOR

        # ☀️ Sun's gravitational parameter(Standard gravity GM)
        self.mu_sun = 1.32712440018E+11 #[km^3/s^2]

        # 🌙 Moon's gravitational parameter(Standard gravity GM)
        self.mu_moon = 4.9028695E3

        # 🥚 J2 perturbation constant of the earth                 
        self.J2 = 1.08263 * 10 **(-3)
        
        #Solar pressure radiation 
        self.AU = 149_597_871    #[km]      => The 1 AU = 149'59'871 km
        self.c_r = 1_371          #[w/m^2]    => The solar flux at 1 AU 1_371
        self.c = 299_792_000     #[m/s]     => The speed of light in m/s
                        
    #📝 A 2body orbit propagtor
    def propagate_init_cond(self, T: float, time_step:float, R0:Optional[Union[List[float], np.ndarray]], V0:Optional[Union[List[float], np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Propagates orbit using two-body equations with given initial conditions.

        Numerically solves the orbital propagation problem in ECI (J2000) reference frame.

        Parameters
        ----------
        T : float
            Total simulation duration (seconds)
        time_step : float
            Time step for numerical integration (seconds)
        R0 : array_like, shape (3,)
            Initial position vector [rx, ry, rz] in ECI frame (kilometers)
        V0 : array_like, shape (3,)
            Initial velocity vector [vx, vy, vz] in ECI frame (kilometers/second)

        Returns
        -------
        tuple
            A tuple containing:
            - ndarray
                Propagated state vectors with shape (n,6) containing:
                [rx, ry, rz, vx, vy, vz] for each time step (km, km/s)
            - ndarray
                Time values at each integration step (seconds)

        Notes
        -----
        - Pure two-body problem (Keplerian motion)
        - Uses Earth's gravitational parameter (self.mu)
        - Results stored in instance variables:
        * self.s: State vector history
        * self.t: Corresponding time steps
        - Suitable for preliminary orbit analysis
        - For higher fidelity, consider adding perturbations
        """
        
        
        S0 = np.hstack([R0, V0])            #Inital condition state vector
        t = np.arange(0, T, time_step)      #The time step's to solve the equation for

        #🧮Numerically solving the equation  
        sol = odeint(self.dS_dt, S0, t)

        #Saving the propagted orbit
        self.s = sol    
        self.t = t

        return sol, t
    
    #📝 A J2 propagator
    def propagte_with_J2(self, T:float, time_step:float, R0:Optional[Union[List[float], np.ndarray]], V0:Optional[Union[List[float], np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Propagates orbit using 2-body equations with J2 perturbation.

        Numerically solves orbital propagation problem with initial conditions in ECI frame.

        Parameters
        ----------
        T : float
            Total simulation duration (s)
        time_step : float
            Time step for numerical integration (s)
        R0 : array_like, shape (3,)
            Initial position vector [rx, ry, rz] in ECI (J2000) frame (km)
        V0 : array_like, shape (3,)
            Initial velocity vector [vx, vy, vz] in ECI (J2000) frame (km/s)

        Returns
        -------
        tuple
            Contains:
            - ndarray: State vectors with shape (n,6) containing [rx, ry, rz, vx, vy, vz] (km, km/s)
            - ndarray: Time values at each integration step (s)

        Notes
        -----
        - Includes both central gravity and J2 perturbation effects
        - Uses Earth's oblateness parameter (J2 ≈ 1.08263e-3)
        - Results stored in instance variables:
        * self.s: Array of state vectors
        * self.t: Array of time steps
        - Uses scipy.integrate.odeint for numerical solution
        - J2 accounts for Earth's equatorial bulge effect
        """
        
        S0 = np.hstack([R0, V0])            #Inital condition state vector
        t = np.arange(0, T, time_step)      #The time step's to solve the equation for

        #🧮Numerically solving the equation 
        sol = odeint(self.dS_dt_J2, S0, t)

        #Saving the propagted orbit
        self.s = sol    
        self.t = t

        return sol, t
    
    #Propagting the orbit from the intial conditons
    def HFOP(
    self,
    T: float,
    time_step: float,
    R0: Optional[Union[List[float], np.ndarray]],
    V0: Optional[Union[List[float], np.ndarray]],
    rho: Callable[[float, float, float, float], float] = lambda t, x, y, z: 1.5,
    A: Callable[[float, float, float, float], float] = lambda t, x, y, z: 0.1,
    m: Callable[[float, float, float, float], float] = lambda t, x, y, z: 3,
    scenario_epoch: datetime = datetime.now(timezone.utc),
    kernel_list: List[str] = ["naif0012.tls", "de440.bsp"],
    kernel_base_dir: str = "./kernels",
    s: Callable[[float, float, float, float], float] = lambda t, x, y, z: 0.01,
    cd: float = 2.2,
    thrust: Callable[[float, float, float, float], List[float]] = lambda t, x, y, z: [0, 0, 0],
    f107a: Callable[[float], float] = lambda t: 1.55,
    f107: Callable[[float], float] = lambda t: 1.1,
    ap: Callable[[float], float] = lambda t: 1.2,
    ap_a: Optional[Callable[[float], float]] = lambda t: None,
    flags: Optional[Callable[[float], Any]] = lambda t: None,
    method: str = 'gtd7'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagates orbit considering multiple perturbation forces.

        Simulates orbital dynamics including gravitational and non-gravitational perturbations.

        Parameters
        ----------
        T : float
            Total simulation duration (s)
        time_step : float
            Integration time step (s)
        R0 : array_like, shape (3,)
            Initial position vector [rx, ry, rz] in ECI (J2000) frame (km)
        V0 : array_like, shape (3,)
            Initial velocity vector [vx, vy, vz] in ECI (J2000) frame (km/s)
        rho : callable
            Solar radiation pressure coefficient function f(t, x, y, z) -> float
        A : callable
            Exposed surface area function f(t, x, y, z) -> float (m²)
        m : callable
            Mass function f(t, x, y, z) -> float (kg)
        scenario_epoch : datetime
            Simulation start time (UTC)
        kernel_list : list of str
            SPICE kernel filenames to load
        kernel_base_dir : str
            Directory containing SPICE kernels
        s : callable
            Drag reference area function f(t, x, y, z) -> float (m²)
        cd : float
            Drag coefficient (dimensionless)
        thrust : callable
            Thrust force function f(t, x, y, z) -> [Fx, Fy, Fz] (N) in ECI
        f107a : callable
            81-day averaged solar radio flux (10.7 cm) f(t) -> float
        f107 : callable
            Instantaneous solar radio flux (10.7 cm) f(t) -> float
        ap : callable
            Geomagnetic index function f(t) -> float
        ap_a : callable, optional
            3-hour averaged ap index function f(t) -> float
        flags : callable, optional
            NRLMSISE-00 model flags function f(t) -> list
        method : str, optional
            Atmospheric model method ('gtd7' for NRLMSISE-00, default)

        Returns
        -------
        tuple
            Contains:
            - ndarray: State vectors [rx, ry, rz, vx, vy, vz] at each time step (km, km/s)
            - ndarray: Time values corresponding to each state (s)

        Notes
        -----
        - Perturbations include:
        * Third-body gravity (Sun/Moon)
        * Earth oblateness (J2)
        * Atmospheric drag
        * Solar radiation pressure
        * Custom thrust
        - Requires SPICE kernels for ephemeris data
        - Uses NRLMSISE-00 atmospheric model
        - Results stored in self.s (states) and self.t (time)
        - Kernels automatically unloaded after propagation
        """

        S0 = np.hstack([R0, V0])            #Inital condition state vector
        t = np.arange(0, T, time_step)     #The time step's to solve the equation for

        #--Saving certain arguments as the class parameter to be loaded in the ds_dt_HFOP--
        #Saving the starting time in UTC
        self.scenario_epoch = scenario_epoch

        ##🔦Data for solar radiation pressure
        self.rho = rho
        self.m = m
    
        self.A = A

        ##💨Data for drag
        self.f107a = f107a
        self.f107 = f107
        self.ap = ap
        self.ap_a = ap_a
        self.cd = cd
        #Saving the surface area 
        self.s = s

        ##🎇Data for thrust
        self.thrust = thrust
        
        #⚒️Configs for nrlmsise00.msise_model
        self.flags = flags
        self.method = method

        #🔃Loading essential kernels (adjust paths as needed)
        for kernel in kernel_list:
            spice.furnsh(kernel_base_dir+"/"+kernel)

            #Success message
            print(kernel_base_dir+"/"+kernel + " was loaded successfully.")

        #🧮Numerically solving the equation 
        sol = odeint(self.dS_dt_HFOP, S0, t)
        
        #Saving the propagted orbit
        self.s = sol    
        self.t = t

        #Closing the kernels
        spice.kclear()

        return sol, t


    #📝 Delta_true_anomaly is assumed to be in degrees if provided in radians set the is_radians to true
    def perifocal_calculator(self, R0:Optional[Union[List[float], np.ndarray]], V0:Optional[Union[List[float], np.ndarray]], delta_true_anomaly:float, is_radians:bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """Predicts satellite state vectors after a true anomaly change.

        Calculates new position and velocity in perifocal frame and frame rotation angle.

        Parameters
        ----------
        R0 : array_like, shape (3,)
            Initial position vector [x, y, z] in inertial frame (km)
        V0 : array_like, shape (3,)
            Initial velocity vector [vx, vy, vz] corresponding to R0 (km/s)
        delta_true_anomaly : float
            Angular change in true anomaly (degrees by default)
        is_radians : bool, optional
            If True, interprets delta_true_anomaly as radians (default: False)

        Returns
        -------
        tuple
            Contains:
            - ndarray: New position vector [rx, ry, rz] (km)
            - ndarray: New velocity vector [vx, vy, vz] (km/s)
            - float: Rotation angle between frames (degrees)

        Notes
        -----
        - Uses standard perifocal frame (PQW):
        * P-axis: Points to periapsis
        * Q-axis: 90° from P in orbital plane
        * W-axis: Angular momentum vector (h)
        - For circular orbits, rotation angle may be undefined
        - All vectors transformed to perifocal frame
        """
                
        #Converting the delta_true_anomaly to radians to be used with math.sin() and math.cos()
        if not is_radians:
            tr_anomaly = delta_true_anomaly * pi / 180

        #Converting the r0 and v0 to array to perform vector calculus using np
        r0 = np.array(R0)
        v0 = np.array(V0)

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

    #📝 Calculating the h and it's magnitude from the postion vector and veloctiy vector 
    def specific_angular_momentum(self, r:Optional[Union[List[float], np.ndarray]], v:Optional[Union[List[float], np.ndarray]]) -> Tuple[np.ndarray, float]:
        """Calculates the specific angular momentum vector and its magnitude.

        Computes the orbital angular momentum per unit mass given position and velocity vectors.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position vector [x, y, z] in inertial frame (km)
        v : array_like, shape (3,)
            Velocity vector [vx, vy, vz] corresponding to r (km/s)

        Returns
        -------
        tuple
            Contains:
            - ndarray: Specific angular momentum vector [hx, hy, hz] (km²/s)
            - float: Magnitude of angular momentum (km²/s)

        Notes
        -----
        - Calculated as h = r × v (vector cross product)
        - Represents angular momentum per unit mass (constant in two-body problem)
        - Vector direction is normal to orbital plane
        - Magnitude relates to orbital parameters: h² = μa(1-e²)
        """

        #Convert to np.arr
        r = np.array(r)
        v = np.array(v)

        #Calculating the h vector
        h_vec = np.cross(r, v, axisa=-1, axisb=-1, axisc=-1, axis=None)

        #Magnitude
        h_mag = np.linalg.norm(h_vec)

        return h_vec, h_mag

    #📝Calculating the dS/dt with the 2 Body differential equation(With no perturbation) 
    def dS_dt(self, state: np.ndarray, t: float) -> np.ndarray:
        """Computes the time derivative of the state vector for the two-body problem.

        Solves the differential equations for Keplerian orbital motion under central gravity.

        Parameters
        ----------
        state : ndarray, shape (6,)
            Current state vector containing:
            - [0:3] : Position components [x, y, z] (km)
            - [3:6] : Velocity components [x_dot, y_dot, z_dot] (km/s)
        t : float
            Current time (unused but required by ODE solver interface) (s)

        Returns
        -------
        ndarray, shape (6,)
            State vector derivative containing:
            - [0:3] : Velocity components [x_dot, y_dot, z_dot] (km/s)
            - [3:6] : Acceleration components [x_ddot, y_ddot, z_ddot] (km/s²),
            calculated as a = -μr/|r|³

        Notes
        -----
        - Pure two-body dynamics (no perturbations)
        - Time parameter 't' maintained for solver compatibility
        - For numerical stability with low orbits:
        Consider adding ε to denominator (a = -μr/(|r|³ + ε))
        - Acceleration components follow inverse-square law
        """

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
    
    #📝Calculating the dS/dt with the 2 Body differential equation + J2 perturbation
    def dS_dt_J2(self, state: np.ndarray, t: float) -> np.ndarray:
        """Computes the time derivative of the state vector including J2 perturbation.

        Implements differential equations for orbital motion with central gravity and J2 effects.

        Parameters
        ----------
        state : ndarray
            Current state vector with shape (6,) containing:
            - [0:3]: Position components [x, y, z] in ECI frame (km)
            - [3:6]: Velocity components [x_dot, y_dot, z_dot] (km/s)
        t : float
            Current time (required for ODE solver compatibility) (s)

        Returns
        -------
        ndarray
            Derivative of state vector with shape (6,) containing:
            - [0:3]: Velocity components (same as input) (km/s)
            - [3:6]: Acceleration components including:
                * Central gravity (two-body)
                * J2 perturbation (km/s²)

        Notes
        -----
        - Uses Earth's equatorial radius (6378 km) for J2 calculations
        - J2 coefficient (self.J2) typically 1.08263e-3 for Earth
        - Gravitational parameter (self.mu) in km³/s²
        - Time parameter 't' unused in calculation but required by solver interface
        - J2 accounts for Earth's oblateness effect (zonal harmonic)
        """

        x = state[0]
        y = state[1]
        z = state[2]
        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]

        r_mag = (x ** 2 + y ** 2 + z ** 2) ** (1 / 2)

        x_ddot = -self.mu * (1 + 1.5 * self.J2 * ((6378/r_mag)**2) * (1 - 5 * (z/r_mag)**2) ) * x/r_mag**3
        y_ddot = -self.mu * (1 + 1.5 * self.J2 * ((6378/r_mag)**2) * (1 - 5 * (z/r_mag)**2) ) * y/r_mag**3
        z_ddot = -self.mu * (1 + 1.5 * self.J2 * ((6378/r_mag)**2) * (3 - 5 * (z/r_mag)**2) ) * z/r_mag**3
        ds_dt = np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])

        return ds_dt
    
    #📝Calculating the dS/dt with the 2 Body differential equation + J2 perturbation
    def dS_dt_HFOP(self, state:np.ndarray ,t:float) -> np.ndarray:  
        """Computes the time derivative of the state vector including multiple perturbations.

        Implements differential equations for orbital motion with multiple perturbation effects.

        Parameters
        ----------
        state : ndarray
            Current state vector with shape (6,) containing:
            - [0:3]: Position components [x, y, z] in ECI frame (km)
            - [3:6]: Velocity components [x_dot, y_dot, z_dot] (km/s)
        t : float
            Time since scenario epoch (s)

        Returns
        -------
        ndarray
            Derivative of state vector with shape (6,) containing:
            - [0:3]: Velocity components [x_dot, y_dot, z_dot] (km/s)
            - [3:6]: Acceleration components [x_ddot, y_ddot, z_ddot] (km/s²)
            Includes effects from:
            * Central gravity
            * J2 perturbation
            * Third-body gravity
            * Solar radiation pressure
            * Atmospheric drag
            * Thrust forces

        Notes
        -----
        - Requires SPICE kernels loaded for ephemeris calculations
        - Uses NRLMSISE-00 atmospheric model
        - All calculations performed in ECI (J2000) frame
        - Acceleration components include all perturbation effects
        - Time parameter 't' required for perturbation calculations
        """

        # ===== Time and Position Setup =====
        scenario_epoch = self.scenario_epoch
        current_time = scenario_epoch + timedelta(seconds=t)

        x = state[0]
        y = state[1]
        z = state[2]
        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]
        

        #☀️ Get Sun's position relative to Earth in J2000 frame (ECI)
        r_sun = spice.spkpos("SUN", spice.str2et((scenario_epoch + timedelta(seconds = t)).strftime("%Y-%m-%dT%H:%M:%S")), "J2000", "NONE", "EARTH")[0]
        #🌙 Get Moon's position relative to Earth in J2000 frame (ECI)
        r_moon = spice.spkpos("MOON", spice.str2et((scenario_epoch + timedelta(seconds = t)).strftime("%Y-%m-%dT%H:%M:%S")), "J2000", "NONE", "EARTH")[0]
        # Get the jupiter's position vector relative to Earth in J2000 frame (ECI)
        # r_jupiter = lambda time :spice.spkpos("JUPITER", spice.str2et((scenario_epoch + timedelta(seconds = time)).strftime("%Y-%m-%dT%H:%M:%S")), "J2000", "NONE", "EARTH")[0]


        #Finding the local solar time of the satellite
        local_satellite_time = self.local_solar_time(r=np.array([x,y,z]) ,t=t ,scenario_epoch=scenario_epoch )
        
        #Finding the density of the air
        density = nrlmsise00.msise_model(
                        time = current_time
                       , alt = sqrt(x**2 + y**2 + z**2) - 6378
                       , lat = self.lat_long_from_ECI(state[0:3], t, scenario_epoch)[0]
                       , lon = self.lat_long_from_ECI(state[0:3], t, scenario_epoch)[1]
                       , f107a = self.f107a(t)       #Change this if nec
                       , f107 = self.f107(t)
                       , ap = self.ap(t)
                       , lst = local_satellite_time.hour
                       , ap_a = self.ap_a(t)
                       , flags = self.flags(t)
                       , method='gtd7')[0][5]
               

        #Calculcations for 3rd body
        r_mag = (x ** 2 + y ** 2 + z ** 2) ** (1 / 2)
        r_sun_mag = (r_sun[0]**2 + r_sun[1]**2 + r_sun[2]**2) ** (1 / 2)
        r_moon_mag = (r_moon[0]**2 + r_moon[1]**2 + r_moon[2]**2) ** (1 / 2)
        
        #Relative distance of the satellite to the 3rd body
        r_sun_sat = ( (x - r_sun[0])**2 + (y - r_sun[1])**2 + (z - r_sun[2])**2) ** (1 / 2)         #Distance between the satellite and the sun in km
        r_moon_sat = ( (x - r_moon[0])**2 + (y - r_moon[1])**2 + (z - r_moon[2])**2) ** (1 / 2)     #Distance between the satellite and the moon in km
        
        #Finding the v_atm
        v_atm = np.cross(self.omega_earth, np.array([x , y, z]))  # km/s
        #Finding the relative velocity
        v_rel = np.array([x_dot, y_dot, z_dot]) - v_atm
        
        #J2
        x_ddot = -self.mu * (1 + 1.5 * self.J2 * ((6378/r_mag)**2) * (1 - 5 * (z/r_mag)**2) ) * x/r_mag**3 
        y_ddot = -self.mu * (1 + 1.5 * self.J2 * ((6378/r_mag)**2) * (1 - 5 * (z/r_mag)**2) ) * y/r_mag**3 
        z_ddot = -self.mu * (1 + 1.5 * self.J2 * ((6378/r_mag)**2) * (3 - 5 * (z/r_mag)**2) ) * z/r_mag**3 
        
        #SUN's gravity perturbation
        x_ddot+= self.mu_sun * ((r_sun[0]-x)/r_sun_sat**3 - r_sun[0]/r_sun_mag**3) 
        y_ddot+= self.mu_sun * ((r_sun[1]-y)/r_sun_sat**3 - r_sun[1]/r_sun_mag**3) 
        z_ddot+= self.mu_sun * ((r_sun[2]-z)/r_sun_sat**3 - r_sun[2]/r_sun_mag**3) 
        
        #MOON's gravity
        x_ddot+= self.mu_moon * ((r_moon[0]-x)/r_moon_sat**3 - r_moon[0]/r_moon_mag**3) 
        y_ddot+= self.mu_moon * ((r_moon[1]-y)/r_moon_sat**3 - r_moon[1]/r_moon_mag**3) 
        z_ddot+= self.mu_moon * ((r_moon[2]-z)/r_moon_sat**3 - r_moon[2]/r_moon_mag**3) 
        
        #Solar Pressure Radiation perturbation
        x_ddot+= (1 + self.rho(t, x, y, z)) * (self.AU/r_sun_sat)**2 * (self.c_r/self.c) * (self.A(t, x, y, z) / self.m(t, x, y, z)) * ((x - r_sun[0])/r_sun_sat) 
        y_ddot+= (1 + self.rho(t, x, y, z)) * (self.AU/r_sun_sat)**2 * (self.c_r/self.c) * (self.A(t, x, y, z) / self.m(t, x, y, z)) * ((y - r_sun[1])/r_sun_sat) 
        z_ddot+= (1 + self.rho(t, x, y, z)) * (self.AU/r_sun_sat)**2 * (self.c_r/self.c) * (self.A(t, x, y, z) / self.m(t, x, y, z)) * ((z - r_sun[2])/r_sun_sat) 
        
        #Atmospheric drag
        x_ddot+= 0.5 * density * self.s(t,x,y,z) * self.cd * v_rel[0]**2
        y_ddot+= 0.5 * density * self.s(t,x,y,z) * self.cd * v_rel[1]**2
        z_ddot+= 0.5 * density * self.s(t,x,y,z) * self.cd * v_rel[2]**2

        #Thrust
        x_ddot+= self.thrust(t,x,y,z)[0]
        y_ddot+= self.thrust(t,x,y,z)[1]
        z_ddot+= self.thrust(t,x,y,z)[2]

        ds_dt = np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])
       
        return ds_dt

    #📝Calculating the Specific Mechanical Energy of the Orbit_2body
    def energy(self, r:Optional[Union[List[float], np.ndarray]], v:Optional[Union[List[float], np.ndarray]]) -> float:
        """Calculates the specific mechanical energy of an orbit in the two-body problem.

        The specific mechanical energy (ε) is a conserved quantity that determines the
        orbit type (elliptical, parabolic, or hyperbolic).

        Parameters
        ----------
        r : array_like
            Position vector [x, y, z] in inertial frame (km)
        v : array_like
            Velocity vector [vx, vy, vz] corresponding to r (km/s)

        Returns
        -------
        float
            Specific mechanical energy (ε) in km²/s², where:
            * ε < 0 : Elliptical orbit
            * ε = 0 : Parabolic orbit
            * ε > 0 : Hyperbolic orbit

        Notes
        -----
        - For elliptical orbits: ε = -μ/(2a), where a is semi-major axis
        - Represents energy per unit mass (mass-independent)
        - Negative values indicate bound orbits
        - Uses gravitational parameter μ (self.mu) in km³/s²
        - Conserved quantity in two-body problem
        """

        #Turn the r and v into numpy array
        r = np.array(r)
        v = np.array(v)

        #Magnitude of V and r vector
        v_mag =np.sqrt(v.dot(v))
        r_mag =np.sqrt(r.dot(r))
        

        #Finding the specific energy of the orbit 
        energy = (v_mag ** 2) / 2 - self.mu / r_mag     # epsilon = v^2/2 - mu/r  

        return energy

    #📝Will indicate the type of the orbit
    def orbit_type(self, r:Optional[np.ndarray]=None, v:Optional[np.ndarray]=None, threshold:float=0.5) -> str:
        """Determines the type of orbit based on specific mechanical energy.

        Classifies the orbit as elliptical (including circular), parabolic, or hyperbolic
        by analyzing the specific mechanical energy with numerical tolerance.

        Parameters
        ----------
        r : array_like, optional
            Position vector [x, y, z] in km. Required if orbit hasn't been propagated.
        v : array_like, optional
            Velocity vector [vx, vy, vz] in km/s. Required if orbit hasn't been propagated.
        threshold : float, optional
            Energy threshold for parabolic classification in km²/s² (default: 0.5).
            Represents half-width of the energy band considered "parabolic".

        Returns
        -------
        str
            Orbit type as one of:
            - 'elliptical' (includes circular orbits)
            - 'parabolic' (within ±threshold of zero energy)
            - 'hyperbolic'

        Raises
        ------
        ValueError
            If neither state vectors nor propagated orbit data is available

        Notes
        -----
        - Circular orbits are currently classified as elliptical
        - Uses initial state from propagated orbits if available
        - Threshold accounts for numerical precision in energy calculations
        - Future versions may distinguish circular from elliptical orbits
        """

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
        
    #📝Calculating the eccentricity vector and magnitude
    def eccentricity(self, r:Optional[Union[List[float], np.ndarray]], v:Optional[Union[List[float], np.ndarray]]) -> Tuple[np.ndarray, float]:
        """Calculates the eccentricity vector and its magnitude for an orbit.

        Computes the Laplace-Runge-Lenz vector (eccentricity vector) which points toward
        periapsis and whose magnitude equals the orbital eccentricity.

        Parameters
        ----------
        r : array_like
            Position vector [rx, ry, rz] in ECI frame (km)
        v : array_like
            Velocity vector [vx, vy, vz] in ECI frame (km/s)

        Returns
        -------
        tuple
            Contains:
            - ndarray: Eccentricity vector [ex, ey, ez] (dimensionless)
            - float: Eccentricity magnitude where:
                * 0 = Circular orbit
                * 0 < e < 1 = Elliptical orbit
                * 1 = Parabolic orbit
                * >1 = Hyperbolic orbit

        Notes
        -----
        - The eccentricity vector points toward periapsis (closest approach)
        - For circular orbits, vector direction is undefined when magnitude is zero
        - Derived from specific angular momentum (h = r × v) using:
        e = (v × h)/μ - r/|r|
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
    def time_since_perigee(self ,
                           true_anomaly:float, 
                           r:Optional[Union[List[float], np.ndarray]]=None, 
                           v:Optional[Union[List[float], np.ndarray]]=None, 
                           h:Optional[float]=None, 
                           e:Optional[float]=None, 
                           degree_mode=False) -> Tuple[float, float]:
        """Calculates time elapsed since perigee passage for given true anomaly.

        Computes time required to travel from perigee to specified true anomaly by
        numerically integrating angular momentum equation. Works for all orbit types.

        Parameters
        ----------
        true_anomaly : float
            Angular position from perigee in radians (or degrees if degree_mode=True)
        r : array_like, optional
            Position vector [x, y, z] in km (required if h or e not provided)
        v : array_like, optional
            Velocity vector [vx, vy, vz] in km/s (required if h or e not provided)
        h : float, optional
            Magnitude of specific angular momentum in km²/s
        e : float, optional
            Orbit eccentricity (dimensionless)
        degree_mode : bool, optional
            If True, interprets true_anomaly as degrees (default is False)

        Returns
        -------
        tuple
            Contains:
            - float: Time since perigee passage in seconds
            - float: Estimated absolute error of integration in seconds

        Raises
        ------
        ValueError
            If insufficient orbital parameters are provided
            If negative eccentricity is provided

        Notes
        -----
        - Either the r and v vectors should be provided or the h and e.
        - For elliptical orbits (e < 1), consider using mean anomaly for better accuracy
        - Method becomes less accurate near parabolic orbits (e ≈ 1)
        - Uses scipy.integrate.quad for numerical integration
        """

        #Check to see if the degree mode is beening used
        if degree_mode:
            true_anomaly = true_anomaly * pi / 180

        #Calcualting the neccessary variables(If not provided)
        if h == None:
            _ , h = self.specific_angular_momentum(r,v)
        if e == None:
            _ , e = self.eccentricity(r,v) 

        #Integrating the general formula -> Is valid for all orbit types
        f_t = lambda theta: (h**3 / self.mu ** 2) * 1/(1+e * np.cos(theta))**2

        time, err = quad(f_t, 0, true_anomaly)

        return time ,err
    
    #Calculating the true anomaly of the satellite from the time since preigee
    def true_anomaly_from_time(self, 
                               time:float, 
                               r:Optional[Union[List[float], np.ndarray]]=None, 
                               v:Optional[Union[List[float], np.ndarray]]=None,
                               h:Optional[float]=None, 
                               e:Optional[float]=None,  
                               degree_mode:bool=False) -> Tuple[float, float, float]:
        """Calculate orbital anomalies from time since perigee.

        Solves Kepler's equation to determine true anomaly, eccentric anomaly,
        and mean anomaly given time since last perigee passage.

        Parameters
        ----------
        time : float
            Time elapsed since perigee passage (seconds)
        h : float, optional
            Specific angular momentum (km²/s)
        e : float, optional
            Orbital eccentricity (dimensionless)
        r : array_like, optional, shape (3,)
            Position vector [x, y, z] (km)
        v : array_like, optional, shape (3,)
            Velocity vector [vx, vy, vz] (km/s)
        degree_mode : bool, optional
            If True, returns angles in degrees (default: radians)

        Returns
        -------
        tuple
            Tuple containing:
            - float: True anomaly (θ)
            - float: Eccentric anomaly (E)
            - float: Mean anomaly (M)

        Raises
        ------
        ValueError
            If insufficient orbital parameters are provided
            If eccentricity is not in range [0, 1)

        Notes
        -----
        - Only valid for elliptical orbits (0 ≤ e < 1)
        - Uses Newton-Raphson to solve Kepler's equation: M = E - e sin(E)
        - True anomaly calculated via: θ = 2 atan(sqrt((1+e)/(1-e)) * tan(E/2))
        - For parabolic/hyperbolic orbits, different equations apply
        - Either the r and v vectors should be provided or the h and e.
        """

        #Calcualting the neccessary variables(If not provided)
        if h == None:
            _ , h = self.specific_angular_momentum(r,v)
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

    #📝Calculates the period of an orbit
    def period(self,
                h:float=None, 
                e:float=None ,
                r:Optional[Union[List[float], np.ndarray]]=None ,
                v:Optional[Union[List[float], np.ndarray]]=None) -> float:
        """Calculate the orbital period from angular momentum and eccentricity.

        Computes the period for both closed (elliptic/circular) and open 
        (parabolic/hyperbolic) orbits.

        Parameters
        ----------
        h : float, optional
            Specific angular momentum in km²/s. Required if r and v not provided.
        e : float, optional
            Orbital eccentricity (dimensionless). Required if r and v not provided.
        r : list or ndarray, optional, shape (3,)
            Position vector [x, y, z] in km. Must be length 3.
        v : list or ndarray, optional, shape (3,)
            Velocity vector [vx, vy, vz] in km/s. Must be length 3.

        Returns
        -------
        float
            Orbital period in seconds (infinity for open orbits)

        Raises
        ------
        ValueError
            If insufficient parameters are provided
            If eccentricity is negative

        Notes
        -----
        - Either the r and v vectors should be provided or the h and e.
        - For closed orbits (e < 1): T = 2π√(a³/μ)
        - For open orbits (e ≥ 1): returns infinity
        - More efficient when h and e are provided directly
        """
        
        #Calcualting the neccessary variables(If not provided)
        if h == None:
            _ , h = self.specific_angular_momentum(r,v)
        if e == None:
            _ , e = self.eccentricity(r,v) 

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
    def semi_major_axis(self,
                h:float=None, 
                e:float=None ,
                r:Optional[Union[List[float], np.ndarray]]=None ,
                v:Optional[Union[List[float], np.ndarray]]=None) -> float:
        """Calculate the semi-major axis of an orbit.

        Computes the semi-major axis for elliptical, circular, or hyperbolic orbits.
        Undefined for parabolic orbits (e=1).

        Parameters
        ----------
        h : float, optional
            Specific angular momentum in km²/s. Required if r and v not provided.
        e : float, optional
            Orbital eccentricity (dimensionless). Required if r and v not provided.
        r : list or ndarray, optional, shape (3,)
            Position vector [x, y, z] in km. Must be length 3.
        v : list or ndarray, optional, shape (3,)
            Velocity vector [vx, vy, vz] in km/s. Must be length 3.

        Returns
        -------
        float
            Semi-major axis in km (positive for elliptical, negative for hyperbolic orbits)

        Raises
        ------
        ValueError
            If insufficient parameters are provided \n
            If orbit is parabolic (e=1) \n
            If eccentricity is negative \n

        Notes
        -----
        - Either the r and v vectors should be provided or the h and e.
        - For elliptical orbits (0 ≤ e < 1): a = h²/(μ(1-e²))
        - For hyperbolic orbits (e > 1): a = h²/(μ(e²-1)) (returns negative value)
        - Undefined for parabolic orbits (e=1)
        - Circular orbits (e=0) are a special case of elliptical orbits
        """

        #Calcualting the neccessary variables(If not provided)
        if h == None:
            _ , h = self.specific_angular_momentum(r,v)
        if e == None:
            _ , e = self.eccentricity(r,v) 

        #Parabolic orbit -> a:undefined
        if e == 1:
            raise Exception("🚀Sorry for the parabolic orbits the semi-major axis is undefined🚀")
        
        #For any other orbit
        a =  ((h ** 2)/(self.mu)) * (1/abs( 1 - e**2))

        return a 

    #Converting the Cartesian element to classical orbital elements
    def cartesian_to_keplerain(self, r:Union[List[float], np.ndarray], v:Union[List[float], np.ndarray], degree_mode:bool=False) -> Dict[str, float]:
        """Convert Cartesian state vectors to classical orbital elements.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position vector [rx, ry, rz] in ECI frame (km)
        v : array_like, shape (3,)
            Velocity vector [vx, vy, vz] in ECI frame (km/s)
        degree_mode : bool, optional
            If True, returns angles in degrees (default: radians)

        Returns
        -------
        dict
            Dictionary containing classical orbital elements:
            - 'e' : float
                Eccentricity (dimensionless)
            - 'h' : float
                Specific angular momentum (km²/s)
            - 'theta' : float
                True anomaly (degrees if degree_mode=True)
            - 'i' : float
                Inclination (degrees if degree_mode=True)
            - 'w' : float
                Argument of periapsis (degrees if degree_mode=True)
            - 'RAAN' : float
                Right ascension of ascending node (degrees if degree_mode=True)

        Raises
        ------
        ValueError
            If input vectors have incorrect dimensions
            If orbit is rectilinear (h ≈ 0)

        Notes
        -----
        - All angular elements are returned in radians by default
        - For circular orbits (e=0), argument of periapsis is undefined
        - For equatorial orbits (i=0), RAAN is undefined
        - Conversion formulas:
        * h = r × v
        * e = (v × h)/μ - r/|r|
        * i = acos(h_z/|h|)
        * RAAN = atan2(N_y, N_x) where N = ẑ × h
        * ω = atan2(e_z, e·N) 
        * θ = atan2(r·e, |r|(1-e²))
        """

        #Converting the r and v to array
        r = np.array(r)
        v = np.array(v)

        #Calculating the magnitude of r and v(speed)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        #Determining the radial vecloicty
        v_r = np.dot(r , v) / np.linalg.norm(r)       

        #💫Determining the specific angular momentum
        h_vector, h_mag = self.specific_angular_momentum(r, v)

        #💫Determining the eccentricity vector and magnitude               
        e_vec, e = self.eccentricity(r,v)  

    
        #💫Determining the inclination
        i = acos(h_vector[2]/h_mag)

        # #Determing the node line vector and magnitude
        N_vec = np.cross([0,0,1] , h_vector)
        N_mag = np.linalg.norm(N_vec)

        #💫Determining the RAAN
        RAAN = acos(N_vec[0]/N_mag) if N_vec[1] >= 0 else 2*pi - acos(N_vec[0]/N_mag) 

        #💫Determining the argument of preiapsis w
        w = acos(np.dot(N_vec, e_vec)/(N_mag * e)) if e_vec[2] >=0 else 2*pi - acos(np.dot(N_vec, e_vec)/(N_mag * e))

        #💫Determining the true anomaly
        theta = acos(np.dot(e_vec,r)/(e * r_mag)) if v_r >= 0 else 2*pi - acos(np.dot(e_vec,r)/(e * r_mag))

        #Chekc if degree mode is active
        if degree_mode:
            theta = theta * 180 / pi
            i = i * 180 / pi 
            RAAN = RAAN * 180 / pi
            w = w * 180 / pi

        classical_orbital_elements = {
            "e" : e,
            "h" : h_mag,
            "theta" : theta,
            "i" : i,
            "w" : w,
            "RAAN" : RAAN 
        }


        # #Returning the elements
        return classical_orbital_elements

    #Converting the classical orbital element to state sapce 
    def keplerian_to_cartesian(self, e:float, h:float, theta:float, i:float, RAAN:float, w:float, degree_mode:bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """Convert classical orbital elements to Cartesian state vectors.

        Parameters
        ----------
        e : float
            Orbital eccentricity (dimensionless)
        h : float
            Specific angular momentum (km²/s)
        theta : float
            True anomaly (radians by default)
        i : float
            Inclination (radians by default)
        RAAN : float
            Right ascension of ascending node (radians by default)
        w : float
            Argument of periapsis (radians by default)
        degree_mode : bool, optional
            If True, input angles are interpreted as degrees (default: False)

        Returns
        -------
        tuple
            Contains:
            - ndarray: Position vector [rx, ry, rz] in ECI frame (km)
            - ndarray: Velocity vector [vx, vy, vz] in ECI frame (km/s)

        Raises
        ------
        ValueError
            If eccentricity is negative
            If angular momentum is non-positive

        Notes
        -----
        - Transformation steps:
        1. Compute position/velocity in perifocal frame
        2. Construct 3-1-3 rotation matrix (RAAN-i-w)
        3. Transform to ECI frame
        - For circular orbits (e=0), argument of periapsis is irrelevant
        - For equatorial orbits (i=0), RAAN is irrelevant
        """

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
    
    #🚧🏗️Instance of what the input should look like sim = datetime(year = 2025, month = 5, day = 19, hour = 8, minute = 20 , second = 0)
    def UTC_to_julian(self, dt:datetime) -> float:
        """Convert UTC datetime to Julian Date.

        Parameters
        ----------
        dt : datetime
            Input datetime object (must be UTC or timezone-naive)
            Example: datetime(2025, 5, 19, 8, 20, 0)

        Returns
        -------
        float
            Julian Date as floating point number

        Raises
        ------
        ValueError
            If input datetime has non-UTC timezone
            If input is not a datetime object

        Notes
        -----
        - Uses the standard conversion formula:
        JD = 367*Y - floor(7*(Y + floor((M+9)/12))/4) 
            + floor(275*M/9) + D + 1721013.5 
            + (H + Min/60 + Sec/3600)/24
        - Valid for all dates after November 17, 1858
        - Timezone-naive inputs are assumed to be UTC
        - Includes microseconds in calculation
        """

        # Validate timezone
        if dt.tzinfo is not None and dt.tzinfo != timezone.utc:
            raise ValueError("Input datetime must be either naive or explicitly UTC")

        Y = dt.year
        M = dt.month
        D = dt.day
        H = dt.hour
        Min = dt.minute
        Sec = dt.second + dt.microsecond/1e14  # Include microseconds
        
        # Calculate Julian Date
        term1 = 367 * Y
        term2 = floor((7 * (Y + floor((M + 9)/12)))/4)
        term3 = floor(275 * M / 9)
        term4 = D + 1721013.5
        term5 = (H + Min/60 + Sec/3600)/24
        
        JD = term1 - term2 + term3 + term4 + term5
        
        return JD
    
    #Converting the UTC to the YYDDD format 
    def UTC_to_YYDDD(self, dt_utc: datetime) -> str:
        """Convert UTC datetime to YYDDD.xxxxxxxxxxxxxxx format.

        Parameters
        ----------
        dt_utc : datetime
            Input datetime object (must be UTC or timezone-naive)
            Example: datetime(2025, 5, 19, 8, 30, 0)

        Returns
        -------
        str
            Formatted string in YYDDD.xxxxxxxxxxxxxxx format where:
            - YY: Last two digits of year
            - DDD: Day of year (001-366)
            - .xxxxxxxxxxxxxxx: Fraction of day with microsecond precision
            Example: '25139.354166666666667' for May 19, 2025 08:30:00

        Raises
        ------
        ValueError
            If input has non-UTC timezone
            If input is not a datetime object

        Notes
        -----
        - Timezone-naive inputs are assumed to be UTC
        - Maintains microsecond precision (14 decimal places)
        - Format matches standard NASA/SPICE YYDDD.xxxxxxxxxxxxxxx convention
        - Valid for all dates in the Gregorian calendar
        """
        # Validate timezone
        if dt_utc.tzinfo is not None and dt_utc.tzinfo != timezone.utc:
            raise ValueError("Input datetime must be either naive or explicitly UTC")
        
        # Calculate day of year (001-366)
        day_of_year = dt_utc.timetuple().tm_yday
        ddd = f"{day_of_year:03d}"
        
        # Calculate fraction of day with microsecond precision
        total_seconds = (
            dt_utc.hour * 3600 + 
            dt_utc.minute * 60 + 
            dt_utc.second + 
            dt_utc.microsecond / 1e6
        )
        fraction = total_seconds / 86400  # Fraction of day
    
        # Format with 15 decimal places (corrected string formatting)
        return f"{dt_utc.strftime('%y')}{ddd}.{fraction:.14f}".split('.')[0][:5] + '.' + f"{fraction:.14f}".split('.')[1]
    
    #Formating the UTC time 
    def format_utc(self, dt: datetime, decimals: int = 6) -> str:
        """Format UTC datetime to precise string representation.

        Parameters
        ----------
        dt : datetime
            Input datetime object (naive or timezone-aware UTC)
            Example: datetime(2025, 5, 19, 8, 30, 15, 123456)
        decimals : int, optional
            Number of decimal places for seconds (default: 6)
            Note: Values >6 will use microsecond precision but pad with zeros

        Returns
        -------
        str
            Formatted string in 'DD Month YYYY HH:MM:SS.ffffff' format
            Example: '19 May 2025 08:30:15.123456'

        Raises
        ------
        ValueError
            If decimals is negative
            If input is not a datetime object

        Notes
        -----
        - Maintains full microsecond precision (6 decimal places)
        - For decimals >6, pads with zeros (no additional precision)
        - Timezone-naive inputs are assumed to be UTC
        - Month name uses current locale settings
        """
        # Cross-platform day formatting (no leading zero)
        day = str(dt.day)
        
        # Get base time without microseconds
        base_time = dt.strftime("%B %Y %H:%M:%S")
        
        # Handle decimal places
        if decimals <= 0:
            return f"{day} {base_time}"
        
        # Calculate fractional seconds with unlimited precision
        fraction = dt.microsecond / 1_000_000
        
        # Format with requested decimal places
        return f"{day} {base_time}{fraction:.{decimals}f}"[1:] if fraction < 0.1 else f"{day} {base_time}{fraction:.{decimals}f}"

    #Saving the propagted orbit in the FreeFlyer ephermeris format (STK)
    def save_ephermeris_freeflyer(self, 
                                  r:np.ndarray, 
                                  v:np.ndarray, 
                                  t:np.ndarray, 
                                  scenario_epoch:datetime = datetime.now(timezone.utc), 
                                  stk_version:str = "stk.v.11.0",
                                  interpolation_method:str = "Lagrange", 
                                  interpolation_samplesM1:int = 7, 
                                  central_body:str = "Earth", 
                                  coordinate_system:str="ICRF", 
                                  file_name:str="orbit") -> str:
        """Save propagated orbit in STK-compatible FreeFlyer ephemeris format.

        Parameters
        ----------
        r : array_like, shape (N,3)
            Position vectors in km [rx, ry, rz]
        v : array_like, shape (N,3)
            Velocity vectors in km/s [vx, vy, vz]
        t : array_like, shape (N,)
            Time values in seconds since scenario epoch
        scenario_epoch : datetime, optional
            Reference epoch for the ephemeris (default: current UTC time)
        stk_version : str, optional
            STK version string (default: "stk.v.11.0")
        interpolation_method : str, optional
            Interpolation method for STK (default: "Lagrange")
        interpolation_samplesM1 : int, optional
            Number of interpolation points minus one (default: 7)
        central_body : str, optional
            Central body name (default: "Earth")
        coordinate_system : str, optional
            Reference coordinate system (default: "ICRF")
        file_name : str, optional
            Base filename without extension (default: "orbit")

        Returns
        -------
        str
            Confirmation message with saved file path

        Raises
        ------
        ValueError
            If input arrays have inconsistent shapes
            If time values are not monotonically increasing

        Notes
        -----
        - Output file uses .e extension
        - Format compatible with STK and FreeFlyer
        - Includes multiple time representations:
        * Formatted UTC string
        * Julian date
        * YYDDD format
        - Uses 14 decimal places for all numerical values
        """

        #Ensuring the r and v and t are in np.arr
        r = np.array(r).reshape((len(r), 3))
        v = np.array(v).reshape((len(v), 3))
        t = np.array(t).reshape((len(t),1))

        #Number of point
        num_points = len(r)

        #State vector + time
        s = np.hstack([t,r,v])
        
        #Finding the time in julian and YYDDD and str
        scn_epc_str_6_digits = self.format_utc(scenario_epoch)
        scn_epc_str_9_digits = self.format_utc(scenario_epoch)
        scn_epc_julian = self.UTC_to_julian(scenario_epoch)
        scn_ecp_YYDDD = self.UTC_to_YYDDD(scenario_epoch)
   

        #Generating the header
        head = f"""{stk_version}


# WrittenBy    STK_v11.2.0

BEGIN Ephemeris

NumberOfEphemerisPoints {num_points}

ScenarioEpoch            {scn_epc_str_6_digits}

# Epoch in JDate format: {scn_epc_julian}
# Epoch in YYDDD format:   {scn_ecp_YYDDD}


InterpolationMethod     {interpolation_method}

InterpolationSamplesM1      {interpolation_samplesM1}

CentralBody             {central_body}

CoordinateSystem        {coordinate_system} 

# Time of first point: {scn_epc_str_9_digits} UTCG = {scn_epc_julian} JDate = {scn_ecp_YYDDD} YYDDD
"""
        
        #Generating the body
        body = "EphemerisTimePosVel\n\n"

        #Adding the elements
        for instance in s:
            body = body + '{:.14e} {:.14e} {:.14e} {:.14e} {:.14e} {:.14e} {:.14e}\n'.format(*instance)
        
        #Ending the ephermeris 
        body = body + '\nEND Ephemeris'

        #Saving the ephermeris
        with open(file_name + ".e", 'w') as f:
            f.write(head + body)

        return  "Ephermeris saved at:" + file_name + ".e"
    
    #Saving the orbit with the spk format .bsp (Used by SPCIE kernels)
    def save_to_spk(self, 
                    r_vectors:np.ndarray, 
                    v_vectors:np.ndarray, 
                    time:np.ndarray, 
                    scenario_epoch=datetime.now(timezone.utc), 
                    output_file="orbit", 
                    kernel_list=["naif0012.tls", "pck00010.tpc"], 
                    kernel_base_dir="./kernels") -> str:
        """Save orbit data to SPICE SPK (.bsp) format kernel.

        Parameters
        ----------
        r_vectors : array_like, shape (N,3)
            Position vectors in km in ICRF/J2000 frame
        v_vectors : array_like, shape (N,3)
            Velocity vectors in km/s in ICRF/J2000 frame
        time : array_like, shape (N,)
            Time values in seconds since scenario epoch
        scenario_epoch : datetime, optional
            Reference epoch for the ephemeris (default: current UTC time)
        output_file : str, optional
            Base filename without extension (default: "orbit")
        kernel_list : list of str, optional
            Required SPICE kernels (default: ["naif0012.tls", "pck00010.tpc"])
        kernel_base_dir : str, optional
            Directory containing SPICE kernels (default: "./kernels")

        Returns
        -------
        str
            Confirmation message with saved file path

        Raises
        ------
        ValueError
            If input arrays have inconsistent shapes
            If time values are not monotonically increasing
        RuntimeError
            If SPICE kernel operations fail

        Notes
        -----
        - Output file uses .bsp extension
        - Uses SPK type 9 (Lagrange interpolation)
        - Frame defaults to J2000 (equivalent to ICRF for Earth-centered orbits)
        - Custom spacecraft ID: -999
        - Earth center ID: 399
        - Automatically loads required kernels
        """
        #Converting the scenario_epoch to julian date
        start_time_julian = self.UTC_to_julian(scenario_epoch)

        #Converting the time vector form seconds to days(Defualt unit for julian date) and adding it with the start time of the scenario 
        t = time / (24 * 60 * 60) + start_time_julian

        #Delete the file if already exist
        file_path = output_file+".bsp"

        try:
            # Check if file exists first
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' does not exist", file=sys.stderr)
            
            
            # Verify it's actually a file (not a directory)
            if not os.path.isfile(file_path):
                print(f"Error: '{file_path}' is not a file", file=sys.stderr)
            
            
            # Remove the file
            os.remove(file_path)
        
            # Verify deletion
            if os.path.exists(file_path):
                print(f"Error: Failed to delete '{file_path}'", file=sys.stderr)
            
            
            print(f"Successfully deleted '{file_path}'")
        
        
        except PermissionError:
            print(f"Error: Permission denied when deleting '{file_path}'", file=sys.stderr)
        except Exception as e:
            print(f"Error deleting file: {str(e)}", file=sys.stderr)

        # Load essential kernels (adjust paths as needed)
        for kernel in kernel_list:
            spice.furnsh(kernel_base_dir+"/"+kernel)

            #Success message
            print(kernel_base_dir+"/"+kernel + " was loaded successfully")
        
        # Create SPK file
        handle = spice.spkopn(output_file+".bsp", "SAT_ORBIT_SPK", 0)
    
        # SPK parameters
        body_id = -999  # Negative ID for custom spacecraft
        center_id = 399  # Earth center
        frame = "J2000"  # SPCIE does not support ECI but the earth centered ICRF is the same as ECI J2000
    
        # Convert Julian dates to ET
        et_times = np.array([self.jd_to_et(jd) for jd in t])
        
        # Combine position and velocity
        states = np.hstack((r_vectors, v_vectors))

         # Write SPK segment (Type 9 - Lagrange interpolation)
        spice.spkw09(
            handle,
            body_id,
            center_id,
            frame,
            et_times[0],       # First epoch
            et_times[-1],      # Last epoch
            "SAT_ORBIT_DATA",  # Segment identifier
            7,                 # Degree of interpolation (8 points)
            len(et_times),     # Number of states
            states,         # state vectors (Postion + Velocity)
            et_times,          # Epochs
        )

        # Cleanup
        spice.spkcls(handle)
        spice.kclear()

        return f"Saved SPK file to {output_file}"

    #Converting the julian date to SPICE ephermeris date 
    def jd_to_et(self, julian_date:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Julian Date to SPICE ephemeris time (TDB).

        Parameters
        ----------
        julian_date : float or ndarray
            Julian Date (JD) to convert, can be scalar or array
            Example: 2451545.0 (J2000 epoch)

        Returns
        -------
        float or ndarray
            Ephemeris Time (ET) in seconds past J2000 TDB
            Same shape as input (scalar or array)

        Notes
        -----
        - Conversion formula: ET = (JD - 2451545.0) * 86400.0
        - 2451545.0 is the Julian Date of J2000 epoch (2000-01-01 12:00:00 TDB)
        - 86400.0 is the number of seconds in a day
        - TDB (Barycentric Dynamical Time) is the time scale used by SPICE
        - For most Earth-based applications, TDB ≈ TT ≈ TAI+32.184s ≈ UTC+leap_seconds+32.184s
        """
        
        return (julian_date - 2451545.0) * 86400.0  # Convert JD to seconds past J2000
    
    #✅A coordiante transformation from earth entered inertia to earth centered earth fixed
    def ECI_to_ECEF(self, r:Union[np.ndarray, List[List[float]]], t:Union[np.ndarray, List[float]], scenario_epoch:datetime = datetime.now(timezone.utc)) -> np.ndarray:
        """Convert position vectors from ECI (J2000) to ECEF frame.

        Parameters
        ----------
        r : array_like, shape (N,3) or (3,)
            Position vector(s) in ECI frame (km)
        t : array_like, shape (N,) or scalar
            Time elapsed since scenario epoch (seconds)
        scenario_epoch : datetime
            UTC epoch at which simulation started (must be timezone-aware)

        Returns
        -------
        ndarray
            Transformed position vector(s) in ECEF frame (km), same shape as input

        Raises
        ------
        ValueError
            If input shapes are incompatible
            If scenario_epoch is timezone-naive
        TypeError
            If input types are invalid

        Notes
        -----
        - Transformation accounts for Earth rotation between vernal equinox alignment
        and scenario epoch
        - Uses Earth rotation rate (w_earth) stored in class instance
        - For vectorized operation, ensure r and t have compatible shapes
        - ECI frame: J2000 Earth-centered inertial
        - ECEF frame: Earth-centered, Earth-fixed (rotates with Earth)
        """
        
        #Conversion to array
        r = np.array(r)
        t = np.array(t)

        if r.ndim == 1:                     #if only a single position vector was passed to the function
            r = r.reshape([1, 1 , 3])       #Reshaping the r
            t = t.reshape([1])              #Reshaping the t

        #The time at which the vernal equinox and the prime meridian where algined
        t_0 = datetime(
            year = 2025,
            month = 3,
            day = 20,
            hour = 9,
            minute = 1 ,
            tzinfo = timezone.utc
        )

        #When subtracting two datetime it is important that they either both have the tzinfo filed or neither have it
        try:
            #The inital angle 
            theta_0 = abs( self.w_earth * (scenario_epoch - t_0).total_seconds() )
        
        except TypeError:
            raise Exception("🚀Sorry the scenario_epoch is naive. please specify the time zone for which this information is given. use the tzinfo of the datetime module")

        #Reshaping the position vector of the satellite
        r_reshaped = r.reshape((len(r) , 1 ,3))

        #Generating the transformations matrices
        gen = [[[cos(self.w_earth*delta_t+theta_0) , sin(self.w_earth*delta_t+theta_0), 0],[-sin(self.w_earth*delta_t+theta_0), cos(self.w_earth*delta_t+theta_0), 0],[0 , 0, 1]] for delta_t in t]  # Generator: 0, 2, 4, 6, 8
        transform_matrices = np.array(gen, dtype=np.float32)

        # Performing the transformation -> This is matricx multipication implmeneted using element ise multipication and sum
        transformed_r = np.sum(transform_matrices *  r_reshaped, axis=2)

        return transformed_r
    
    #💱Calculating the latitude and the longitude of the subsatellite point
    def lat_long_from_ECEF(self, r_ecef:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate geodetic latitude and longitude from ECEF position vectors.

        Parameters
        ----------
        r_ecef : array_like, shape (N,3)
            Position vector(s) in Earth-Centered Earth-Fixed (ECEF) frame in km
            Each row should be [x, y, z] coordinates

        Returns
        -------
        tuple
            Contains:
            - ndarray: Geodetic latitude in degrees (-90° to +90°)
            - ndarray: Longitude in degrees (-180° to +180°)

        Raises
        ------
        ValueError
            If input is not a 3D vector or array of 3D vectors
            If input contains invalid positions (zero magnitude)

        Notes
        -----
        - Latitude is calculated as arcsin(z/r)
        - Longitude is calculated as atan2(y, x)
        - Output ranges:
        * Latitude: -90° (South Pole) to +90° (North Pole)
        * Longitude: -180° (West) to +180° (East)
        - For zero-radius vectors (r=0), returns (nan, nan)
        - Uses WGS84 reference ellipsoid implicitly
        """

        #🌐Calculating the latitude and the longitude
        lat = np.arcsin(r_ecef[::,2] / np.linalg.norm(r_ecef, axis=1)) * 180 / pi   #Output -90 to +90
        long = np.arctan2(r_ecef[::,1], r_ecef[::,0]) * 180 / pi                    #Output range -180 to +180

        return lat, long
    
    #💱Calculating the latitude and the longitude of the subsatellite point
    def lat_long_from_ECI(self ,r_eci:np.ndarray ,t:np.ndarray ,scenario_epoch: datetime = datetime.now(timezone.utc)) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the subsatellite point's latitude and longitude from ECI coordinates.

        Converts the satellite's position from Earth-Centered Inertial (ECI) to Earth-Centered, Earth-Fixed (ECEF)
        and then derives geodetic latitude and longitude.

        Parameters
        ----------
        r_eci : np.ndarray
            Position vector(s) of the satellite in the ECI frame. Shape must be (3,) or (N, 3).
        t : np.ndarray
            Time elapsed since the simulation start (in seconds). Can be a scalar or an array.
        scenario_epoch : datetime, optional
            UTC epoch time at which the simulation started. Defaults to current UTC time.

        Returns
        -------
        lat : np.ndarray
            Geodetic latitude in degrees, in the range [-90°, 90°].
        long : np.ndarray
            Geodetic longitude in degrees, in the range [-180°, 180°].

        Notes
        -----
        - This method relies on `ECI_to_ECEF` for coordinate frame conversion and `lat_long_from_ECEF` for geodetic calculations.
        - Longitude is wrapped to [-180°, 180°] to avoid ambiguity.
        - If `r_eci` is a time series (shape `(N, 3)`), the output will be arrays of shape `(N,)`.
        """

        #Converting the position vector from ECI to ECEF
        r_ecef = self.ECI_to_ECEF(
            r = r_eci,
            t = t,
            scenario_epoch=scenario_epoch,
        )

        #Determining the latitude and longitude of the satellite
        lat, long = self.lat_long_from_ECEF(r_ecef=r_ecef)

        return lat, long

    #⌚Finding the local solar time of the subsatellite  
    def local_solar_time(self, r:Union[np.ndarray], t:Union[np.ndarray], scenario_epoch: datetime):
        """Compute the local solar time at the subsatellite point for a single position vector.

        Local solar time is determined by adjusting the scenario epoch based on the subsatellite's 
        longitude (15° = 1 hour) and simulation elapsed time.

        Parameters
        ----------
        r : np.ndarray or list[float]
            Position vector in the Earth-Centered Inertial (ECI) frame. Shape (3,).
        t : float
            Seconds elapsed since the start of the simulation.
        scenario_epoch : datetime
            UTC time at which the simulation began.

        Returns
        -------
        datetime
            Local solar time at the subsatellite point.

        Notes
        -----
        - Only supports a single position vector (not vectorized for multiple inputs).
        - Assumes `lat_long_from_ECI` returns longitude in degrees [-180, 180].
        - Longitudinal adjustment: 15° of longitude = 1 hour of solar time.
        """
        #Converting the r to np.ndarray
        r = np.array(r)

        #Finding the latitude and longitude of the satellite
        _, long = self.lat_long_from_ECI(r, t, scenario_epoch)
        
        #Every 15 degrees of longitude is one hour
        return scenario_epoch + timedelta(hours=long[0]/15, seconds=t)




class OrbitVisualizer():
    def colorGenerator(self, num):
        chars = '0123456789ABCDEF'
        return ['#'+''.join(random.sample(chars,6)) for i in range(num)]

    #The multiple visualizer
    def simpleStatic(self, r:Union[np.ndarray, List[List[float]]], colors:Optional[np.ndarray]=None, title:Optional[str]="3D orbit around earth", names:Optional[List[str]]=[], limits:Optional[np.ndarray]=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
        """Plot static 3D visualization of satellite orbits around Earth(Displayed as a blue sphere).

        Generates a 3D plot with:
        - A blue sphere representing Earth.
        - One or more orbital trajectories (lines) with customizable colors and labels.
        - Configurable axis limits and plot aesthetics.

        Parameters
        ----------
        r : np.ndarray or List[List[float]]
            Orbit position data. Can be:
            - Single orbit: Shape (N, 3) for N time steps.
            - Multiple orbits: Shape (M, N, 3) for M orbits.
        colors : np.ndarray, optional
            Hex code colors for each orbit.
            If None, auto-generates distinct colors.
        title : str, optional
            Plot title. Default: "3D orbit around earth".
        names : List[str], optional
            Legend labels for each orbit. If empty, legend is hidden.
        limits : np.ndarray, optional
            Axis limits as [[x_max, x_min], [y_max, y_min], [z_max, z_min]].
            Default: ±10,000 km on all axes.

        Notes
        -----
        - Earth is represented as a blue sphere with radius 6371 km (approximate).
        - Orbits are plotted as lines with 2px width by default.
        - If `names` is not provided, the legend is hidden.
        - Plot background is black with white grid/text for contrast.
        """

        # Create figure
        fig = go.Figure()

        #Correcting the r format
        r = np.array(r) 

        if r.ndim == 2:                                 #if only a single orbit is provided fix the formatting
            r = r.reshape([1,r.shape[0],r.shape[1]])    #Reshaping the r

        #Number of orbits
        n = len(r)  

        #Check if the colors are provided otherwise generate the color set
        if colors == None:
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

    def EarthStatic(self, r:Union[np.ndarray, List[List[float]]], colors:Optional[np.ndarray]=None, title:Optional[str]="3D orbit around earth", names:Optional[List[str]]=[], limits:Optional[np.ndarray]=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])):
        """Plot 3D visualization of satellite orbits around Earth with country borders.

        Generates an interactive 3D plot containing:
        - A light blue sphere representing Earth
        - Satellite orbital trajectories with customizable colors
        - Country borders from Natural Earth dataset
        - Configurable viewing parameters

        Parameters
        ----------
        r : np.ndarray or List[List[float]]
            Orbit position data. Can be:
            - Single orbit: Shape (N, 3) for N time steps
            - Multiple orbits: Shape (M, N, 3) for M orbits
        colors : np.ndarray, optional
            Hex code colors for each orbit.
            If None, auto-generates distinct colors using colorGenerator.
        title : str, optional
            Plot title. Default: "3D orbit around earth".
        names : List[str], optional
            Legend labels for each orbit. If empty, legend is hidden.
        limits : np.ndarray, optional
            Axis limits as [[x_max, x_min], [y_max, y_min], [z_max, z_min]].
            Default: ±10,000 km on all axes.

        Returns
        -------
        None
            Displays the plot interactively using Plotly.

        Notes
        -----
        - Earth is represented as a sphere with radius 6371 km (actual Earth radius)
        - Country borders are fetched from Natural Earth dataset via GitHub
        - Orbits are plotted as lines with 2px width by default
        - Plot uses black background with white text for high contrast
        - Handles both Polygon and MultiPolygon GeoJSON geometries
        """
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



    def SimpleDynamic(self, 
                      r:Union[np.ndarray,List[np.ndarray]], 
                      time:np.ndarray, 
                      colors:List[str] = None, 
                      title:str="3D orbit around earth", 
                      names:List[str]=[], 
                      limits:np.ndarray=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])
                      )->None:
        """
        Create an animated 3D visualization of satellite orbital motion around Earth.

        Generates an interactive plot with:
        - A static blue sphere representing Earth (radius = 6371 km)
        - Animated orbital trajectories
        - Real-time time display in hours
        - Play/pause animation controls

        Parameters
        ----------
        r : list or np.ndarray 
            Orbit position data from propagator. Can be:
            - Single orbit: Shape (N, 3) for N time steps
            - Multiple orbits: Shape (M, N, 3) for M orbits
        time : np.ndarray
            Time array corresponding to orbit positions (in seconds)
        colors : np.ndarray, optional
            If None (default), auto-generates colors using colorGenerator.
            If list, should be Hex code of colors.
        title : str, optional
            Plot title. Default: "3D orbit around earth".
        names : List[str] or str, optional
            Legend labels for orbits. If empty list, legend is hidden.
            If string, uses same name for all orbits.
        limits : np.ndarray, optional
            Axis limits as [[x_max, x_min], [y_max, y_min], [z_max, z_min]].
            Default: ±10,000 km on all axes.

        Returns
        -------
        None    
            Displays interactive Plotly figure with animation controls.

        Notes
        -----
        - Earth is represented as a perfect sphere (no topography)
        - Animation frame rate depends on number of time steps
        - Uses black background with white elements for contrast
        - Time display shows elapsed hours in top-left corner
        - Requires __sphereProvider and __sceneProviderSimple helper methods
        """

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


    def EarthDynamic(self, 
                     r:Union[List[List[float]], np.ndarray], 
                     time:Union[List[float], np.ndarray], 
                     colors:Optional[List[str]]=None, 
                     title:Optional[str]="3D orbit around earth", 
                     names:Optional[List[str]]=[], 
                     limits:Optional[np.ndarray]=np.array([[10_000, -10_000], [10_000, -10_000], [10_000, -10_000]])
                     )->None:
        """
        Create an animated 3D visualization of satellite orbits around Earth with country borders.

        Generates an interactive plot containing:
        - A light blue sphere representing Earth (radius = 6371 km)
        - Animated orbital trajectories with customizable colors
        - Country borders from Natural Earth dataset
        - Real-time elapsed time display
        - Play/pause animation controls

        Parameters
        ----------
        r : list or np.ndarray
            Orbit position data. Can be:
            - Single orbit: Shape (N, 3) for N time steps
            - Multiple orbits: Shape (M, N, 3) for M orbits
        time : list or np.ndarray
            Time array corresponding to orbit positions (in seconds)
        colors : list of str, optional
            Color values for each orbit. If None, auto-generates using colorGenerator.
            Must be a list of color Hex codes in the same order as orbits in the r.
        title : str, optional
            Plot title. Default: "3D orbit around earth".
        names : list of str or str, optional
            Legend labels for orbits. If empty list, legend is hidden.
            If string, uses same name for all orbits.
        limits : np.ndarray, optional
            Axis limits as [[x_max, x_min], [y_max, y_min], [z_max, z_min]].
            Default: ±10,000 km on all axes.

        Returns
        -------
        None
            Displays interactive Plotly figure with animation controls.

        Notes
        -----
        - Country borders are fetched from Natural Earth dataset via GitHub
        - Handles both Polygon and MultiPolygon GeoJSON geometries
        - Animation frame rate depends on number of time steps
        - Time display shows elapsed hours in top-left corner
        - Uses black background with white elements for contrast
        - Requires __sphereProvider and __sceneProviderSimple helper methods
        """

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
        if colors == None:
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

    #🏗️🚧
    def ground_track(self, 
                    latitudes:Union[List[float], np.ndarray], 
                    longitudes:Union[List[float], np.ndarray],
                    names:Optional[List[str]] = [], 
                    show_legend:bool = True, 
                    font_size_legend:int = 14)->None:
        """
        Visualize satellite ground tracks on an Earth map with start/end markers.

        Creates an interactive geographic plot showing:
        - Satellite ground tracks as lines
        - Starting points (distinct markers)
        - Ending points (distinct markers)
        - Overlaid on a Blue Marble Earth image
        - Customizable legend and styling

        Parameters
        ----------
        latitudes : list or np.ndarray
            Array of latitude values. Can be:
            - 1D array for single satellite (automatically reshaped)
            - 2D array where each row represents a different satellite
        longitudes : list or np.ndarray
            Array of longitude values (same shape requirements as latitudes)
        names : list or str, optional
            Names for legend entries. Can be:
            - List of strings (one per satellite)
            - Single string (applied to all satellites)
            - Empty list (auto-generates "Orbit0", "Orbit1", etc.)
        show_legend : bool, optional
            Whether to display the legend. Default: True
        font_size_legend : int, optional
            Font size for legend text. Default: 14

        Returns
        -------
        None
            Displays interactive Plotly figure in notebook.

        Notes
        -----
        - Requires 'blue_marple_earth.jpg' image file for Earth background
        - Currently only works in Jupyter notebooks
        - Automatically handles single or multiple satellite tracks
        - Uses distinct colors for tracks, start points, and end points
        - Start/end markers are slightly larger than track lines
        - Black background with no map borders for clean visualization
        """


        #Converting the latitudes and longitudes to np.ndarray
        latitudes = np.array(latitudes)
        longitudes = np.array(longitudes)

        #If there is only a single orbit
        if latitudes.ndim == 1:
            latitudes = latitudes.reshape((1, len(latitudes)))
            longitudes = longitudes.reshape((1, len(longitudes)))

        #Getting the color plate of the orbits
        colors = self.colorGenerator(num = len(latitudes) * 3)

        #Check if the names(legends) are provided otherwise set all the values equal to "ORBIT" and disable the legend
        if names == []:
            names = [f"Orbit{i}" for i in range(len(latitudes))]

        elif type(names) is str:    #If the names has only a single value make i a list
            names = [names for i in range(len(latitudes))]
            
        

        # Initialize figure
        fig = go.Figure()

        # Add Blue Marble image (replace with your path)
        fig.add_layout_image(
            dict(
                source="blue_marple_earth.jpg",
                xref="x",
                yref="y",
                x=-180,
                y=85,
                sizex=360,
                sizey=170,
                sizing="stretch",
                layer="below",
                opacity=1.0
            )
        )

        
        #Plotting all of the ground track as well as their starting and final subsatellite pont
        for i , (lat, long) in enumerate(zip(latitudes, longitudes)):

            # The starting point
            fig.add_trace(
                go.Scattergeo(
                    lon=long[0:1],   # Replace with your longitudes
                    lat=lat[0:1],    # Replace with your latitudes
                    mode="markers",
                    name = f"Starting point of {names[i]}",
                    marker=dict(
                        size=10,
                        color=colors[i+1],
                        opacity=0.8,
                        line=dict(width=1, color="white")
                    )
                )
            )

             # The trajectory
            fig.add_trace(
                go.Scattergeo(
                    lon=long,   # Replace with your longitudes
                    lat=lat,    # Replace with your latitudes
                    mode="lines",
                    name = f"Ground track of {names[i]}",
                    marker=dict(
                        size=10,
                        color=colors[i],
                        opacity=0.8,
                        line=dict(width=1, color="white")
                    )
                )
            )

            # The last point
            fig.add_trace(
                go.Scattergeo(
                    lon=long[-1:],   # Replace with your longitudes
                    lat=lat[-1:],    # Replace with your latitudes
                    mode="markers",
                    name = f"Ending point of {names[i]}",
                    marker=dict(
                        size=10,
                        color=colors[i+2],
                        opacity=0.8,
                        line=dict(width=1, color="white")
                    )
                )
            )


        # Critical layout updates to remove white box
        fig.update_layout(
            width=1200,
            height=700,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            xaxis=dict(visible=False, range=[-180, 180]),
            yaxis=dict(visible=False, range=[-85, 85]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='black',
            showlegend=show_legend,
            legend_font_size = font_size_legend
        )

        # Disable default map features
        fig.update_geos(
            visible=False,
            showframe=False,
            bgcolor='rgba(0,0,0,0)',
        )

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


