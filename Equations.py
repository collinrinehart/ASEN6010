import numpy as np
import numpy.linalg as la
import math

# Collin Rinehart
# ASEN5050 Equation definitions

#   Shortened way to input a 3D vector for manipulation
def vector(a, b, c):
    vec = np.array([[a, b, c]])
    return vec


#   Shortened way to calculate the magnitude/ norm of a vector
def mag(a):
    return la.norm(a)


#   Function to compute transformation of a vector from the xyz frame to the r theta h frame.
#   Inputs: the vector to transform, inclination, RAAN, AOP, and true anomaly.
def xyz_to_rthetah(Txyz, i, RAAN, w, trueanomaly):
    C1 = np.array([[math.cos(RAAN)*math.cos(w + trueanomaly)-math.sin(RAAN)*math.cos(i)*math.sin(w + trueanomaly), -math.cos(RAAN)*math.sin(w + trueanomaly) - math.sin(RAAN)*math.cos(i)*math.cos(w + trueanomaly), math.sin(RAAN)*math.sin(i)],
                  [math.sin(RAAN) * math.cos(w + trueanomaly) + math.cos(RAAN) * math.cos(i) * math.sin(w + trueanomaly), -math.sin(RAAN) * math.sin(w + trueanomaly) + math.cos(RAAN) * math.cos(i) * math.cos(w + trueanomaly), -math.cos(RAAN) * math.sin(i)],
                  [math.sin(i)*math.sin(w+trueanomaly), math.sin(i)*math.cos(w+trueanomaly), math.cos(i)]])
    T_rthetah = np.matmul(np.transpose(C1), np.transpose(Txyz))
    return T_rthetah


#   Function to compute transformation of a vector from the r theta h frame to the xyz frame.
#   Inputs: the vector to transform, inclination, RAAN, AOP, and true anomaly.
def rthetah_to_xyz(Trthetah, i, RAAN, w, trueanomaly):
    C1 = np.array([[math.cos(RAAN)*math.cos(w + trueanomaly)-math.sin(RAAN)*math.cos(i)*math.sin(w + trueanomaly), -math.cos(RAAN)*math.sin(w + trueanomaly) - math.sin(RAAN)*math.cos(i)*math.cos(w + trueanomaly), math.sin(RAAN)*math.sin(i)],
                  [math.sin(RAAN) * math.cos(w + trueanomaly) + math.cos(RAAN) * math.cos(i) * math.sin(w + trueanomaly), -math.sin(RAAN) * math.sin(w + trueanomaly) + math.cos(RAAN) * math.cos(i) * math.cos(w + trueanomaly), -math.cos(RAAN) * math.sin(i)],
                  [math.sin(i)*math.sin(w+trueanomaly), math.sin(i)*math.cos(w+trueanomaly), math.cos(i)]])
    print(C1)
    T_xyz = np.matmul(C1, np.transpose(Trthetah))
    return T_xyz


#   Function to compute transformation of a vector from the xyz frame to the pqw frame.
#   Inputs: the vector to transform, inclination, RAAN, and AOP.
def xyz_to_pqw(Txyz, i, RAAN, w):
    C2 = np.array([[math.cos(RAAN)*math.cos(w)-math.sin(RAAN)*math.cos(i)*math.sin(w), -math.cos(RAAN)*math.sin(w) - math.sin(RAAN)*math.cos(i)*math.cos(w), math.sin(RAAN)*math.sin(i)],
                  [math.sin(RAAN) * math.cos(w) + math.cos(RAAN) * math.cos(i) * math.sin(w), -math.sin(RAAN) * math.sin(w) + math.cos(RAAN) * math.cos(i) * math.cos(w), -math.cos(RAAN) * math.sin(i)],
                  [math.sin(i)*math.sin(w), math.sin(i)*math.cos(w), math.cos(i)]])
    Tpqw1 = np.matmul(np.transpose(C2), np.transpose(Txyz))
    return Tpqw1


#   Function to compute transformation of a vector from the pqw frame to the xyz frame.
#   Inputs: the vector to transform, inclination, RAAN, and AOP.
def pqw_to_xyz(Tpqw, i, RAAN, w):
    C2 = np.array([[math.cos(RAAN)*math.cos(w)-math.sin(RAAN)*math.cos(i)*math.sin(w), -math.cos(RAAN)*math.sin(w) - math.sin(RAAN)*math.cos(i)*math.cos(w), math.sin(RAAN)*math.sin(i)],
                  [math.sin(RAAN) * math.cos(w) + math.cos(RAAN) * math.cos(i) * math.sin(w), -math.sin(RAAN) * math.sin(w) + math.cos(RAAN) * math.cos(i) * math.cos(w), -math.cos(RAAN) * math.sin(i)],
                  [math.sin(i)*math.sin(w), math.sin(i)*math.cos(w), math.cos(i)]])
    Txyz = np.matmul(C2, np.transpose(Tpqw))
    return Txyz


#   Function to compute transformation of a vector from the r theta h frame to the pqw frame.
#   Inputs: the vector to transform and true anomaly.
def rthetah_to_pqw(Txyz, trueanomaly):
    C3 = np.array([[math.cos(trueanomaly), math.sin(trueanomaly), 0],
                  [-math.sin(trueanomaly), math.cos(trueanomaly), 0],
                  [0, 0, 1]])
    Tpqw2 = np.matmul(np.transpose(C3), np.transpose(Txyz))
    return Tpqw2


#   Function to compute transformation of a vector from the pqw from to the r theta h frame.
#   Inputs: the vector to transform and true anomaly.
def pqw_to_rthetah(Tpqw, trueanomaly):
    C3 = np.array([[math.cos(trueanomaly), math.sin(trueanomaly), 0],
                  [-math.sin(trueanomaly), math.cos(trueanomaly), 0],
                  [0, 0, 1]])
    Trthetah = np.matmul(C3, np.transpose(Tpqw))
    return Trthetah


#   Class definitions for calculating the classical orbital elements from a state vector.
#   Inputs: position and velocity vectors in the xyz frame and the gravitational parameter for the central body
#   The definitions themselves are self-explanatory.
class OrbitalElements:

    def __init__(self, r, v, mu):
        self.r = r
        self.v = v
        self.mu = mu
        self.h = np.cross(r, v)
        self.energy = 0.5 * (mag(v) ** 2) - (mu / mag(r))
        self.a = self.semi_major_axis()
        self.e = self.eccentricity()
        self.i = self.inclination()
        self.n_unit = self.line_of_nodes_unitvector()
        self.RAAN = self.RightAscension()
        self.w = self.Argument_of_Periapsis()
        self.trueanomaly = self.trueAnomaly()

    def semi_major_axis(self):
        a = -self.mu/(2*self.energy)
        return a

    def eccentricity(self):
        e = ((np.cross(self.v, self.h))/self.mu) - (self.r/mag(self.r))
        self.emag = mag(e)
        return e

    def inclination(self):
        i =math.acos(self.h[2]/mag(self.h))  # No sign check needed
        return i

    def line_of_nodes_unitvector(self):
        z = np.array([[0,0,1]])
        nvec = np.cross(z, self.h)
        n_unit = nvec/mag(nvec)
        return n_unit

    def RightAscension(self):
        RAAN = math.acos(self.n_unit[0, 0])
        if self.n_unit[0, 1] < 0:            # Sign Check
            RAAN = -RAAN
        return RAAN

    def Argument_of_Periapsis(self):
        var = np.dot(self.n_unit, np.transpose(self.e))
        w = math.acos(var/mag(self.e))
        if self.e[2] < 0:            # Sign check
            w = -w
        return w

    def trueAnomaly(self):
        var = np.dot(self.r, np.transpose(self.e))
        ta = math.acos(var / (mag(self.e)*mag(self.r)))
        if np.dot(self.r, np.transpose(self.v)) < 0:        # Sign Check
            ta = -ta
        return ta


def mean_motion(a, mu_body):
    n = math.sqrt(mu_body/(a**3))
    return n


def time_past_peri(n, e, E):
    """ Calculates time past periapsis [sec] from mean motion [rad/s], eccentricity, and eccentric anomaly [rad]"""
    tpp = (E - e*math.sin(E))/(n)
    return tpp


def trueA_to_E(trueA, e):
    """ Calculates true eccentric anomaly [rad] from true anomaly [rad]"""
    E = 2*math.atan(math.sqrt((1-e)/(1+e))*math.tan(trueA/2))
    return E


def E_to_trueA(E, e):
    """ Calculates true anomaly [rad] from eccentric anomaly [rad] and eccentricity"""
    TA = 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E/2))
    return TA


def kepler_iteration(tpp, a, e, mu_body):
    """ Function to used Newton's method to iterate over Kepler's equation and solve for eccentric anomaly [rad].
    Inputs: time past periapsis [sec], semi-major axis [km], eccentricity, central body grev. parameter.
    Outputs: eccentric anomaly [rad], number of iterations, mean motion [rad/s]"""
    n = math.sqrt(mu_body / (a ** 3))       # Mean motion calc
    M = n*tpp                               # Mean Anomoly Calc
    Ef = M                                  # setting initial guess for Ecc Anom, setting as mean anomaly
    iter_num = 0                            # Iteration counter
    tolerance = 0.00001                     # Tolerance for ending iteration

    while abs(M - (Ef - e * math.sin(Ef))) > tolerance:             # Newtons Iteration method
        Ef = Ef - (Ef - e*math.sin(Ef) - M)/(1 - e * math.cos(Ef))
        iter_num += 1
    return Ef, iter_num, n


