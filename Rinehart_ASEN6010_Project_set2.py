import math
import imageio as iio
from types import FrameType
import matplotlib.pyplot as plt
plt.style.use( 'dark_background' )
import numpy as np
from pathlib import Path

dist_handler = {
        'km'    : 1.0,
        'ER'    : 1 / 6378.0,
        'JR'    : 1 / 71490.0,
        'AU'    : 6.68459e-9,
        r'$\dfrac{km}{s}$': 1.0}

COLORS = [ 
	'm', 'deeppink', 'chartreuse', 'w', 'springgreen', 'peachpuff',
	'white', 'lightpink', 'royalblue', 'lime', 'aqua' ] * 100


def tilde(vec):
    #  Uses Numpy array column vectors to define a tilde matrix
    #  Returns a 3x3 NumPy array
    vec_tilde = np.array([[0, -float(vec[2]), float(vec[1])],
                           [float(vec[2]), 0, -float(vec[0])],
                           [-float(vec[1]), float(vec[0]), 0]])
    return vec_tilde


def mag(a):
    # Calculates the Euclidean Norm (i.e. magnitude) of a vector
    return np.linalg.norm(a)


def DCM_to_Quaternion(DCM):
    # Calculates quaternions from a DCM using Shepard's Method
    # Returns a 4x1 numpy array
    B0_sqrd = 0.25 * (1 + np.trace(DCM))
    B1_sqrd = 0.25 * (1 + 2 * DCM[0, 0] - np.trace(DCM))
    B2_sqrd = 0.25 * (1 + 2 * DCM[1, 1] - np.trace(DCM))
    B3_sqrd = 0.25 * (1 + 2 * DCM[2, 2] - np.trace(DCM))
    B = [B0_sqrd, B1_sqrd, B2_sqrd, B3_sqrd]

    if max(B) == B0_sqrd:
        B0 = math.sqrt(B0_sqrd)
        B1 = (DCM[1, 2] - DCM[2, 1])/(4*B0)
        B2 = (DCM[2, 0] - DCM[0, 2]) / (4 * B0)
        B3 = (DCM[0, 1] - DCM[1, 0]) / (4 * B0)

    elif max(B) == B1_sqrd:
        B1 = math.sqrt(B1_sqrd)
        B0 = (DCM[1, 2] - DCM[2, 1])/(4*B1)
        B2 = (DCM[0, 1] + DCM[1, 0]) / (4 * B1)
        B3 = (DCM[2, 0] + DCM[0, 2]) / (4 * B1)

    elif max(B) == B2_sqrd:
        B2 = math.sqrt(B2_sqrd)
        B0 = (DCM[2, 0] - DCM[0, 2]) / (4 * B2)
        B1 = (DCM[0, 1] + DCM[1, 0]) / (4 * B2)
        B3 = (DCM[1, 2] + DCM[2, 1]) / (4 * B2)

    elif max(B) == B3_sqrd:
        B3 = math.sqrt(B3_sqrd)
        B0 = (DCM[0, 1] - DCM[1, 0]) / (4 * B3)
        B1 = (DCM[2, 0] + DCM[0, 2]) / (4 * B3)
        B2 = (DCM[1, 2] + DCM[2, 1]) / (4 * B3)

    Q = np.array([[B0], [B1], [B2], [B3]])
    Q = Q/mag(Q)
    return Q


def Quaternion_to_DCM(Q):
    # Calculates the DCM from quaternion Q
    # Returns a 3x3 numpy array
    B0 = Q[0]
    B1 = Q[1]
    B2 = Q[2]
    B3 = Q[3]

    DCM11 = B0**2 + B1**2 - B2**2 - B3**2
    DCM12 = 2*(B1*B2 + B0*B3)
    DCM13 = 2*(B1*B3 - B0*B2)

    DCM21 = 2*(B1*B2 - B0*B3)
    DCM22 = B0**2 - B1**2 + B2**2 - B3**2
    DCM23 = 2*(B2*B3 + B0*B1)

    DCM31 = 2*(B1*B3 + B0*B2)
    DCM32 = 2*(B2*B3 - B0*B1)
    DCM33 = B0**2 - B1**2 - B2**2 + B3**2

    DCM = np.array([[DCM11, DCM12, DCM13], 
                    [DCM21, DCM22, DCM23], 
                    [DCM31, DCM32, DCM33]])

    return DCM


def DCM2MRP(DCM):
    # Calculates MRPs from a DCM. DCM must be a 3x3 numpy array
    # Returns a 3x1 numpy array
    Q = DCM_to_Quaternion(DCM)
    sigma1 = Q[1]/(1 + Q[0])
    sigma2 = Q[2]/(1 + Q[0])
    sigma3 = Q[3]/(1 + Q[0])
    sigma = check_for_shadow_set(np.array([sigma1, sigma2, sigma3]))
    return sigma


def MRP2DCM(sigma):
    # Calculates a DCM from an MRP set. sigma must be a 3x1 numpy array column vector
    # Returns a 3x3 numpy array
    DCM = np.identity(3) + (1/(1 + mag(sigma)**2)**2) * (8*tilde(sigma) @ tilde(sigma) - 4*(1-mag(sigma)**2)*tilde(sigma))
    return DCM


def Quaternion_to_MRP(Q):
    # Direct conversion from quaternion to MRP
    # returns 3x1 numpy array
    B0 = Q[0]
    B1 = Q[1]
    B2 = Q[2]
    B3 = Q[3]

    s1 = B1/(1+B0)
    s2 = B2/(1+B0)
    s3 = B3/(1+B0)

    MRP = np.array([[s1],[s2],[s3]])
    return MRP


def MRP_to_Quaternion(MRP):
    # Direct conversion from MRP to Quaternion
    # returns a 4x1 numpy array
    sigma_2 = np.linalg.norm(MRP)**2
    B0 = (1 - sigma_2)/(1 + sigma_2)
    B1 = 2*MRP[0]/(1 + sigma_2)
    B2 = 2*MRP[1]/(1 + sigma_2)
    B3 = 2*MRP[2]/(1 + sigma_2)

    Q = np.array([[B0],[B1],[B2],[B3]])
    return Q


def check_for_shadow_set(MRP):
    # Checks if the MRPs should be switched to the shadow set. If not, it returns itself
    if mag(MRP) > 1:
        shadowset = (-1/(mag(MRP)**2)) * MRP
        return shadowset
    else:
        return MRP


def mrp_subtraction(MRP3, MRP1):
    # Calculates the relative orientation between two frames expressed as 2 MRP sets, such that "mrp1 + mrp2 = mrp3"
    # Output is constrained to mag(sigma) < 1
    # Returns a 3x1 numpy array

    M1 = np.linalg.norm(MRP1)**2
    M3 = np.linalg.norm(MRP3)**2
    denom = (1 + (M1*M3) + np.dot(2*MRP1.T, MRP3))
    #print("denom = ", denom)

    if abs(denom) < 0.1:    # This check is added to switch an MRP if the addition equation is near singular
        if M1 > M3:
            MRP1 = (-1/(mag(MRP1)**2)) * MRP1
        else:
            MRP3 = (-1/(mag(MRP3)**2)) * MRP3
    
    MRP2 = ((1-M1)*MRP3 - (1-M3)*MRP1 + 2*np.cross(MRP3, MRP1, axis=0)) / (1 + (M1*M3) + np.dot(2*MRP1.T, MRP3)) # This is singular if the denominator goes to zero.. (-3 pts in homework, the above check was added)

    if np.linalg.norm(MRP2) > 1:
        MRP2 = (-1/(np.linalg.norm(MRP2)**2)) * MRP2
    
    return MRP2


def attitude_error(t, sigma_BN, b_w_BN, r_vec_N, v_vec_N, RN_funct, wRN_funct):
    # Calculates the attitude error and rate error based on current states and current reference
    dt = timestep()
    BN = MRP2DCM(sigma_BN)
    BR = BN @ RN_funct(t, r_vec_N, v_vec_N).T
    N_wRN = wRN_funct(t, r_vec_N, v_vec_N)
    sigma_BR = DCM2MRP(BR)  # calculates the attitude error
    
    b_w_RN = BN @ N_wRN
    b_wBR = b_w_BN - b_w_RN  # calculates rate error in body frame
    if t == 0:
        b_w_RN_dot = np.zeros((3,1))
    else:
        b_w_RN_dot = (wRN_funct(t, r_vec_N, v_vec_N)-wRN_funct(t-dt, r_vec_N, v_vec_N))/(dt)
    
    return sigma_BR, b_wBR, b_w_RN, b_w_RN_dot


def spotlight_reference(t, r_vec_N, v_vec_N):
    body_pointing_vec = np.array([[0],[0],[1]]) # point S/C +Z at target ground station ('no' error)
    body_clocking_vec = np.array([[0],[1],[0]]) # point S/C +Y at the inertial clocking vector (point error accumulates here)
    body_3rd_vec = np.cross(body_pointing_vec, body_clocking_vec, axisa=0, axisb=0, axisc=0)
    body_1st_vec = body_pointing_vec
    body_2nd_vec = np.cross(body_3rd_vec, body_1st_vec, axisa=0, axisb=0, axisc=0)

    RM = np.concatenate((body_1st_vec, body_2nd_vec, body_3rd_vec), axis = 1)   #M is the 'intermediate' reference frame

    r_groundsite_N, v_groundsite_N = groundsite(t)
    delta_r = r_groundsite_N - r_vec_N      # Vector pointing from S/C to groundsite in inertial frame
    delta_r_unit = delta_r/(mag(delta_r))   # normalized vector
    

    inertial_pointing_vec = delta_r_unit    # inertial pointing direction ('no' error)
    inertial_clocking_vec = v_vec_N/mag(v_vec_N)    # inertial clocking target (velocity direciton) (error accumulates here)
    M_3rd_vec = np.cross(inertial_pointing_vec, inertial_clocking_vec, axisa=0, axisb=0, axisc=0)
    M_1st_vec = inertial_pointing_vec
    M_2nd_vec = np.cross(M_3rd_vec, M_1st_vec, axisa=0, axisb=0, axisc=0)
    
    NM = np.concatenate((M_1st_vec, M_2nd_vec, M_3rd_vec), axis = 1)   #M is the 'intermediate' reference frame
    RN = RM @ NM.T # full defines the pointing reference frame
   # RN = NM
    #print(NM @ body_pointing_vec)
    return RN


def obj_track_reference(t, r_vec_N, v_vec_N):
    body_pointing_vec = np.array([[0],[0],[1]]) # point S/C +Z at target ground station ('no' error)
    body_clocking_vec = np.array([[0],[1],[0]]) # point S/C +Y at the inertial clocking vector (point error accumulates here)
    body_3rd_vec = np.cross(body_pointing_vec, body_clocking_vec, axisa=0, axisb=0, axisc=0)
    body_1st_vec = body_pointing_vec
    body_2nd_vec = np.cross(body_3rd_vec, body_1st_vec, axisa=0, axisb=0, axisc=0)

    RM = np.concatenate((body_1st_vec, body_2nd_vec, body_3rd_vec), axis = 1)   #M is the 'intermediate' reference frame

    r_obj_N, v_obj_N = object_states(t)
    delta_r = r_obj_N - r_vec_N      # Vector pointing from S/C to groundsite in inertial frame
    delta_r_unit = delta_r/(mag(delta_r))   # normalized vector

    inertial_pointing_vec = delta_r_unit    # inertial pointing direction ('no' error)
    inertial_clocking_vec = v_vec_N/mag(v_vec_N)    # inertial clocking target (velocity direciton) (error accumulates here)
    
    M_3rd_vec = np.cross(inertial_pointing_vec, inertial_clocking_vec, axisa=0, axisb=0, axisc=0)
    M_1st_vec = inertial_pointing_vec
    M_2nd_vec = np.cross(M_3rd_vec, M_1st_vec, axisa=0, axisb=0, axisc=0)

    NM = np.concatenate((M_1st_vec, M_2nd_vec, M_3rd_vec), axis = 1)   #M is the 'intermediate' reference frame
    RN = RM @ NM.T # full defines the pointing reference frame
    return RN


def attitude_error_spotlight(t, sigma_BN, b_w_BN, r_vec_N_list, v_vec_N_list):
    # Calculates the attitude error and rate error based on current states and current reference
    
    RN = spotlight_reference(t, r_vec_N_list[-1], v_vec_N_list[-1])
    if len(r_vec_N_list) == 2:
        RN_m1 = spotlight_reference(t, r_vec_N_list[-2], v_vec_N_list[-2])
        RN_m2 = RN_m1
    elif len(r_vec_N_list) == 1:
        RN_m1 = RN
        RN_m2 = RN
    else:
        RN_m1 = spotlight_reference(t, r_vec_N_list[-2], v_vec_N_list[-2])
        RN_m2 = spotlight_reference(t, r_vec_N_list[-3], v_vec_N_list[-3])

     # Calculates the rotation rate of the GMO-pointing reference frame in inertial frame components
    dC = (RN - RN_m1)
    w_tilde = -dC @ RN.T
    R_w_RcN = np.array([[-w_tilde[1, 2]], [w_tilde[0, 2]], [-w_tilde[0, 1]]])
    N_wRN =  RN.T @ R_w_RcN

    dC2 = (RN_m1 - RN_m2)
    w_tilde2 = -dC2 @ RN_m1.T
    R_w_RcN_m1 = np.array([[-w_tilde2[1, 2]], [w_tilde2[0, 2]], [-w_tilde2[0, 1]]])
    N_wRN_m1 =  RN_m1.T @ R_w_RcN_m1


    dt = timestep()
    BN = MRP2DCM(sigma_BN)
    BR = BN @ RN.T
    #N_wRN = wRN_funct(t, r_vec_N, v_vec_N)
    sigma_BR = DCM2MRP(BR)  # calculates the attitude error
    b_w_RN = BN @ N_wRN
    b_wBR = b_w_BN - b_w_RN  # calculates rate error in body frame
    #if t == 0:
     #   b_w_RN_dot = np.zeros((3,1))
    #else:
    b_w_RN_dot = BN @ ((N_wRN-N_wRN_m1)/(dt))
    return sigma_BR, b_wBR, b_w_RN, b_w_RN_dot


def attitude_error_obj_track(t, sigma_BN, b_w_BN, r_vec_N_list, v_vec_N_list):
    # Calculates the attitude error and rate error based on current states and current reference
    
    RN = obj_track_reference(t, r_vec_N_list[-1], v_vec_N_list[-1])
    if len(r_vec_N_list) == 2:
        RN_m1 = obj_track_reference(t, r_vec_N_list[-2], v_vec_N_list[-2])
        RN_m2 = RN_m1
    elif len(r_vec_N_list) == 1:
        RN_m1 = RN
        RN_m2 = RN
    else:
        RN_m1 = obj_track_reference(t, r_vec_N_list[-2], v_vec_N_list[-2])
        RN_m2 = obj_track_reference(t, r_vec_N_list[-3], v_vec_N_list[-3])

     # Calculates the rotation rate of the GMO-pointing reference frame in inertial frame components
    dC = (RN - RN_m1)
    w_tilde = -dC @ RN.T
    R_w_RcN = np.array([[-w_tilde[1, 2]], [w_tilde[0, 2]], [-w_tilde[0, 1]]])
    N_wRN =  RN.T @ R_w_RcN

    dC2 = (RN_m1 - RN_m2)
    w_tilde2 = -dC2 @ RN_m1.T
    R_w_RcN_m1 = np.array([[-w_tilde2[1, 2]], [w_tilde2[0, 2]], [-w_tilde2[0, 1]]])
    N_wRN_m1 =  RN_m1.T @ R_w_RcN_m1


    dt = timestep()
    BN = MRP2DCM(sigma_BN)
    BR = BN @ RN.T
    #N_wRN = wRN_funct(t, r_vec_N, v_vec_N)
    sigma_BR = DCM2MRP(BR)  # calculates the attitude error
    b_w_RN = BN @ N_wRN
    b_wBR = b_w_BN - b_w_RN  # calculates rate error in body frame
    #if t == 0:
     #   b_w_RN_dot = np.zeros((3,1))
    #else:
    b_w_RN_dot = BN @ ((N_wRN-N_wRN_m1)/(dt))
    
    return sigma_BR, b_wBR, b_w_RN, b_w_RN_dot


def Rs_N(t, r_vec_N = None, v_vec_N = None):
    # Calculates the 'Inertial-to-Sun-Pointing Frame' DCM
    # Defined as the body [0.707, 0.707, 0] body vector pointed at sun
    # +Z body vector aligned with inertial +Z
    angle = math.radians(45)
    RsN = np.array([[ math.cos(angle), math.sin(angle), 0], 
                    [-math.sin(angle), math.cos(angle), 0], 
                    [               0,               0, 1]])
    return RsN


def RsN_rate(t, r_vec_N = None,  v_vec_N = None):
    # Returns the Rotation rate vector for the Sun-pointing frame in the inertial frame
    # Sun-pointing frame defined as an inertial frame for this project
    w_RsN = np.array([[0], [0], [0]])
    return w_RsN


def H_N(r_vec_N, v_vec_N):
    h_vec_N = np.cross(r_vec_N, v_vec_N, axisa=0, axisb=0, axisc=0)
 
    r_unit_N = r_vec_N/np.linalg.norm(r_vec_N)
    v_unit_N = v_vec_N/np.linalg.norm(v_vec_N)
    h_unit_N = h_vec_N/np.linalg.norm(h_vec_N)

    r_unit_N.reshape(3,1)
    v_unit_N.reshape(3,1)
    h_unit_N.reshape((3,1))
  
    NH = np.concatenate((r_unit_N, v_unit_N, h_unit_N), axis = 1)
    HN = NH.T
    return HN


def Rn_N(t, r_vec_N = None, v_vec_N = None):
    # Calculates the 'Inertial-to-Nadir pointing Frame' DCM
    # Defined as the body [0, 0, -1] body vector pointed nadir
    # body [0, 1, 0] pointed in velocity direction

    HN = H_N(r_vec_N, v_vec_N)

    angle = math.radians(90)
    Rot = np.array([[math.cos(angle), 0, -math.sin(angle)], 
                    [              0, 1,                0],
                    [math.sin(angle), 0,  math.cos(angle)]])
    
    RnN = Rot @ HN
    return RnN


def RnN_rate(t, r_vec_N = None, v_vec_N = None):
    # Returns the Rotation rate vector for the nadir-pointing frame in the inertial frame
    mu = 3.986004415E5  # Mu Earth [km^3/s^2]
    theta_dot = math.sqrt(mu/(mag(r_vec_N)**3))  # [rad/s] 
    w_Rn_H = np.array([[0], [0], [theta_dot]])
    HN = H_N(r_vec_N, v_vec_N)
    w_Rn_N = HN.T @ w_Rn_H
    return w_Rn_N


def generate_reference(t):
    # Calculates the 'Inertial-to-Hill Frame' DCM as a reference trajectory
    RAAN_LMO = math.radians(20)
    i_LMO = math.radians(30) 
    theta_dot_LMO = 10 * 0.000884797 # [rad/s]
    theta_LMO = math.radians(60) + theta_dot_LMO * t 

    HN = np.array([[math.cos(RAAN_LMO)*math.cos(theta_LMO)-math.sin(RAAN_LMO)*math.cos(i_LMO)*math.sin(theta_LMO),
                        math.sin(RAAN_LMO) * math.cos(theta_LMO) + math.cos(RAAN_LMO) * math.cos(i_LMO) * math.sin(theta_LMO),
                        math.sin(i_LMO)*math.sin(theta_LMO)],
                    [-math.cos(RAAN_LMO)*math.sin(theta_LMO) - math.sin(RAAN_LMO)*math.cos(i_LMO)*math.cos(theta_LMO),
                        -math.sin(RAAN_LMO) * math.sin(theta_LMO) + math.cos(RAAN_LMO) * math.cos(i_LMO) * math.cos(theta_LMO),
                        math.sin(i_LMO)*math.cos(theta_LMO)],
                    [math.sin(RAAN_LMO)*math.sin(i_LMO), -math.cos(RAAN_LMO) * math.sin(i_LMO), math.cos(i_LMO)]])

    H_w_HN = np.array([[0], [0], [theta_dot_LMO]]) # expressed in reference frame components
    w_HN = HN.T @ H_w_HN
    w_HN_dot = np.array([[0], [0], [0]]) # assumes circular orbit
    return HN, w_HN, w_HN_dot # returns stuff expressed in the N frame


def object_states(time):
    # Calculates the Inertial R and V vectors of a given spacecraft/object in orbit
    r_earth = 6378
    mu = 3.986004415E5  # Mu Earth [km^3/s^2]
    obj_alt = 600
    r_obj = r_earth + obj_alt
    RAAN_obj = math.radians(0)
    i_obj = math.radians(0)
    theta_dot_obj = math.sqrt(mu/(r_obj**3))  # [rad/s]   
    theta_obj_t = math.radians(250) + theta_dot_obj * time

    r_orbitframe_obj = np.array([[r_obj], [0], [0]])
    v_orbitframe_obj = np.array([[0], [math.sqrt(mu / r_obj)], [0]])

    # Calculates the [NH] DCM for spacecraft/object

    NH_obj = np.array([[math.cos(RAAN_obj) * math.cos(theta_obj_t) - math.sin(RAAN_obj) * math.cos(i_obj) * math.sin(theta_obj_t),
                         -math.cos(RAAN_obj) * math.sin(theta_obj_t) - math.sin(RAAN_obj) * math.cos(i_obj) * math.cos(theta_obj_t),
                         math.sin(RAAN_obj) * math.sin(i_obj)],
                        [math.sin(RAAN_obj) * math.cos(theta_obj_t) + math.cos(RAAN_obj) * math.cos(i_obj) * math.sin(theta_obj_t),
                         -math.sin(RAAN_obj) * math.sin(theta_obj_t) + math.cos(RAAN_obj) * math.cos(i_obj) * math.cos(theta_obj_t),
                         -math.cos(RAAN_obj) * math.sin(i_obj)],
                        [math.sin(i_obj) * math.sin(theta_obj_t), math.sin(i_obj) * math.cos(theta_obj_t), math.cos(i_obj)]])
    
    r_inertialframe_obj = NH_obj @ r_orbitframe_obj
    v_inertialframe_obj = NH_obj @ v_orbitframe_obj

    return r_inertialframe_obj, v_inertialframe_obj


def groundsite(time):
    #boulder_site_lat = 40.009808 * math.pi/180      #[rads]
    #boulder_site_long = -105.244104 * math.pi/180   #[rads]
    boulder_site_lat = 90 * math.pi/180      #[rads] North Pole
    boulder_site_long = 0 * math.pi/180   #[rads] North Pole
    
    r_earth = 6378 #[km]
    r_vec_gs = np.array([[r_earth],[0],[0]])
    v_unit_gs = np.array([[0],[1],[0]])

    rot_long = np.array([[math.cos(boulder_site_long),  math.sin(boulder_site_long), 0],
                          [-math.sin(boulder_site_long), math.cos(boulder_site_long), 0],
                          [0,                            0,                          1]])

    rot_lat = np.array([[math.cos(boulder_site_lat), 0, -math.sin(boulder_site_lat)],
                        [0,                          1,                           0],
                        [math.sin(boulder_site_lat), 0, math.cos(boulder_site_lat)]])

    r_vec_gs_ECF = rot_lat @ rot_long @ r_vec_gs
    v_unit_gs_ECF = rot_lat @ rot_long @ v_unit_gs

    earth_lambda_0 = 0 # [rads]
    earth_rot_rate = 7.292115E-5    #[rad/s]
    earth_lambda = earth_lambda_0 + earth_rot_rate * time

    rot_lambda = np.array([[math.cos(earth_lambda), math.sin(earth_lambda), 0],
                          [-math.sin(earth_lambda), math.cos(earth_lambda), 0],
                          [0,                    0,                         1]])

    r_groundsite_N = rot_lambda @ r_vec_gs_ECF
    v_groundsite_N = r_earth*math.cos(boulder_site_lat)*(rot_lambda @ v_unit_gs_ECF)
    
    return r_groundsite_N, v_groundsite_N


def evaluate_control_reference(time, state, w_gframe, G_list, num_cmgs, mode, r_vec_N_list, v_vec_N_list, L_h_dump, control_reference=None):
    I_list = inertia_properties()
    I = I_list[0]
    Iws = I_list[3][0,0] 
    w_t = w_gframe[:,1]
    w_g = w_gframe[:,2]
    ext_torque_drag = 0.0* np.ones((3,1))
    ext_torque = ext_torque_drag + L_h_dump
    sigmaBN = state[0:3]
    #BN = MRP2DCM(sigmaBN)
    b_wBN = state[3:6]
    OMEGA = state[6:6+num_cmgs]

    r_vec_N = state[6+num_cmgs:6+num_cmgs+3]
    v_vec_N = state[6+num_cmgs+3:6+num_cmgs+6]
    
    # Control Gains
    K = 40 * 0.00555555555  # [Nm]
    P = 20 * 0.1666666666  # [Nm/s]

    # Calculate gryoscopic feed forward terms for later use
    gyro_terms = np.array([[0, 0, 0]])
    for cmg in range(num_cmgs):
        gyro_terms = gyro_terms +  (Iws*OMEGA[cmg]*w_g[cmg]*G_list[1][:,cmg] + Iws*OMEGA[cmg]*w_t[cmg]*G_list[2][:,cmg]).T
    
    # Evaluate which control reference to use at the current time step
    if control_reference == 'Full-Mission':
        r_obj_N, v_obj_N = object_states(time)
        r_groundsite_N, v_groundsite_N = groundsite(time)
        angle_to_obj = math.degrees(math.acos(np.dot(r_vec_N.T, r_obj_N)/(mag(r_vec_N)*mag(r_obj_N))))
        angle_to_gs = math.degrees(math.acos(np.dot(r_vec_N.T, r_groundsite_N)/(mag(r_vec_N)*mag(r_groundsite_N))))
        
        if mode == 'Initialization' or  mode == 'De-tumble':
            mission_mode = 'De-tumble'
            control_reference = mission_mode
            if mag(b_wBN) < 0.001:
                #print(mag(b_wBN))
                mission_mode = 'Inertial-Hold'
                control_reference = mission_mode
        else:
            # Evaluate which mission mode/reference frame is desired
            if angle_to_gs < 20:      # If GMO spacecraft is visible by LMO spacecraft   TODO: redefine this condition
                control_reference = 'Spotlight'
            elif angle_to_obj < 20:      # If GMO spacecraft is visible by LMO spacecraft   TODO: redefine this condition
                control_reference = 'Object-Track'
            elif r_vec_N[1] > -2793:   # If LMO spacecraft is 'in the sun'. Sun is defined as in the +Y inertial direciton.
                control_reference = 'Sun-Track'
            else:   # if neither of these are true, point at Mars and do science     TODO: redefine this condition
                control_reference = 'Nadir-Point'
            
        # elif condition:      # If GMO spacecraft is visible by LMO spacecraft   TODO: redefine this condition
                #control_reference = 'Science'
            


    if control_reference == 'None': 
        Lr = np.zeros((3,1))
        sigmaBR = np.zeros((3,1))
        b_wBR = np.zeros((3,1))
        b_w_RN = np.zeros((3,1))
        b_w_RN_dot = np.zeros((3,1))
        gyro_terms = np.zeros((3,1))
        mission_mode = "Idle"

    elif control_reference == 'Regulation':
        sigmaBR = sigmaBN
        b_wBR = b_wBN
        b_w_RN = np.zeros((3,1))
        b_w_RN_dot = np.zeros((3,1))
        mission_mode = control_reference

    elif control_reference == 'Spotlight':
        sigmaBR, b_wBR, b_w_RN, b_w_RN_dot = attitude_error_spotlight(time, sigmaBN, b_wBN, r_vec_N_list, v_vec_N_list)
        mission_mode = control_reference

    elif control_reference == 'Object-Track':
        sigmaBR, b_wBR, b_w_RN, b_w_RN_dot = attitude_error_obj_track(time, sigmaBN, b_wBN, r_vec_N_list, v_vec_N_list)
        mission_mode = control_reference

    elif control_reference == 'Sun-Track':
        sigmaBR, b_wBR, b_w_RN, b_w_RN_dot = attitude_error(time, sigmaBN, b_wBN, r_vec_N, v_vec_N, Rs_N, RsN_rate)
        mission_mode = control_reference

    elif control_reference == 'Nadir-Point':
        sigmaBR, b_wBR, b_w_RN, b_w_RN_dot = attitude_error(time, sigmaBN, b_wBN, r_vec_N, v_vec_N, Rn_N, RnN_rate)
        mission_mode = control_reference

    elif control_reference == 'De-tumble':
        sigmaBR = np.zeros((3,1))
        b_wBR = b_wBN
        b_w_RN = np.zeros((3,1))
        b_w_RN_dot = np.zeros((3,1))
        mission_mode = control_reference

    elif control_reference == 'Inertial-Hold':
        sigmaBR = np.zeros((3,1))
        b_wBR = b_wBN
        b_w_RN = np.zeros((3,1))
        b_w_RN_dot = np.zeros((3,1))
        mission_mode = control_reference

    #else:
       # mission_mode = control_reference
        #RN, N_w_RN, N_w_RN_dot = generate_reference(time)


    #sigmaBR, b_wBR = attitude_error(time, sigmaBN, b_wBN, RN, N_w_RN)
    #b_w_RN = BN @ N_w_RN
    # b_w_RN_dot = BN @ N_w_RN_dot
    
    Lr = -K * sigmaBR - P * b_wBR + I @ (b_w_RN_dot - tilde(b_wBN) @ b_w_RN) + tilde(b_w_RN) @ I @ b_wBN - ext_torque + gyro_terms.reshape(3,1)
    
    return Lr, sigmaBR, b_wBR, b_w_RN, mission_mode


def inertia_properties():
    # Contains the inertia configuration for the spacecraft, Gimbals, and Wheels
    # Spacecraft inertia in body frame

    G_0 = load_spacecraft_configuration()

    I_s = np.array([[10, 0, 0],
                    [0, 5, 0],
                    [0, 0, 7.5]])   # [kg*m^2]

    # Gimbal inertia in Gimbal frame
    I_G = np.array([[.0, 0, 0],
                    [0, .0, 0],
                    [0, 0, .0]])    # [kg*m^2]

    # wheel inertia in wheel frame, note due to symmetry w_I_W = g_I_W
    I_W = np.array([[.0005, 0, 0],
                    [0, .0002, 0],
                    [0, 0, .0002]]) # [kg*m^2]

    # combined wheel and gimbal inertia, expressed in gimbal frame
    J = I_G + I_W

    # total spacecraft inertia as a function of gimbal states
    terms = np.zeros((3,3))
    for i in range(len(G_0)):
        terms = terms + (G_0[i] @ J @ G_0[i].T)

    I_tot = I_s + terms

    return [I_tot, I_s, I_G, I_W, J]


def update_gimbal_frames(G_0, gamma_t, gamma_zeros):
    # Updates gimbal frame DCMs using an array of N number of initial gimbal frame DCMs (G_0, 3x3), the list of current gimbal angles, and gimbal initial angles
    # Returns N number of 3x3 numpy arrays
    g_frame_next = []
    
    for index, frame in enumerate(G_0):
        g_s = (math.cos(gamma_t[index]-gamma_zeros[index])* frame[:,0] + math.sin(gamma_t[index]-gamma_zeros[index])* frame[:,1])/mag(math.cos(gamma_t[index]-gamma_zeros[index])* frame[:,0] + math.sin(gamma_t[index]-gamma_zeros[index])* frame[:,1])
        g_t = (-math.sin(gamma_t[index]-gamma_zeros[index])* frame[:,0] + math.cos(gamma_t[index]-gamma_zeros[index])* frame[:,1])/mag( -math.sin(gamma_t[index]-gamma_zeros[index])* frame[:,0] + math.cos(gamma_t[index]-gamma_zeros[index])* frame[:,1])
        g_g = frame[:,2]
        newframe = np.column_stack([g_s, g_t, g_g])
        g_frame_next.append(newframe)

    return g_frame_next


def g_frames_2_g_mats(gimbal_frames):
    # Rearranges gimbal frame DCMs into the form of Gs, Gt, and Gg matrices 
    num_cmgs = len(gimbal_frames)
    Gs = np.array(np.column_stack([gimbal_frames[i][:,0].T for i in range(num_cmgs)]))
    Gt = np.array(np.column_stack([gimbal_frames[i][:,1].T for i in range(num_cmgs)]))
    Gg = np.array(np.column_stack([gimbal_frames[i][:,2].T for i in range(num_cmgs)]))

    return [Gs, Gt, Gg]


def g_mats_2_g_frames(G_list):
    # Rearranges Gs, Gt, and Gg matrices into the N number of gimbal frame DCMs
    num_cmgs = len(G_list[0][0,:])
    frame_list = []
    for i in range(num_cmgs):
        frame_s = G_list[0][:,i]
        frame_t = G_list[1][:,i]
        frame_g = G_list[2][:,i]
        frame = np.array(np.column_stack([frame_s, frame_t, frame_g]))
        frame_list.append(frame)
    return frame_list


def load_spacecraft_configuration():
    # defines VSCMG configuration with gimbal angles of zero. Column 1 is gs, column 2 is gt and column 3 is gg for each CMG
    # Change these values or add complete matrices to change the VSCMG configuration in the spacecraft

    # The configuration below describes 4x reaction wheels in a 45 deg pyrimidal configuration
    BG_0 = np.array([[math.cos(math.pi/4), 0, -math.sin(math.pi/4)],
                      [0,1,0],
                      [math.sin(math.pi/4), 0, math.cos(math.pi/4)]])

    BG_1 = np.array([[-math.cos(math.pi/4), 0, math.sin(math.pi/4)],
                      [0,1,0],
                      [math.sin(math.pi/4), 0, math.cos(math.pi/4)]])

    BG_2 = np.array([[0,-1, 0],
                      [math.cos(math.pi/4), 0, -math.sin(math.pi/4)],
                      [math.sin(math.pi/4), 0, math.cos(math.pi/4)]])

    BG_3 = np.array([[0,1, 0],
                      [-math.cos(math.pi/4), 0, math.sin(math.pi/4)],
                      [math.sin(math.pi/4), 0, math.cos(math.pi/4)]])
    
    return BG_0, BG_1, BG_2, BG_3


def load_initial_conditions():
    # Spacecraft initial r and v vectors
    #position_0 = [np.array([[-1303.301557], [-3287.245782], [5998.429856]])] #initial position vector (ECI)
    #velocity_0 = [np.array([[0.449126925], [6.578003736], [3.699223314]])] #initial position vector (ECI)
    position_0 = [np.array([[0], [0], [6962]])] #initial position vector (ECI)
    velocity_0 = [np.array([[0], [7.56], [0]])] #initial position vector (ECI)

    # Spacecraft Body Initial Conditions
    mrp0 = [np.array([[.4],[0.3],[-0.3]])] #initial attitude as a MRP
    b_w_BN_0 = [(math.pi / 180) * np.array([[0.5], [0.05], [0.5]])]  # Initial angular velocity in deg/s, expressed in body frame and converted to rad/s

    #mrp0 = [np.array([[0],[0],[0]])] #initial attitude as a MRP
    #b_w_BN_0 = [(math.pi / 180) * np.array([[0], [0.0], [0.5]])]  # Initial angular velocity in deg/s, expressed in body frame and converted to rad/s

    # Gimbal Initial conditions and Load initial spacecraft configuration (gimbal angles = 0)
    gimbal_frames_0 = load_spacecraft_configuration()
    num_cmgs = len(gimbal_frames_0)                       # determines the number of CMGs used from the spacecraft configuration. 
                                                                # Used to build initial condition matrices generally

    # Wheel initial Conditions
    OMEGA_0 = [np.zeros((num_cmgs, 1))]
    #OMEGA_0 = [np.array([[10],[10],[-10],[-10]])]
    OMEGA_0 = [np.array([[50],[50],[-50],[-50]])]

    return mrp0, b_w_BN_0, OMEGA_0, gimbal_frames_0, num_cmgs, position_0, velocity_0


def subservo(state, des_OMEGA_dot, w_gframe, I_list, num_cmgs):
    #K_gam = 0.5  # gimbal gain [1/sec]

    I_ws = I_list[3][0,0]
    #J_s = I_list[4][0,0]
    #J_t = I_list[4][1,1]
    #J_g = I_list[4][2,2]
    #w_s = w_gframe[:,0]
    #w_t = w_gframe[:,1]

    OMEGA = state[6:6+num_cmgs]
    
    rw_max_torque = .040 #Nm
    speed_limits = 500
    u_s = (I_ws*des_OMEGA_dot)
    
    for i in range(num_cmgs):
        if abs(OMEGA[i]) > speed_limits and u_s[i] > 0:
            u_s[i] = 0
        elif abs(u_s[i]) > rw_max_torque:
            u_s[i] = math.copysign(rw_max_torque, u_s[i])
    #print(u_s)

    
    return u_s


def steering_law(t, state, Lr, I_list, w_gframe, G_list, b_w_RN, num_cmgs):
    Iws = I_list[3][0,0] 
    Js = I_list[4][0,0]
    Jt = I_list[4][1,1]
    Jg = I_list[4][2,2]

    OMEGA = state[6:6+num_cmgs]
    w_s = w_gframe[:,0]
    w_t = w_gframe[:,1]

    Gs = G_list[0]
    Gt = G_list[1]

    D0 = np.zeros((3, num_cmgs))
    D1 = np.zeros((3, num_cmgs))
    D2 = np.zeros((3, num_cmgs))
    D3 = np.zeros((3, num_cmgs))
    D4 = np.zeros((3, num_cmgs))

    for cmg in range(num_cmgs):
        D0[:, cmg] = Iws*Gs[:,cmg]
        D1[:, cmg] = (Iws*OMEGA[cmg] + 0.5*Js*w_s[cmg])*Gt[:,cmg] + 0.5*Js*w_t[cmg]*Gs[:,cmg]
        D2[:, cmg] = 0.5*Jt*(w_t[cmg]*Gs[:,cmg] + w_s[cmg]*Gt[:,cmg])
        D3[:, cmg] = Jg*(w_t[cmg]*Gs[:,cmg] - w_s[cmg]*Gt[:,cmg])
        D4[:, cmg] = list(0.5*(Js-Jt)*(Gs[:,cmg].reshape((3,1)) @ Gt[:,cmg].reshape((1,3)) @ b_w_RN + Gt[:,cmg].reshape((3,1)) @ Gs[:,cmg].reshape((1,3)) @ b_w_RN))

    D = D1 - D2 + D3 + D4
    Q = np.block([D0, D])

    # Control portion of steering law implementation
    W_s = 1
    W_g = 0
    W_s_matrix = W_s * np.identity(num_cmgs)
    W_g_matrix = W_g * np.identity(num_cmgs)
    W = np.block([[W_s_matrix, np.zeros((num_cmgs, num_cmgs))],
                [np.zeros((num_cmgs, num_cmgs)), W_g_matrix]]) 

    eta_dot_control = W @ Q.T @ np.linalg.inv(Q @ W @ Q.T) @ (-Lr)
    
    OMEGA_dot_des = eta_dot_control[0:num_cmgs]
    return OMEGA_dot_des


def ODE(state_vector, control_vector, inertia_list, num_cmgs, G_0, L_h_dump,):
    # Ordinary differential equations for States for our system/spacecraft
    mu_earth = 3.986004415E5
    #Extract necessary parameters for later calculation 
    I = inertia_list[0]             # total inertia
    Iws = inertia_list[3][0,0] 

    # extract states
    x = float(state_vector[6+num_cmgs])
    y = float(state_vector[6+num_cmgs+1])
    z = float(state_vector[6+num_cmgs+2])

    vx = state_vector[6+num_cmgs+3]
    vy = state_vector[6+num_cmgs+4]
    vz = state_vector[6+num_cmgs+5]

    sigmaBN = state_vector[0:3]
    w = state_vector[3:6]
    OMEGA = state_vector[6:6+num_cmgs]

    Gs, Gt, Gg = g_frames_2_g_mats(G_0)

    # Calculate the S/C rates as seen in the gimbal frames
    #w_s = [Gs[:,i].T @ w for i in range(num_cmgs)]
    w_t = [Gt[:,i].T @ w for i in range(num_cmgs)]
    w_g = [Gg[:,i].T @ w for i in range(num_cmgs)]

    # Control Torques
    u = control_vector
    u_s = control_vector[0:num_cmgs]

    #External torque on the vehicle
    L_ext = 0.0* np.ones((3,1))
    L = L_ext + L_h_dump

    # Orbit differential equations
    xdot = vx
    vxdot = -(mu_earth*x) / (x**2 + y**2 + z**2)**(3/2)
    ydot = vy
    vydot = -(mu_earth*y) / (x**2 + y**2 + z**2)**(3/2) 
    zdot = vz
    vzdot = -(mu_earth * z) / (x**2 + y**2 + z**2) ** (3 / 2) 

    pos_dot = np.array([[xdot], [ydot], [zdot]]).reshape(3,1)
    vel_dot = np.array([[vxdot], [vydot], [vzdot]]).reshape(3,1)

    # MRP Kinematic Differential Equations
    s_tilde = tilde(sigmaBN)
    s2 = mag(sigmaBN)**2
    s_dot = 0.25*((1 - s2)*np.identity(3) + 2*s_tilde + 2*sigmaBN @ sigmaBN.T) @ w

    # VSCMG coupled equations of motions, build M matrix
    row1 = np.concatenate((I, Iws*Gs), axis = 1)
    row3 = np.concatenate((Iws*Gs.T, Iws*np.identity(num_cmgs)), axis = 1)
    M = np.concatenate((row1, row3), axis=0)

    # row1 = np.concatenate((I, Jg*Gg, Iws*Gs), axis = 1)
    # row2 = np.concatenate((Jg*Gg.T, Jg*np.identity(num_cmgs), np.zeros((num_cmgs,num_cmgs))), axis = 1)
    # row3 = np.concatenate((Iws*Gs.T, np.zeros((num_cmgs,num_cmgs)), Iws*np.identity(num_cmgs)), axis = 1)
    # M = np.concatenate((row1, row2, row3), axis=0)
    
    terms = np.zeros((1,3))
    for i in range(num_cmgs):
        # next_term = ((Js*gamma_dot[i]*w_t[i] - (Jt-Jg)*w_t[i]*gamma_dot[i]) * Gs[:,i] + 
        #                                     (((Js*w_s[i] + Iws*OMEGA[i])*gamma_dot[i]) - (Jt+Jg)*w_s[i]*gamma_dot[i] + Iws*OMEGA[i]*w_g[i]) * Gt[:,i] -
        #                                     (Iws*OMEGA[i]*w_t[i]) * Gg[:,i])

        next_term = ((Iws*OMEGA[i]*w_g[i]) * Gt[:,i] - (Iws*OMEGA[i]*w_t[i]) * Gg[:,i])
        
        terms = terms + next_term
        
    # w function in S&J ex 4.15
    f_w = -tilde(w) @ I @ w + L - terms.T
   
    # OMEGA function in S&J ex 4.15
    f_OMEGA = u_s #np.array([u_s[i] for i in range(num_cmgs)]).reshape(num_cmgs,1)

    # Right hand side of expressions (S&J ex 4.15)
    RHS = np.concatenate((f_w, f_OMEGA), axis=0) 

    # Solve matrix equation 
    solution = np.linalg.inv(M) @ RHS
    
    #extract particular elements of the solution vector for debugging
    w_dot = solution[0:3]
    OMEGA_dot = solution[3:3+num_cmgs]

    # The below is used to force the CMGs into 'locked' configurations to represent CMG cases, RW cases, or Rigid body cases
    #OMEGA_dot = np.zeros((num_cmgs,1))

    state_dot = np.concatenate((s_dot, w_dot, OMEGA_dot, pos_dot, vel_dot), axis=0)
    return state_dot


def timestep():
    return 0.5


def calc_ang_mom_b(state_vector):
    #extract inertias for later calculations
    num_cmgs = 4
    I_list = inertia_properties()
    I_s = I_list[1]
    I_ws = I_list[3][0,0]
    I_wt = I_list[3][1,1]
   
    w_BN = state_vector[3:6]
    OMEGA = state_vector[6:6+num_cmgs]
 
    H_B_vec_b = I_s @ w_BN

    gimbal_frames = load_spacecraft_configuration()
    Gs, Gt, Gg = g_frames_2_g_mats(gimbal_frames)

    w_s = [Gs[:,j].T @ w_BN for j in range(num_cmgs)]
    w_t = [Gt[:,j].T @ w_BN for j in range(num_cmgs)]
    w_g = [Gg[:,j].T @ w_BN for j in range(num_cmgs)]

    # Loop to calculate gimbal angular momentum, wheel angular momentum, and system kinetic energy for each CMG
    H_W_vec_b = np.zeros((3,1))
    
    for cmg in range(num_cmgs):                                            
        H_W_vec_b = H_W_vec_b + (np.asarray(
                                                    (I_ws*(w_s[cmg] + OMEGA[cmg])*Gs[:,cmg]) 
                                                    + (I_wt*w_t[cmg]*Gt[:,cmg]) 
                                                    + (I_wt*(w_g[cmg])*Gg[:,cmg])).reshape(3,1)) 
    ang_mom_b = H_B_vec_b + H_W_vec_b
    ang_mom_b.reshape((3,1))
    return ang_mom_b


def momentum_dump(state_vector, num_cmgs):
    V = 0.000005   # momentum controller gain
    threshold = 0.0
    momentum_bias = np.array([[25],[25],[-25],[-25]])
    OMEGA = state_vector[6:6+num_cmgs]
    delta_OMEGA = OMEGA - momentum_bias

    gimbal_frames = load_spacecraft_configuration()
    Gs, Gt, Gg = g_frames_2_g_mats(gimbal_frames)


    if mag(delta_OMEGA) > threshold:
        L_h_dump = -V * Gs @ delta_OMEGA
    else:
        L_h_dump = np.array([[0], [0], [0]])

    return L_h_dump


def integrate(time, control_reference=None):
    # Setup initial values/lists to build on
    time_step = timestep() # integration time step in seconds
    int_time = np.linspace(0, time, (int(time/time_step))+1)

    # Load initial conditions and build initial lists to build on
    attitude, rates, OMEGA, gimbal_frames, num_cmgs, position, velocity = load_initial_conditions()
    state = [np.concatenate((attitude[0], rates[0], OMEGA[0], position[0], velocity[0]), axis=0)]
    att_err = [np.array([[0], [0], [0]])]
    rate_err = [np.array([[0], [0], [0]])]
    control = [np.zeros((2*num_cmgs, 1))]
    I_list = inertia_properties()
    Gs, Gt, Gg = g_frames_2_g_mats(gimbal_frames)
    G_list = [Gs, Gt, Gg]
    w_gframe = []
    ang_mom_b = [calc_ang_mom_b(state[0])]
    mission_mode = ['Initialization']
    
    # Perform Runge-Kutta 4th Order Integration for specified time
    for t in int_time:
        x = int(t/time_step)

        w_s = [Gs[:,i].T @ state[x][3:6] for i in range(num_cmgs)]
        w_t = [Gt[:,i].T @ state[x][3:6] for i in range(num_cmgs)]
        w_g = [Gg[:,i].T @ state[x][3:6] for i in range(num_cmgs)]
        w_gframe.append(np.concatenate((w_s, w_t, w_g), axis=1))   # 4x3 matrix where columns are all s components, t components, then g components

        r_vec_N_list = [item[-6:-3] for item in state[-3:]]
        v_vec_N_list = [item[-3:] for item in state[-3:]]
        
        L_h_dump = momentum_dump(state[x], num_cmgs)

        Lr, sigmaBR, wBR, b_w_RN, mode = evaluate_control_reference(t, state[x], w_gframe[x], G_list, num_cmgs, mission_mode[x], r_vec_N_list, v_vec_N_list, L_h_dump, control_reference) 
        OMEGA_dot_des = steering_law(t, state[x], Lr, I_list, w_gframe[x], G_list, b_w_RN, num_cmgs)
        
        # Test desired OMEGA dot and gamma dots
        #OMEGA_dot_des = -0.01*math.cos(0.05*t)*np.ones((num_cmgs,1))
        
        u = subservo(state[x], OMEGA_dot_des, w_gframe[x], I_list, num_cmgs)
        #u = 0.001*math.sin(0.05*t)*np.ones((2*num_cmgs,1))
        
        k1 = ODE(state[x], u, I_list, num_cmgs, gimbal_frames, L_h_dump,)
        k2 = ODE(state[x] + 0.5*time_step*k1, u, I_list, num_cmgs, gimbal_frames, L_h_dump,)
        k3 = ODE(state[x] + 0.5*time_step*k2, u, I_list, num_cmgs, gimbal_frames, L_h_dump,)
        k4 = ODE(state[x] + time_step*k3, u, I_list, num_cmgs, gimbal_frames, L_h_dump,)
        new_state = state[x] + (time_step*1/6)*(k1 + 2*k2 + 2*k3 + k4)
        new_state[0:3] = check_for_shadow_set(new_state[0:3])

        # Save results from current time step
        state.append(new_state)
        attitude.append(new_state[0:3])
        rates.append(new_state[3:6])
        OMEGA.append(new_state[6:6+num_cmgs])
        position.append(new_state[6+num_cmgs:6+num_cmgs+3])
        velocity.append(new_state[6+num_cmgs+3:6+num_cmgs+6])
        att_err.append(sigmaBR)
        rate_err.append(wBR)
        control.append(u)
        mission_mode.append(mode)
        ang_mom_b.append(calc_ang_mom_b(new_state))


    # Remove last data point from each list so that there are the same number of elements as time
    state.pop()
    attitude.pop()
    rates.pop()
    OMEGA.pop()
    position.pop()
    velocity.pop()
    att_err.pop()
    rate_err.pop()
    control.pop()
    mission_mode.pop()
    ang_mom_b.pop()
   
    return int_time, attitude, rates, OMEGA, w_gframe, control, att_err, rate_err, position, velocity, mission_mode, ang_mom_b


def plot_states(time, attitude, rates, OMEGA, position, mission_mode, title):
    t = time

    plt.figure()
    plt.plot(t, [item[0] for item in attitude])
    plt.plot(t, [item[1] for item in attitude])
    plt.plot(t, [item[2] for item in attitude])
    plt.title(title + ": Spacecraft Attitude vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Body Attitude [MRPs, Sigma BN]')
    plt.legend(['sigma1', 'sigma2', 'sigma3'])
    plt.grid(True)

    plt.figure()
    plt.plot(t, [item[0] for item in rates])
    plt.plot(t, [item[1] for item in rates])
    plt.plot(t, [item[2] for item in rates])
    plt.title(title + ': Angular Rates vs. Time')
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Body Rates [Rad/Sec]')
    plt.legend(['w1', 'w2', 'w3'])
    plt.grid(True)

    # plt.figure()
    # plt.plot(t, [item[0] for item in data[3]])
    # plt.plot(t, [item[1] for item in data[3]])
    # plt.plot(t, [item[2] for item in data[3]])
    # plt.plot(t, [item[3] for item in data[3]])
    # plt.title(title + ': Gamma vs. Time')
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('Gamma Angles [Rad]')
    # plt.legend(['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    # plt.grid(True)

    # plt.figure()
    # plt.plot(t, [item[0] for item in data[4]])
    # plt.plot(t, [item[1] for item in data[4]])
    # plt.plot(t, [item[2] for item in data[4]])
    # plt.plot(t, [item[3] for item in data[4]])
    # plt.title(title + ': Gamma_dot vs. Time')
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('Rates [Rad/Sec]')
    # plt.legend(['gamma_dot1', 'gamma_dot2', 'gamma_dot3', 'gamma_dot4'])
    # plt.grid(True)

    plt.figure()
    plt.plot(t, [item[0] for item in OMEGA])
    plt.plot(t, [item[1] for item in OMEGA])
    plt.plot(t, [item[2] for item in OMEGA])
    plt.plot(t, [item[3] for item in OMEGA])
    plt.title(title + ': OMEGA vs. Time')
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Wheel Rates [Rad/Sec]')
    plt.legend(['OMEGA1', 'OMEGA2', 'OMEGA3', 'OMEGA4'])
    plt.grid(True)

    # TODO: add plots for remaining parts of state vector
    plt.figure()
    ax = plt.axes(projection='3d')
    x = [float(item[0]) for item in position]
    y = [float(item[1]) for item in position]
    z = [float(item[2]) for item in position]
    ax.plot3D(x, y, z, 'blue')
    #plt.legend(['-0.1', '0', '0.1'])
    plt.xlabel('X position [km]')
    plt.ylabel('Y position [km]')
    #ax = plt.figure.gca(projection='3d')
    ax.set_zlabel('Z position [km]')
    # # plt.zlabel('Z position [km]')
    ax.scatter3D(7000, 7000, 7000, c='gray')
    ax.scatter3D(-7000, -7000, -7000, c='gray')

    plt.figure()
    plt.plot(t, mission_mode)
    plt.title(title + ': Mission Mode vs. Time')
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Mode')


def plot_errors(time, att_err, rate_err,title):
    t = time
    plt.figure()
    plt.plot(t, [item[0] for item in att_err])
    plt.plot(t, [item[1] for item in att_err])
    plt.plot(t, [item[2] for item in att_err])
    plt.title(title + ': Attitude Error (MRPs, Sigma BR) vs. Time')
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Attitude Error [Sigma BR]')
    plt.legend(['sigma1', 'sigma2', 'sigma3'])
    plt.grid(True)

    plt.figure()
    plt.plot(t, [item[0] for item in rate_err])
    plt.plot(t, [item[1] for item in rate_err])
    plt.plot(t, [item[2] for item in rate_err])
    plt.title(title + ': Rate Error vs. Time')
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Rate Error [Rad/Sec]')
    plt.legend(['w1', 'w2', 'w3'])
    plt.grid(True)


def plot_control(time, control, title):
    t = time
    plt.figure()
    plt.plot(t, [item[0] for item in control])
    plt.plot(t, [item[1] for item in control])
    plt.plot(t, [item[2] for item in control])
    plt.plot(t, [item[3] for item in control])
    plt.title(title + ": U_s Control Input vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Control Torque [N-m]')
    plt.legend(['u_s1', 'u_s2', 'u_s3', 'u_s4'])
    plt.grid(True)

    # plt.figure()
    # plt.plot(t, [item[4] for item in data[10]])
    # plt.plot(t, [item[5] for item in data[10]])
    # plt.plot(t, [item[6] for item in data[10]])
    # plt.plot(t, [item[7] for item in data[10]])
    # plt.title(title + ": U_g Control Input vs. Time")
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('Control Torque [N-m]')
    # plt.legend(['u_g1', 'u_g2', 'u_g3', 'u_g4'])
    # plt.grid(True)


def plot_conservation(time, attitude, rates, OMEGA, control, title):
    t = time
    num_cmgs = len(OMEGA[0])
    #extract inertias for later calculations

    I_list = inertia_properties()
    I_s = I_list[1]
    I_gs = I_list[2][0,0]
    I_ws = I_list[3][0,0]
    I_wt = I_list[3][1,1]
    Jt = I_list[4][1,1]
    Jg = I_list[4][2,2]

    L = 0.0* np.ones((3,1))

    Hmag = []
    H_B_vec_b = []
    H_G_vec_b = []
    H_W_vec_b = []
    H_tot_vec_N = []
    BN = []
    T = []
    T_dot = [np.zeros(1)]
    T_dot_inputs = [np.zeros(1)]
    T_dot_diff = []

    w_BN = rates
    u_vec = control

    # Perform angular momentum and Energy calculations for each time step
    for i in range(len(t)):
        # Calculate Angular Momentum
        BN.append(MRP2DCM(attitude[i]))
        H_B_vec_b.append(I_s @ w_BN[i])

        gimbal_frames = load_spacecraft_configuration()
        Gs, Gt, Gg = g_frames_2_g_mats(gimbal_frames)

        w_s = [Gs[:,j].T @ w_BN[i] for j in range(num_cmgs)]
        w_t = [Gt[:,j].T @ w_BN[i] for j in range(num_cmgs)]
        w_g = [Gg[:,j].T @ w_BN[i] for j in range(num_cmgs)]

        # Loop to calculate gimbal angular momentum, wheel angular momentum, and system kinetic energy for each CMG
        H_G_vec_terms = np.zeros((3,1))
        H_W_vec_terms = np.zeros((3,1))
        
        T_dot_terms = w_BN[i].T @ L
        T_terms = 0.5*w_BN[i].T @ I_s @ w_BN[i]
       # T_terms = np.zeros((3,1))
        for cmg in range(num_cmgs):
            #w_s = data[8][i][cmg][0]
            #w_t = data[8][i][cmg][1]
            #w_g = data[8][i][cmg][2]

            # H_G_vec_terms = H_G_vec_terms + ((np.asarray(
            #                                         (I_gs*w_s[cmg]*Gs[:,cmg]) 
            #                                         + (I_gt*w_t[cmg]*Gt[:,cmg]) 
            #                                         + I_gg*(w_g[cmg] + gamma_dot[i][cmg])*Gg[:,cmg])).reshape((3,1)))
                                                    

            H_W_vec_terms = H_W_vec_terms + (np.asarray(
                                                        (I_ws*(w_s[cmg] + OMEGA[i][cmg])*Gs[:,cmg]) 
                                                        + (I_wt*w_t[cmg]*Gt[:,cmg]) 
                                                        + (I_wt*(w_g[cmg])*Gg[:,cmg])).reshape(3,1)) 

              

            T_terms = T_terms + (0.5 * (I_ws*(OMEGA[i][cmg] + w_s[cmg])**2 
                                         + I_gs*(w_s[cmg]**2) 
                                         + Jt*(w_t[cmg]**2)
                                         + Jg*((w_g[cmg])**2)))
            
            T_dot_terms = T_dot_terms + (OMEGA[i][cmg]*u_vec[i][cmg])

        #print(mag(w_BN[i]))
        H_W_vec_b.append(H_W_vec_terms) 
        H_G_vec_b.append(H_G_vec_terms)
        T.append(T_terms[0])
        T_dot_inputs.append(float(T_dot_terms))

        if (T[i]-T[i-1])/(t[i]-t[i-1]) == 0:  
            T_dot.append((T_dot[-1]))
        else:
            T_dot.append((T[i]-T[i-1])/(t[i]-t[i-1]))
        #print((T[i]-T[i-1])/(t[i]-t[i-1]))

        T_dot_diff.append(T_dot[i]-T_dot_inputs[i])
        
        H_tot_vec_N.append(BN[i].T @ (H_B_vec_b[i] + H_W_vec_b[i]))
        Hmag.append(mag(H_tot_vec_N[i]))
    
    H_vec_x = [item[0] for item in H_tot_vec_N]
    H_vec_y = [item[1] for item in H_tot_vec_N]
    H_vec_z = [item[2] for item in H_tot_vec_N]
    #print(Hmag[-1])
    #T_dot = [(T[i]-T[i-1])/(t[i]-t[i-1]) for i in range(1,len(t))]
    #T_dot.append((T[-1]-T[-2])/(t[i]-t[i-1]))
    T_dot.pop()
    T_dot_inputs.pop()
    
    # Generate Plots
    plt.figure()
    plt.plot(t, Hmag)
    plt.title(title + ": Angular Momentum Magnitude vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Angular Momentum [kg-m^2/s]')
    plt.ticklabel_format(useOffset=False)
    plt.grid(True)

    plt.figure()
    plt.plot(t, H_vec_x)
    plt.plot(t, H_vec_y)
    plt.plot(t, H_vec_z)
    plt.title(title + ": Inertial Angular Momentum vector vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Angular Momentum [kg-m^2/s]')
    plt.legend(['H_x', 'H_y', 'H_z'])
    plt.ticklabel_format(useOffset=False)
    plt.grid(True)
    

    plt.figure()
    plt.plot(t, T)
    plt.title(title + ": Kinetic Energy vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Kinetic Energy [J]')
    plt.ticklabel_format(useOffset=False)
    plt.grid(True)

    plt.figure()
    plt.plot(t, T_dot)
    plt.plot(t,T_dot_inputs)
    plt.title(title + ": Power vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Power [J/s]')
    plt.legend(['T_dot_num', 'T_dot_analytic'])
    plt.ticklabel_format(useOffset=False)
    plt.grid(True)

    plt.figure()
    #plt.semilogy(t, T_dot_diff)
    plt.plot(t, T_dot_diff)
    plt.title(title + ": Difference in numerical vs predicted power")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Power [J/s]')
    #plt.yscale('log')
    plt.ticklabel_format(useOffset=False)
    plt.grid(True)

    # plt.figure()
    # plt.plot(t, w_gframe_0_s)
    # plt.plot(t, w_gframe_0_t)
    # plt.plot(t, w_gframe_0_g)
    # plt.title('w_gframe_0 vs. Time')
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('rad/s')
    # plt.ticklabel_format(useOffset=False)
    # plt.grid(True)

    # plt.figure()
    # plt.plot(t, w_gframe_1_s)
    # plt.plot(t, w_gframe_1_t)
    # plt.plot(t, w_gframe_1_g)
    # plt.title('w_gframe_1 vs. Time')
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('rad/s')
    # plt.ticklabel_format(useOffset=False)
    # plt.grid(True)

    # plt.figure()
    # plt.plot(t, w_gframe_2_s)
    # plt.plot(t, w_gframe_2_t)
    # plt.plot(t, w_gframe_2_g)
    # plt.title('w_gframe_2 vs. Time')
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('rad/s')
    # plt.ticklabel_format(useOffset=False)
    # plt.grid(True)

    # plt.figure()
    # plt.plot(t, w_gframe_3_s)
    # plt.plot(t, w_gframe_3_t)
    # plt.plot(t, w_gframe_3_g)
    # plt.title('w_gframe_3 vs. Time')
    # plt.xlabel('Time from Epoch [sec]')
    # plt.ylabel('rad/s')
    # plt.ticklabel_format(useOffset=False)
    # plt.grid(True)


def plot_checks(data, title):
    t = data[0]

    plt.figure()
    plt.plot(t, data[13])
    plt.title(title + ": Condition Number vs. Time")
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Condition number Kappa')
    plt.grid(True)

    plt.figure()
    plt.plot(t, data[14])
    plt.title(title + ': delta (singularity measure) vs. Time')
    plt.xlabel('Time from Epoch [sec]')
    plt.ylabel('Singularity Index')
    plt.grid(True)


def plot_reference_frames( frames, args={}, vectors = [], plots = [], planes = [] ):
	_args = {
		'figsize'       : ( 12, 12 ),
		'base_frame'    : True,
		'base_color'    : 'w',
		'base_label'    : 'Inertial',
		'frame_labels'  : [ '' ] * len( frames ),
		'frame_colors'  : [ 'm', 'c', 'b' ],
		'frame_zorders' : [ 10 ] * len( frames ),
		'vector_colors' : [ 'm', 'c', 'b' ],
		'vector_labels' : [ '' ] * len( vectors ),
		'vector_texts'  : True,
		'plots_labels'  : [ '' ] * len( plots ),
		'plots_colors'  : [ 'm' ],
		'plots_styles'  : [ '-' ] * len( plots ),
		'eq_plane'      : False,
		'eq_plane_color': 'c',
		'plane_labels'  : [ '' ] * len( planes ),
		'plane_colors'  : [ 'w' ],
		'plane_alphas'  : [ 0.3 ] * len( planes ),
		'no_axes'       : True,
		'axes_no_fill'  : False,
		'legend'        : True,
		'xlabel'        : 'X',
		'ylabel'        : 'Y',
		'zlabel'        : 'Z',
		'xlim'          : 1,
		'ylim'          : 1,
		'zlim'          : 1,
		'title'         : '',
		'azimuth'       : None,
		'elevation'     : None,
		'show'          : False,
		'filename'      : False,
		'dpi'           : 300,
		'frame_text_scale' : 1.1,
		'vector_text_scale': 1.3
	}
	for key in args.keys():
		_args[ key ] = args[ key ]
    
	fig      = plt.figure( figsize = _args[ 'figsize' ] )
	ax       = fig.add_subplot( 111, projection = '3d'  )
	zeros    = [ 0.0, 0.0, 0.0 ]
	n        = 0
	identity = [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ]
    
	for frame in frames:
		'''
		The frame is passed into the quiver method by rows, but they
		are being plotted by columns. So the 3 basis vectors of the frame
		are the columns of the 3x3 matrix
		'''

		ax.quiver( zeros, zeros, zeros,
			frame[ 0, : ], frame[ 1, : ], frame[ 2, : ],
			color  = _args[ 'frame_colors'  ][ n ],
			label  = _args[ 'frame_labels'  ][ n ],
			zorder = _args[ 'frame_zorders' ][ n ] )

		if _args[ 'vector_texts' ]:
			frame *= _args[ 'frame_text_scale' ]
			ax.text( frame[ 0, 0 ], frame[ 1, 0 ], frame[ 2, 0 ], 'X',
				color = _args[ 'frame_colors' ][ n ] )
			ax.text( frame[ 0, 1 ], frame[ 1, 1 ], frame[ 2, 1 ], 'Y',
				color = _args[ 'frame_colors' ][ n ] )
			ax.text( frame[ 0, 2 ], frame[ 1, 2 ], frame[ 2, 2 ], 'Z',
				color = _args[ 'frame_colors' ][ n ] )
		n += 1

	if _args[ 'base_frame' ]:
		ax.quiver( zeros, zeros, zeros,
			identity[ 0 ], identity[ 1 ], identity[ 2 ],
			color  = _args[ 'base_color' ],
			label  = _args[ 'base_label' ],
			zorder = 0 )

		if _args[ 'vector_texts' ]:
			ax.text( _args[ 'frame_text_scale' ], 0, 0, 'X',
				color = _args[ 'base_color' ] )
			ax.text( 0, _args[ 'frame_text_scale' ], 0, 'Y',
				color = _args[ 'base_color' ] )
			ax.text( 0, 0, _args[ 'frame_text_scale' ], 'Z',
				color = _args[ 'base_color' ] )
	n = 0
	for plot in plots:
		ax.plot( plot[ :, 0 ], plot[ :, 1 ], plot[ :, 2 ],
			_args[ 'plots_colors' ][ n ] + _args[ 'plots_styles' ][ n ],
			label = _args[ 'plots_labels' ][ n ] )
		n += 1

	n = 0
	for vector in vectors:
		ax.quiver( 0, 0, 0,
			vector[ 0 ], vector[ 1 ], vector[ 2 ],
			color = _args[ 'vector_colors' ][ n ],
			label = _args[ 'vector_labels' ][ n ] )

		if _args[ 'vector_texts' ]:
			vector *= _args[ 'vector_text_scale' ]
			ax.text( vector[ 0 ], vector[ 1 ], vector[ 2 ],
				_args[ 'vector_labels' ][ n ],
				color = _args[ 'vector_colors' ][ n ] )
		n += 1

	n = 0
	for plane in planes:
		ax.plot_surface( plane[ 0 ], plane[ 1 ], plane[ 2 ],
			color  = _args[ 'plane_colors' ][ n ],
			alpha  = _args[ 'plane_alphas' ][ n ],
			zorder = 0 )

	ax.set_xlabel( _args[ 'xlabel' ] )
	ax.set_ylabel( _args[ 'ylabel' ] )
	ax.set_zlabel( _args[ 'zlabel' ] )
	ax.set_xlim( [ -_args[ 'xlim' ], _args[ 'xlim' ] ] )
	ax.set_ylim( [ -_args[ 'ylim' ], _args[ 'ylim' ] ] )
	ax.set_zlim( [ -_args[ 'zlim' ], _args[ 'zlim' ] ] )
	ax.set_box_aspect( [ 1, 1, 1 ] )
	ax.set_title( _args[ 'title' ] )

	if _args[ 'legend' ]:
		ax.legend()

	if _args[ 'no_axes' ]:
		ax.set_axis_off()

	if _args[ 'axes_no_fill' ]:
		ax.w_xaxis.pane.fill = False
		ax.w_yaxis.pane.fill = False
		ax.w_zaxis.pane.fill = False

	if _args[ 'azimuth' ] is not None:
		ax.view_init( elev = _args[ 'elevation' ],
					  azim = _args[ 'azimuth'   ] )

	if _args[ 'show' ]:
		plt.show()
        

	if _args[ 'filename' ]:
		plt.savefig( _args[ 'filename' ], dpi = _args[ 'dpi' ] )
		print( 'Saved', _args[ 'filename' ] )

	plt.close()


def plot_orbits( rs, args, vectors = [] ):
	_args = {
		'figsize'      : ( 10, 8 ),
		'labels'       : [ '' ] * len( rs ),
		'colors'       : COLORS[ : ],
		'traj_lws'     : 3,
		'dist_unit'    : 'km',
		'groundtracks' : False,
		'cb_radius'    : 6378.0,
		'cb_SOI'       : None,
		'cb_SOI_color' : 'c',
		'cb_SOI_alpha' : 0.7,
		'cb_axes'      : True,
		'cb_axes_mag'  : 2,
		'cb_cmap'      : 'Blues',
		'cb_axes_color': 'w',
		'axes_mag'     : 0.8,
		'axes_custom'  : None,
		'title'        : 'Trajectories',
		'legend'       : True,
		'axes_no_fill' : True,
		'hide_axes'    : False,
		'azimuth'      : False,
		'elevation'    : False,
		'show'         : False,
		'filename'     : False,
		'dpi'          : 300,
		'vector_colors': [ '' ] * len( vectors ),
		'vector_labels': [ '' ] * len( vectors ),
		'vector_texts' : False
	}
	for key in args.keys():
		_args[ key ] = args[ key ]

	fig = plt.figure( figsize = _args[ 'figsize' ] )
	ax  = fig.add_subplot( 111, projection = '3d'  )

	max_val = 0
	n       = 0

	for r in rs:
		_r = r.copy() #* dist_handler[ _args[ 'dist_unit' ] ]

		ax.plot( _r[ :,0 ], _r[ :,1 ], _r[ :,2 ],
        #ax.plot( _r[ 0 ], _r[ 1 ], _r[ 2 ],
			color = _args[ 'colors' ][ n ], label = _args[ 'labels' ][ n ],
			zorder = 10, linewidth = _args[ 'traj_lws' ] )
		ax.plot( [ _r[ :,0 ] ], [ _r[ :,1 ] ], [ _r[ :,2 ] ], 'o',
			color = _args[ 'colors' ][ n ] )

		# if _args[ 'groundtracks' ]:
		# 	rg  = _r / np.linalg.norm( r, axis = 1 ).reshape( ( r.shape[ 0 ], 1 ) )
		# 	rg *= _args[ 'cb_radius' ]

		# 	ax.plot( rg[ :, 0 ], rg[ :, 1 ], rg[ :, 2 ], cs[ n ], zorder = 10 )
		# 	ax.plot( [ rg[ 0, 0 ] ], [ rg[ 0, 1 ] ], [ rg[ 0, 2 ] ], cs[ n ] + 'o', zorder = 10 )			

		max_val = max( [ _r.max(), max_val ] )
		n += 1

	for vector in vectors:
		ax.quiver( 0, 0, 0,
			vector[ 'r' ][ 0 ], vector[ 'r' ][ 1 ], vector[ 'r' ][ 2 ],
			color = vector[ 'color' ], label = vector[ 'label' ] )

		if _args[ 'vector_texts' ]:
			vector[ 'r' ] *= _args[ 'vector_text_scale' ]
			ax.text( vector[ 'r' ][ 0 ], vector[ 'r' ][ 1 ], vector[ 'r' ][ 2 ],
				vector[ 'label' ],
				color = vector[ 'color' ] )

	_args[ 'cb_radius' ] *= dist_handler[ _args[ 'dist_unit' ] ]
	_u, _v = np.mgrid[ 0:2*np.pi:20j, 0:np.pi:20j ]
	_x     = _args[ 'cb_radius' ] * np.cos( _u ) * np.sin( _v )
	_y     = _args[ 'cb_radius' ] * np.sin( _u ) * np.sin( _v )
	_z     = _args[ 'cb_radius' ] * np.cos( _v )
	ax.plot_surface( _x, _y, _z, cmap = _args[ 'cb_cmap' ], zorder = 1 )

	if _args[ 'cb_SOI' ] is not None:
		_args[ 'cb_SOI' ] *= dist_handler[ _args[ 'dist_unit' ] ]
		_x *= _args[ 'cb_SOI' ] / _args[ 'cb_radius' ]
		_y *= _args[ 'cb_SOI' ] / _args[ 'cb_radius' ]
		_z *= _args[ 'cb_SOI' ] / _args[ 'cb_radius' ]
		ax.plot_wireframe( _x, _y, _z,
			color = _args[ 'cb_SOI_color' ],
			alpha = _args[ 'cb_SOI_alpha' ] )

	if _args[ 'cb_axes' ]:
		l       = _args[ 'cb_radius' ] * _args[ 'cb_axes_mag' ]
		x, y, z = [ [ 0, 0, 0 ], [ 0, 0, 0  ], [ 0, 0, 0 ] ]
		u, v, w = [ [ l, 0, 0 ], [ 0, l, 0 ], [ 0, 0, l ] ]
		ax.quiver( x, y, z, u, v, w, color = _args[ 'cb_axes_color' ] )

	xlabel = 'X (%s)' % _args[ 'dist_unit' ]
	ylabel = 'Y (%s)' % _args[ 'dist_unit' ]
	zlabel = 'Z (%s)' % _args[ 'dist_unit' ]

	if _args[ 'axes_custom' ] is not None:
		max_val = _args[ 'axes_custom' ]
	else:
		max_val *= _args[ 'axes_mag' ]

	ax.set_xlim( [ -max_val, max_val ] )
	ax.set_ylim( [ -max_val, max_val ] )
	ax.set_zlim( [ -max_val, max_val ] )
	ax.set_xlabel( xlabel )
	ax.set_ylabel( ylabel )
	ax.set_zlabel( zlabel )
	ax.set_box_aspect( [ 1, 1, 1 ] )
	ax.set_aspect( 'auto' )

	if _args[ 'azimuth' ] is not False:
		ax.view_init( elev = _args[ 'elevation' ],
					  azim = _args[ 'azimuth'   ] )
	
	if _args[ 'axes_no_fill' ]:
		ax.w_xaxis.pane.fill = False
		ax.w_yaxis.pane.fill = False
		ax.w_zaxis.pane.fill = False		

	if _args[ 'hide_axes' ]:
		ax.set_axis_off()

	if _args[ 'legend' ]:
		plt.legend()

	if _args[ 'filename' ]:
		plt.savefig( _args[ 'filename' ], dpi = _args[ 'dpi' ] )
		print( 'Saved', _args[ 'filename' ] )

	if _args[ 'show' ]:
		plt.show()

	plt.close()


def make_animation(attitudes, errors):
    i = 0
    file_list=[]
    file_list_raw=[]
    for att_frame, err_frame in zip(attitudes, errors):
        next_filename = "C:/Users/colli/Documents/Git/ASEN6010/Images/frame_" + str(i) + ".png"
        file_list_raw.append(next_filename)
        config = {
        'frame_labels': ['S/C Attitude', 'Reference Frame'],
        'show': False,
        'filename': next_filename
                }
        plot_reference_frames([att_frame, err_frame], config)
        file_list.append(iio.imread(next_filename))
        i += 1
    
    images = []
   # for file in Path("C:/Users/colli/Documents/Git/ASEN6010/Images").iterdir():
       # im = iio.imread(file)
       # images.append(im)

    #iio.mimsave('C:/Users/colli/Documents/Git/ASEN6010/animation.gif', images, duration=0.2)
    #iio.mimsave('C:/Users/colli/Documents/Git/ASEN6010/animation.gif', file_list)
    print("Saving GIF")
    with iio.get_writer('C:/Users/colli/Documents/Git/ASEN6010/animation.gif', mode='I') as writer:
        for filename in file_list_raw:
            image = iio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    # Run simulation
    t = 1000 # Simulation length in seconds
    int_time, attitude, rates, OMEGA, w_gframe, control, att_err, rate_err, position, velocity, mission_mode, ang_mom_b  = integrate(t, control_reference='Full-Mission')
    #I_list = inertia_properties()
    #gimbal_frames = load_spacecraft_configuration()
    #Gs, Gt, Gg = g_frames_2_g_mats(gimbal_frames)
    #G_list = [Gs, Gt, Gg]

    plot_states(int_time, attitude, rates, OMEGA, position, mission_mode, 'Project')
    plot_control(int_time, control, 'Project')
    plot_conservation(int_time, attitude, rates, OMEGA, control, 'Project')
    plot_errors(int_time, att_err, rate_err, 'Project')
    #plot_checks([int_time, attitude, rates, OMEGA, G_list, w_gframe, control, att_err, rate_err], 'Project')
    # args = {'show'         : True}
    # print(position[0].reshape(1,3))
    # pos_row  = np.vstack((position[0].reshape(1,3), position[1].reshape(1,3)))
    # print(pos_row[:,1])
    #plot_orbits([position[0].reshape(1,3),position[1].reshape(1,3)], args)
    
    # Post process data for animations
    attitudeDCMs = [MRP2DCM(MRP) for MRP in attitude]
    referenceDCMs = [MRP2DCM(MRP_err).T @ MRP2DCM(MRP) for MRP, MRP_err in zip(attitude, att_err)]
    
    
    config = {
        'frame_labels': ['S/C Attitude', 'Reference Frame'],
        'show': True
                }

    #make_animation(attitudeDCMs, referenceDCMs)
    plot_reference_frames([attitudeDCMs[-1], referenceDCMs[-1]], config)
    
    plt.show()