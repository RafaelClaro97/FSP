import numpy as np
from numpy import genfromtxt
from numpy.lib.function_base import average
from math import tan, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

dt = 0.25

def move(x, dt, u):
    return x + u*dt


def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def residual_h(a, b):
    y = a - b
    # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
    for i in range(0, len(y), 2):
        y[i + 1] = normalize_angle(y[i + 1])
    return y

def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def Hxy(x, landmarks):
    """ takes a state variable and returns the measurement
    that would correspond to that state. """
    hx = []
    for lmark in landmarks:
        px, py = lmark
        dist = sqrt((px - x[0])**2 + (py - x[1])**2)
        angle = atan2(py - x[1], px - x[0])
        hx.extend([dist, normalize_angle(angle - x[2])])
    return np.array(hx)

def Hyaw(x, landmarks):
    """ takes a state variable and returns the measurement
    that would correspond to that state. """
    hx = []
    for lmark in landmarks:
        px, py = lmark
        dist = sqrt((px - x[0])**2 + (py - x[1])**2)
        angle = atan2(py - x[1], px - x[0])
        hx.extend([dist, normalize_angle(angle)])
    return np.array(hx)

def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = atan2(sum_sin, sum_cos)
    return x

def run_localization(
    H, cmds, landmarks, init, obs_polar, sigma_vel, sigma_steer, sigma_range, 
    sigma_bearing, ellipse_step=1, step=1):

    points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0, 
                                    subtract=residual_x)
    ukf = UKF(dim_x=3, dim_z=2*len(landmarks), fx=move, hx=H,
              dt=dt, points=points, x_mean_fn=state_mean, 
              z_mean_fn=z_mean, residual_x=residual_x, 
              residual_z=residual_h)

    ukf.x = init
    ukf.P = np.diag([.1, .1, .05])
    ukf.R = np.diag([sigma_range**2, sigma_bearing**2]*len(landmarks))
    ukf.Q = np.eye(3)*0.0001
    
    track = []

    for i, u in enumerate(cmds):

        ukf.predict(u=u)
        ukf.update(obs_polar[i,:2], landmarks=landmarks)

        track.append(ukf.x)


    track = np.array(track)

    return ukf, track


########## MAIN #########

gt = np.genfromtxt('Data/gt.csv', delimiter='\t')[1:, 1:4]
obs_polar = np.genfromtxt('Data/obs_polar.csv', delimiter='\t')[1:, 1:4]
cmds = np.genfromtxt('Data/cmd.csv', delimiter='\t')[1:,1:4]
landmarks = np.array([[0, 0]])

ukf_xy,track_xy = run_localization(Hxy, cmds, landmarks, [0,0,0], obs_polar, sigma_vel=0.1, sigma_steer=np.radians(1), sigma_range=0.3, sigma_bearing=0.1)
ukf_yaw,track_yaw = run_localization(Hyaw, cmds, landmarks, [0,0,0], obs_polar, sigma_vel=0.1, sigma_steer=np.radians(1), sigma_range=0.3, sigma_bearing=0.1)

# Error Calculation
euc_error = np.sqrt((track_xy[:, 0] - gt[:, 0])**2 + (track_xy[:, 1] - gt[:, 1])**2)
ori_error = abs(np.arctan2(np.sin(track_yaw[:, 2] - gt[:, 2]), np.cos(track_yaw[:, 2] - gt[:, 2]))*180.0/np.pi)

# Average Error
print(average(euc_error), average(ori_error))

# Plot
plt.subplot(2, 2, 1)
plt.plot(gt[:, 0], gt[:,1], color='g', lw=2)
plt.plot(track_xy[:, 0], track_xy[:,1], color='r', lw=2)
plt.plot(gt[0, 0], gt[0, 1], 'bX', markersize=8)
plt.title("UKF Robot Position")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend(["Ground Truth", "Estimation", "Starting Position"])

plt.subplot(2, 2, 2)
plt.plot(gt[:, 2]*180.0/np.pi, color='g', lw=2)
plt.plot(track_yaw[:, 2]*180.0/np.pi, color='r', lw=2)
plt.title("UKF Robot Orientation")
plt.xlabel("N samples")
plt.ylabel("Yaw (deg)")
plt.legend(["Ground Truth", "Estimation"])

plt.subplot(2, 2, 3)
plt.plot(euc_error, color='b', lw=2)
plt.title("UKF Position Euclidean Error")
plt.xlabel("N samples")
plt.ylabel("Error (m)")
plt.legend(["Position Euclidean Error"])

plt.subplot(2, 2, 4)
plt.plot(ori_error, color='b', lw=2)
plt.title("UKF Orientation Error")
plt.xlabel("N samples")
plt.ylabel("Error (deg)")
plt.legend(["Orientation Error"])

plt.show()