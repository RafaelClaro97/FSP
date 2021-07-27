import numpy as np
from numpy.lib.function_base import average
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import matplotlib.pyplot as plt
 
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
 
#def mouseCallback(event, x, y, flags, null):
def particle_filter(rho, theta, yaw, cmd_x, cmd_y, cmd_yaw):
    global est_list
    global particles
    global weights
    global landmarks
    
    #noise=sensorSigma * np.random.randn(1,2) + sensorMu
    
    heading = np.arctan2(cmd_y, cmd_x)
    distance = np.sqrt(cmd_x**2 + cmd_y**2)

    std=np.array([0.1, 0.1, 0.05])
    u=np.array([heading, distance, cmd_yaw])
    predict(particles, u, std, dt=0.25)
    
    zs = [rho, theta, yaw]
    update(particles, weights, z=zs, R=2, landmarks=landmarks)

    if neff(weights) < N/2:
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)

    mean, var = estimate(particles, weights)

    if mean[2] > np.pi:
        mean[2] -= (np.pi*2)
    elif mean[2] < -np.pi:
        mean[2] += np.pi*2
    mean[2] *= 180/np.pi

    est_list.append(mean)
    
 
#sensorMu=0
#sensorSigma=3
 
sensor_std_err=5
 
 
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(-np.pi, np.pi, size=N)
    return particles
 
def predict(particles, u, std, dt=1.):
    N = len(particles)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist
    particles[:, 2] += (u[2] * dt) + (np.random.randn(N) * std[2])
   
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
       
        distance=np.power((particles[:,0] - landmark[0])**2 + (particles[:,1] - landmark[1])**2, 0.5)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])
 
 
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)
    
def neff(weights):
    return 1. / np.sum(np.square(weights))
 
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
 
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j<N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
    
def estimate(particles, weights):
    pos = particles[:, 0:3]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var
 
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)
 
    
WIDTH=12
HEIGHT=12
x_range=np.array([-WIDTH,WIDTH])
y_range=np.array([-HEIGHT,HEIGHT])
 
#Number of partciles
N=1000
 
landmarks=np.array([[0,0]])
NL = len(landmarks)
particles=create_uniform_particles(x_range, y_range, N)
weights = np.array([1.0]*N)

est_list = []

gt = np.genfromtxt('Data/gt.csv', delimiter='\t')[1:,1:4]
obs_polar = np.genfromtxt('Data/obs_polar.csv', delimiter='\t')[1:,1:4]
cmd = np.genfromtxt('Data/cmd.csv', delimiter='\t')[1:,1:4]

for i in range(len(obs_polar)):
    particle_filter(obs_polar[i, 0], obs_polar[i, 1], obs_polar[i, 2], cmd[i, 0], cmd[i, 1], cmd[i, 2])

est_list = np.array(est_list)

# Error Calculation
euc_error = np.sqrt((est_list[:, 0] - gt[:, 0])**2 + (est_list[:, 1] - gt[:, 1])**2)
ori_error = abs(np.arctan2(np.sin(est_list[:, 2]*np.pi/180.0 - gt[:, 2]), np.cos(est_list[:, 2]*np.pi/180.0 - gt[:, 2]))*180.0/np.pi)

# Average Error
print(average(euc_error), average(ori_error))

# Plot
plt.subplot(2, 2, 1)
plt.plot(gt[:, 0], gt[:, 1], color='g', lw=2)
plt.plot(est_list[:, 0], est_list[:, 1], color='r', lw=2)
plt.plot(gt[0, 0], gt[0, 1], 'bX', markersize=8)
plt.axis('equal')
plt.title("Particle Filter Robot Position")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend(["Ground Truth", "Estimation", "Starting Position"])

plt.subplot(2, 2, 2)
plt.plot(gt[:, 2]*180.0/np.pi, color='g', lw=2)
plt.plot(est_list[:, 2], color='r', lw=2)
plt.axis('equal')
plt.title("Particle Filter Robot Orientation")
plt.xlabel("N samples")
plt.ylabel("Yaw (deg)")
plt.legend(["Ground Truth", "Estimation"])

plt.subplot(2, 2, 3)
plt.plot(euc_error, color='b', lw=2)
plt.title("PF Position Euclidean Error")
plt.xlabel("N samples")
plt.ylabel("Error (m)")
plt.legend(["Position Euclidean Error"])

plt.subplot(2, 2, 4)
plt.plot(ori_error, color='b', lw=2)
plt.title("PF Orientation Error")
plt.xlabel("N samples")
plt.ylabel("Error (deg)")
plt.legend(["Orientation Error"])

plt.show()