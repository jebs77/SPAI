import numpy as np
import matplotlib.pyplot as plt
import ot
import scipy as sp

# Function of the transport plan
def transport_plan(P, Q):
    # Convert point lists into numpy arrays
    P_array = np.array(P)
    Q_array = np.array(Q)
    # Calculate the cost matrix, the squared euclidean distance between points
    cost_matrix = ot.dist(P_array, Q_array, metric='sqeuclidean')
    
    # Compute the amount of mass to transport from P to Q
    a, b = np.ones((len(P),)) / len(P), np.ones((len(Q),)) / len(Q)
    
    # Compute the optimal transport matrix using the EMD solver
    # T = ot.emd(a, b, cost_matrix)
    
    # Compute the optimal transport matrix using the Sinkhorn solver
    T = ot.bregman.sinkhorn(a, b, cost_matrix, reg=1e-3, numItermax=100000)

    return T

# Function to construct the interpolated point cloud
def construct_interpolated_point_cloud(P, Q, T, kappa):
    # Assuming u and v are given or can be calculated from P and Q
    u = np.sum(T, axis=1)
    v = np.sum(T, axis=0)
    
    # Initialize the variables
    K = len(P) + len(Q)  # total number of points after interpolation
    interpolated_points = []
    
    # Copy the transport plan
    T_prime = np.copy(T)
    
    # Start with the algorithm
    k = 1
    for i, ui in enumerate(u):
        if ui > 0:
            if np.sum(T[:, i]) == 0:  # Check for vanishing point
                r_k = (1 - kappa) * ui
                z_k = P[i]
                interpolated_points.append((r_k, z_k))
                k += 1
            else:
                T_prime[:, i] = T[:, i] + (1 - kappa) * ui * T[:, i] / np.linalg.norm(T[:, i], 1)
    
    for j, vj in enumerate(v):
        if vj > 0:
            if np.sum(T[j, :]) == 0:  # Check for appearing point
                r_k = kappa * vj
                z_k = Q[j]
                interpolated_points.append((r_k, z_k))
                k += 1
            else:
                T_prime[j, :] = T_prime[j, :] + kappa * vj * T_prime[j, :] / np.linalg.norm(T_prime[j, :], 1)
    
    for i in range(len(P)):
        for j in range(len(Q)):
            if T_prime[i, j] > 0:  # Check for moving point
                r_k = T_prime[i, j]
                z_k = (1 - kappa) * P[i] + kappa * Q[j]
                interpolated_points.append((r_k, z_k))
                k += 1
    
    # Return the interpolated point cloud
    return np.array([z for r, z in interpolated_points])
    # You should already have your P, Q, and T defined here, as well as the value for kappa.
    # interpolated_point_cloud = construct_interpolated_point_cloud(P, Q, T, kappa)

# Function to generate random points with pressure
def generate_random_sources(num_sources):
    positions = np.random.rand(num_sources, 3) * 5  # random positions within a 5x5x5 cube
    pressures = np.random.rand(num_sources)  # random pressures between 0 and 1
    return [{'position': tuple(pos), 'pressure': pressure} for pos, pressure in zip(positions, pressures)]

def calculate_cloud_position(P, Q):
    P_positions = np.array([list(d['position']) for d in P])
    Q_positions = np.array([list(d['position']) for d in Q])
    return P_positions, Q_positions

def calculate_cloud_pressure(P, Q):
    P_pressures = np.array([d['pressure'] for d in P])
    Q_pressures = np.array([d['pressure'] for d in Q])
    return P_pressures, Q_pressures

def visualize_interpolation(P, Q, T, kappa):
    # Define the virtual sources for P and Q as numpy arrays
    P_positions, Q_positions = calculate_cloud_position(P, Q)

    # Define the pressures as arrays
    P_pressures, Q_pressures = calculate_cloud_pressure(P, Q)

    # Construct the interpolated point cloud
    interpolated_point_cloud = construct_interpolated_point_cloud(P_positions, Q_positions, T, kappa)

    # Plot the original and interpolated point clouds
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original P point cloud with label for legend
    ax.scatter(P_positions[:, 0], P_positions[:, 1], P_positions[:, 2], s=1+(P_pressures), c='r', marker='o', label='Original P')

    # Plot the original Q point cloud with label for legend
    ax.scatter(Q_positions[:, 0], Q_positions[:, 1], Q_positions[:, 2], s=1+(Q_pressures), c='b', marker='^', label='Original Q')

    # Plot the interpolated point cloud with label for legend
    ax.scatter(interpolated_point_cloud[:, 0], interpolated_point_cloud[:, 1], interpolated_point_cloud[:, 2], s=50, c='g', marker='x', label='Interpolated')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Original and Interpolated Point Clouds')

    # Add legend to the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    plt.show()

def delta_impulse(t, tau, fs):
    # Create a delta impulse function for discrete time
    # t is the time array, tau is the time at which the impulse occurs, fs is the sampling rate
    return np.where(np.isclose(t, tau, atol=1/fs), 1.0, 0.0)

def calculate_omnidirectional_RIR(P, c, fs, max_time):
    # fs is the sampling rate
    # max_time is the maximum time for the RIR in seconds

    # Define the time vector with a time step of 1/fs
    t = np.arange(0, max_time, 1/fs)
    
    # Initialize the RIR to zero
    h_t = np.zeros_like(t)

    # Calculate RIR as superposition of scaled delta functions
    for source in P:
        p_i = source[0]
        x_i = source[1:]
        
        # Calculate the time delay for the source
        tau_i = np.linalg.norm(x_i) / c
        
        # Add the impulse to the RIR
        h_t += p_i * delta_impulse(t, tau_i, fs)
    
    return t, h_t

def getFromFile(index, filename):
    data = sp.io.loadmat(filename)
    RIRs = data['RIRs']
    DOA = RIRs['DOA'][0][index]
    P = RIRs['P'][0][index]

    # Adjusted for 2400 data points
    num_points = min(len(DOA), len(P), 600)

    data_list = [{'position': (DOA[i][0], DOA[i][1], DOA[i][2]), 'pressure': abs(P[i])} for i in range(num_points)]
    
    return data_list

# positions = np.random.rand(num_sources, 3) * 5  # random positions within a 5x5x5 cube
# pressures = np.random.rand(num_sources)  # random pressures between 0 and 1
# return [{'position': tuple(pos), 'pressure': pressure} for pos, pressure in zip(positions, pressures)]