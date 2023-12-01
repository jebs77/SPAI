import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cvxpy as cp
import sys

def cp_transport_plan(P, Q):
    # Extract positions and pressures
    P_positions = calculate_cloud_position(P)
    Q_positions = calculate_cloud_position(Q)
    P_pressures = calculate_cloud_pressure(P)
    Q_pressures = calculate_cloud_pressure(Q)

    # Define your data
    N_size = len(P)
    M_size = len(Q)
    m_ones = np.ones((M_size,1))
    n_ones = np.ones((N_size,1))
    m1_ones = np.ones((M_size+1,1))
    n1_ones = np.ones((N_size+1,1))
    max_num = np.ones((1,1)) * sys.maxsize
    zero = np.zeros((1,1))

    # Define the dimensions
    xi = 1.05 * np.linalg.norm(P_positions-Q_positions)**2
    C = np.zeros((N_size, M_size))
    for i, x in enumerate(P_positions):
        for j, y in enumerate(Q_positions):
            C[i, j] = np.sum((x - y)**2)

    print("Processing...")
    print("Dimensions of m_ones:", m_ones.shape)
    print("Dimensions of C:", C.shape)
    C_bar = cp.vstack([cp.hstack([C, xi * m_ones]) , cp.hstack([xi * n_ones.T, max_num])])    

    u = cp.Variable((N_size,1))
    v = cp.Variable((M_size,1))

    # Variables
    T = cp.Variable((N_size, M_size), nonneg=True)
    T_bar = cp.vstack([cp.hstack([T, u]),cp.hstack([v.T, zero])])

    # Objective function
    objective = cp.Minimize(cp.sum(cp.multiply(C_bar, T_bar)))
    constrain1 = cp.hstack([T, u])
    constrain2 = cp.hstack([T.T, v])
    # Constraints
    constraints = [
        constrain1 @ m1_ones == P_pressures,
        constrain2 @ n1_ones == Q_pressures
    ]

    # Formulate and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return T.value, u.value, v.value

# Function to construct the interpolated point cloud
def construct_interpolated_point_cloud(P, Q, T, u, v, kappa):
    # Calculate P_position and Q_position
    P_positions = calculate_cloud_position(P)
    Q_positions = calculate_cloud_position(Q)

    # Initialize the variables
    K = len(P_positions) + len(Q_positions)  # total number of points after interpolation
    interpolated_points = []
    
    # Copy the transport plan
    T_prime = np.copy(T)
    
    # Start with the algorithm
    k = 1
    for i in range(len(u)):
        if u[i] > 0:
            if np.sum(T[:, i]) == 0:  # Check for vanishing point
                r_k = (1 - kappa) * u[i]
                z_k = P_positions[i]
                interpolated_points.append((r_k, z_k))
                k += 1
            else:
                T_prime[:, i] = T[:, i] + (1 - kappa) * u[i] * T[:, i] / np.linalg.norm(T[:, i], 1)

    for j in range(len(v)):
        if v[j] > 0:
            if np.sum(T[j, :]) == 0:  # Check for appearing point
                r_k = kappa * v[j]
                z_k = Q_positions[j]
                interpolated_points.append((r_k, z_k))
                k += 1
            else:
                T_prime[j, :] = T_prime[j, :] + kappa * v[j] * T_prime[j, :] / np.linalg.norm(T_prime[j, :], 1)

    for i in range(len(P_positions)):
        for j in range(len(Q_positions)):
            if T_prime[i, j] > 0:  # Check for moving point
                r_k = T_prime[i, j]
                z_k = (1 - kappa) * P_positions[i] + kappa * Q_positions[j]
                interpolated_points.append((r_k, z_k))
                k += 1

    # After calculating interpolated points and pressures, construct the final cloud
    interpolated_cloud = [{'position': tuple(z), 'pressure': r} for r, z in interpolated_points]

    # Return the interpolated cloud
    return interpolated_cloud

# Function to generate random points with pressure
def generate_random_sources(num_sources):
    positions = np.random.rand(num_sources, 3) * 5  # random positions within a 5x5x5 cube
    pressures = np.random.rand(num_sources)  # random pressures between 0 and 1
    return [{'position': tuple(pos), 'pressure': pressure} for pos, pressure in zip(positions, pressures)]

def calculate_cloud_position(P):
    P_positions = np.array([list(d['position']) for d in P])
    return P_positions

def calculate_cloud_pressure(P):
    P_pressures = np.array([d['pressure'] for d in P])
    return P_pressures

def visualize_interpolation(P, Q, T, u, v, kappa):
    # Define the virtual sources for P and Q as numpy arrays
    P_positions = calculate_cloud_position(P)
    Q_positions = calculate_cloud_position(Q)

    # Define the pressures as arrays
    P_pressures = calculate_cloud_pressure(P)
    Q_pressures = calculate_cloud_pressure(Q)

    # Construct the interpolated point cloud
    interpolated_point_cloud = construct_interpolated_point_cloud(P, Q, T, u, v, kappa)
    interpolated_point_cloud_positions = calculate_cloud_position(interpolated_point_cloud)
    interpolated_point_cloud_pressures = calculate_cloud_pressure(interpolated_point_cloud)

    # Plot the original and interpolated point clouds
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original P point cloud with label for legend
    ax.scatter(P_positions[:, 0], P_positions[:, 1], P_positions[:, 2], s=1+(P_pressures), c='r', marker='o', label='Original P')

    # Plot the original Q point cloud with label for legend
    ax.scatter(Q_positions[:, 0], Q_positions[:, 1], Q_positions[:, 2], s=1+(Q_pressures), c='b', marker='^', label='Original Q')

    # Plot the interpolated point cloud with label for legend
    ax.scatter(interpolated_point_cloud_positions[:, 0], interpolated_point_cloud_positions[:, 1], interpolated_point_cloud_positions[:, 2], s=1+(interpolated_point_cloud_pressures), c='g', marker='x', label='Interpolated')

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
        x_i = source['position']
        p_i = source['pressure']

        # Calculate the time delay for the source
        tau_i = np.linalg.norm(x_i) / c
        
        # Add the impulse to the RIR
        h_t += p_i * delta_impulse(t, tau_i, fs)
    
    return t, h_t

def getFromFile(position, filename):
    data = sp.io.loadmat(filename)
    RIRs = data['RIRs']
    index = -1
    print(len(RIRs['x_position'][0]))
    for i in range(len(RIRs['x_position'][0])):
        if RIRs['x_position'][0][i] == position:
            index = i
            break
    
    if (index == -1):
        raise ValueError("Position not found in RIRs['x_position'][0]")
    
    DOA = RIRs['DOA'][0][index]
    P = RIRs['P'][0][index]

    # Adjusted for 2400 data points
    num_points = min(len(DOA), len(P),2400)

    data_list = [{'position': (DOA[i][0], DOA[i][1], DOA[i][2]), 'pressure': abs(P[i])} for i in range(num_points)]
    
    return data_list
