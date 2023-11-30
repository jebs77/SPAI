import SPAIdermenLib as SPAI
import matplotlib.pyplot as plt


P = SPAI.getFromFile('000', 'room_impulse_responses.mat')
Q = SPAI.getFromFile('020', 'room_impulse_responses.mat')

R = SPAI.getFromFile('010', 'room_impulse_responses.mat')
L = SPAI.getFromFile('010', 'room_impulse_responses.mat')
print(P)
# P= SPAI.generate_random_sources(10)
# Q= SPAI.generate_random_sources(10)
kappa = 0.5

P_positions= SPAI.calculate_cloud_position(P)
Q_positions= SPAI.calculate_cloud_position(Q)
R_positions= SPAI.calculate_cloud_position(R)
L_positions= SPAI.calculate_cloud_position(L)

T1 = SPAI.transport_plan(P_positions, Q_positions)

# T2 = SPAI.transport_plan(R_positions, L_positions)

SPAI.visualize_interpolation(P, Q, T1, kappa)

# SPAI.visualize_interpolation(R, L, T2, kappa)

# Example usage:
# Set the speed of sound to 343 m/s (speed of sound in air at 20 degrees Celsius)
# and a sampling rate of 44100 Hz (standard for audio)
speed_of_sound = 343
sampling_rate = 48000
max_simulation_time = 0.1  # in seconds

X_1 = SPAI.construct_interpolated_point_cloud(P, Q, T1, kappa)

# t1, h_t1 = SPAI.calculate_omnidirectional_RIR(X_1, speed_of_sound, sampling_rate, max_simulation_time)
# t2, h_t2 = SPAI.calculate_omnidirectional_RIR(X_2, speed_of_sound, sampling_rate, max_simulation_time)

t1, h_t1 = SPAI.calculate_omnidirectional_RIR(X_1, speed_of_sound, sampling_rate, max_simulation_time)
t2, h_t2 = SPAI.calculate_omnidirectional_RIR(R, speed_of_sound, sampling_rate, max_simulation_time)

# Plot the RIR
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t1, h_t1)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Simulated Omnidirectional RIR')

plt.subplot(2, 1, 2)
plt.plot(t2, h_t2)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Simulated Omnidirectional RIR')
plt.show()