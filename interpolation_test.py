import SPAIdermenLib as SPAI
import matplotlib.pyplot as plt


P = SPAI.getFromFile('000', 'room_impulse_responses.mat')
Q = SPAI.getFromFile('020', 'room_impulse_responses.mat')
print(P)
# P= SPAI.generate_random_sources(10)
# Q= SPAI.generate_random_sources(10)
kappa = 0.5

P_positions, Q_positions = SPAI.calculate_cloud_position(P, Q)

T = SPAI.transport_plan(P_positions, Q_positions)

SPAI.visualize_interpolation(P, Q, T, kappa)

# Example usage:
# Set the speed of sound to 343 m/s (speed of sound in air at 20 degrees Celsius)
# and a sampling rate of 44100 Hz (standard for audio)
speed_of_sound = 343
sampling_rate = 48000
max_simulation_time = 0.1  # in seconds

X_1 = interpolated_point_cloud = SPAI.construct_interpolated_point_cloud(P_positions, Q_positions, T, kappa)

t1, h_t1 = SPAI.calculate_omnidirectional_RIR(X_1, speed_of_sound, sampling_rate, max_simulation_time)

# Plot the RIR
plt.figure()
plt.plot(t1, h_t1)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Simulated Omnidirectional RIR')
plt.show()
