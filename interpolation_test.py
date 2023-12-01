import SPAIdermenLib as SPAI
import matplotlib.pyplot as plt

P = SPAI.getFromFile('140', 'room_impulse_responses.mat')
Q = SPAI.getFromFile('150', 'room_impulse_responses.mat')

R = SPAI.getFromFile('145', 'room_impulse_responses.mat')
kappa = 0.5

T, u, v = SPAI.cp_transport_plan(P, Q)

SPAI.visualize_interpolation(P, Q, T, u, v, kappa)

speed_of_sound = 343
sampling_rate = 48000
max_simulation_time = 0.1  # in seconds

X_1 = SPAI.construct_interpolated_point_cloud(P, Q, T, u, v, kappa)

t1, h_t1 = SPAI.calculate_omnidirectional_RIR(X_1, speed_of_sound, sampling_rate, max_simulation_time)
t2, h_t2 = SPAI.calculate_omnidirectional_RIR(R, speed_of_sound, sampling_rate, max_simulation_time)

# Plot the RIR
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t1, h_t1)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Simulated Omnidirectional RIR of interpolated point cloud betweem positions 140 and 150')

plt.subplot(2, 1, 2)
plt.plot(t2, h_t2)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Real RIR of position 145')
plt.show()