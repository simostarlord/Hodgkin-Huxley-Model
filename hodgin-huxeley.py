import numpy as np
import matplotlib.pyplot as plt

# Constants
C_m = 1.0  # Membrane capacitance (uF/cm^2)
g_Na = 120.0  # Maximum sodium conductance (mS/cm^2)
g_K = 36.0  # Maximum potassium conductance (mS/cm^2)
g_L = 0.3  # Leak conductance (mS/cm^2)
E_Na = 50.0  # Sodium reversal potential (mV)
E_K = -77.0  # Potassium reversal potential (mV)
E_L = -54.4  # Leak reversal potential (mV)

# Time parameters
dt = 0.01  # Time step (ms)
T = 100  # Total simulation time (ms)
time = np.arange(0, T + dt, dt)

# Stimulus: Pulses with noise
I = np.zeros(len(time))
I[100:400] = 10.0
I += np.random.normal(0, 0.5, size=len(time))  # Add noise


# Functions for gating variables
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)


def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))


def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)


# Initialize state variables
V = -65.0
m = alpha_m(V) / (alpha_m(V) + beta_m(V))
h = alpha_h(V) / (alpha_h(V) + beta_h(V))
n = alpha_n(V) / (alpha_n(V) + beta_n(V))

# Adaptive Threshold Parameters
threshold = -55.0
threshold_decay = 0.05
spike_amplitude = 20.0
refractory_period = 2.0  # ms
last_spike_time = -refractory_period

# Store values for plotting
V_trace = []
spike_times = []

# Simulation loop
for t_idx, t in enumerate(time):
    # Ionic currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # Adaptive threshold increases after spike
    if t - last_spike_time < refractory_period:
        I[int(t_idx)] = 0  # No input during refractory period

    # Hodgkin-Huxley equation
    dV = (I[t_idx] - (I_Na + I_K + I_L)) / C_m
    V += dV * dt

    # Update gating variables
    m += (alpha_m(V) * (1 - m) - beta_m(V) * m) * dt
    h += (alpha_h(V) * (1 - h) - beta_h(V) * h) * dt
    n += (alpha_n(V) * (1 - n) - beta_n(V) * n) * dt

    # Spike detection and refractory period handling
    if V >= threshold and t - last_spike_time > refractory_period:
        V = spike_amplitude  # Generate spike
        last_spike_time = t
        spike_times.append(t)
        threshold += 5  # Adaptive threshold increase after spike

    # Threshold decays over time (adaptive behavior)
    threshold -= threshold_decay * dt
    threshold = max(-55, threshold)

    # Store values for plotting
    V_trace.append(V)

# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(time, V_trace, label='Membrane Potential (mV)', color='blue')
for spike_time in spike_times:
    plt.axvline(x=spike_time, color='red', linestyle='--', alpha=0.6)
plt.axhline(-65, linestyle='--', color='gray', label="Resting Potential")
plt.axhline(-55, linestyle='--', color='green', label="Threshold Potential")
plt.title('Enhanced Hodgkin-Huxley Model with Noise, Refractory Period, and Adaptive Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.show()
