import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

params = {"t_0": 0, "dt": 1/10, "v": -60, "V_0": -60, "tau": 2, "V_thresh": 20, "A_pl": 0.05, "A_mn": 0.05, "tau_pl": 20, "tau_mn": 20, "learning_rate": 0.01}
all_neurons = {}
all_synapses = {}
t = 0
duration = 120
time_window = [5, 110]


class LIFNeuron:
    def __init__(self, name, V_thresh=params["V_thresh"]):
        self.v = params["v"]
        self.V_0 = params["V_0"]
        self.tau = params["tau"]
        self.dt = params["dt"]
        self.V_thresh = V_thresh
        self.name = name
        self.external_input = 0
        self.voltages = []
        self.times = []
        self.spike_times = []
        self.connections: List[tuple["LIFNeuron", "Synapse"]] = []

        assert self.name not in all_neurons, f"Neurons with the same name ({self.name}) cannot exist"
        all_neurons[self.name] = self

    def connect(self, target_neuron: "LIFNeuron", synapse: "Synapse"):
        self.connections.append((target_neuron, synapse))
        print(f"{self.name} connected to {target_neuron.name} with synapse {synapse.name}")

    def receive_input(self, current, current_time):
        print(f"{self.name} received input current of {current} at {current_time.__round__(3)}, initial value: {self.external_input}", end="")
        self.external_input += current
        print(f", total value: {self.external_input.__round__(3)}")

    def fire(self, current_time):
        print(f"{self.name} fires at t={current_time.__round__(3)}")
        for target_neuron, synapse in self.connections:
            target_neuron.receive_input(synapse.weight, current_time)

    def exponential_decay_2_neurons(self, target_n):
        global t
        vs1 = []
        Is1 = []
        times1 = []
        vs2 = []
        Is2 = []
        times2 = []
        t = 0
        while t < duration:
            vs1.append(self.v)
            times1.append(t)
            vs2.append(target_n.v)
            times2.append(t)

            dv1 = - (self.v - self.V_0) / self.tau
            dv2 = - (target_n.v - target_n.V_0) / target_n.tau
            if time_window[0] < t < time_window[1]:
                random.seed(t * 21 - 76 + np.exp(t + 2))
                noise1 = random.uniform(1.5, 7.2)
                random.seed(np.exp(2*t-7) + 8*t - 92 + t//6)
                noise2 = random.uniform(1.5, 7.2)
                if t.__round__(1) == 25.1:
                    self.receive_input(20, t)
                I1 = self.external_input + noise1
                I2 = target_n.external_input + noise2
                dv1 += I1
                dv2 += I2
                Is1.append(I1)
                Is2.append(I2)
            else:
                Is1.append(0)
                Is2.append(0)

            self.v += dv1 * self.dt
            target_n.v += dv2 * target_n.dt

            if self.v >= self.V_thresh:
                self.spike_times.append(t)
                self.fire(t)
                self.v = self.V_0

            if target_n.v >= target_n.V_thresh:
                target_n.spike_times.append(t)
                target_n.fire(t)
                target_n.v = target_n.V_0
                target_n.external_input = 0

            # self.external_input = 0
            # target_n.external_input = 0

            t += self.dt

        return (
            np.array(times1), np.array(vs1), np.array(self.spike_times), np.array(Is1),
            np.array(times2), np.array(vs2), np.array(target_n.spike_times), np.array(Is2)
        )


class Synapse:
    def __init__(self, name, pre_neuron: LIFNeuron, post_neuron: LIFNeuron, weight: float):
        self.name = name
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.A_pl = params["A_pl"]  # mV
        self.A_mn = params["A_mn"]  # mV
        self.tau_pl = params["tau_pl"]  # ms
        self.tau_mn = params["tau_mn"]  # ms

        assert self.name not in all_synapses, f"Synapses with the same name ({self.name}) cannot exist"
        all_synapses[self.name] = self

    def update_weight(self, pre_spike_times, post_spike_times):
        assert len(pre_spike_times) > 0, "pre_spike_times cannot be an empty list"
        assert len(post_spike_times) > 0, "post_spike_times cannot be an empty list"
        deltaW = 0
        # learning_rate = params["learning_rate"]

        # Truncate spike times to the same size, then calculate STDP for each corresponding spike
        min_length = min(len(pre_spike_times), len(post_spike_times))
        pre_spike_times = pre_spike_times[:min_length]
        post_spike_times = post_spike_times[:min_length]
        for pre_time, post_time in zip(pre_spike_times, post_spike_times):
            delta_t = pre_time - post_time
            if delta_t <= 0:
                deltaW += self.A_pl * np.exp(delta_t / self.tau_pl)
                print(f"Weight between {self.pre_neuron.name} and {self.post_neuron.name} increasing by {deltaW}")
            else:
                deltaW += -self.A_mn * np.exp(-delta_t / self.tau_mn)
                print(f"Weight between {self.pre_neuron.name} and {self.post_neuron.name} decreasing by {deltaW}")

                # self.weight = max(0.0, min(1.0, self.weight))
            # deltaW *= learning_rate
            print(f"Weight between {self.pre_neuron.name} and {self.post_neuron.name} updated from {self.weight}",
                  end="")
            self.weight += deltaW
            print(f", to {self.weight}")


# Create neuron and get data
n1 = LIFNeuron("n1")
n2 = LIFNeuron("n2")
s12 = Synapse("s12", n1, n2, 18)
n1.connect(n2, s12)
n1.receive_input(15, t)
# times1, voltages1, spikes1 = n1.exponential_decay()
# times2, voltages2, spikes2 = n2.exponential_decay()
times1, voltages1, spikes1, current1, times2, voltages2, spikes2, current2 = n1.exponential_decay_2_neurons(n2)

# # Create animation for fig 1
# fig1, (ax11, ax12) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
#
# # Plot 1: Voltage over time for first neuron
# line11, = ax11.plot([], [], lw=2)
# ax11.axhline(y=n1.V_thresh, color='gray', linestyle='--', label='Threshold')
# ax11.set_ylabel("Voltage (mV)")
# ax11.set_xlim(0, times1[-1])
# ax11.set_ylim(min(voltages1)-5, max(voltages1)+5)
# ax11.set_title("Membrane Potential Over Time for Neuron 1")
#
# # Plot 2: Voltage over time for second neuron
# line12, = ax12.plot([], [], lw=2)
# ax12.set_ylabel("Voltage (mV)")
# ax12.set_xlabel("Time (ms)")
# ax12.set_xlim(0, times2[-1])
# ax12.set_ylim(min(voltages1)-5, max(voltages1)+5)
# ax12.set_title("Membrane Potential Over Time for Neuron 2")
#
# # Create animation for fig 2
# fig2, (ax21, ax22) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
#
# # Plot current over time for first neuron
# line21, = ax21.plot([], [], lw=2)
# ax21.set_ylabel("Current")
# ax21.set_xlim(0, times1[-1])
# ax21.set_ylim(min(current1)-5, max(current1)+5)
# ax21.set_title("Current Over Time for Neuron 1")
#
# # Plot current over time for second neuron
# line22, = ax22.plot([], [], lw=2)
# ax22.set_ylabel("Current")
# ax22.set_xlabel("Time (ms)")
# ax22.set_xlim(0, times1[-1])
# ax22.set_ylim(min(current1)-5, max(current1)+5)
# ax22.set_title("Current Over Time for Neuron 2")

# Plot membrane potential over time for both neurons
fig1, (ax11, ax12) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot 1: Neuron 1
ax11.plot(times1, voltages1, lw=2, label="Neuron 1 Voltage")
ax11.axhline(y=n1.V_thresh, color='gray', linestyle='--', label='Threshold')
ax11.set_ylabel("Voltage (mV)")
ax11.set_xlim(0, times1[-1])
ax11.set_ylim(min(voltages1) - 5, max(voltages1) + 5)
ax11.set_title("Membrane Potential Over Time for Neuron 1")
ax11.legend()

# Plot 2: Neuron 2
ax12.plot(times2, voltages2, lw=2, label="Neuron 2 Voltage")
ax12.set_ylabel("Voltage (mV)")
ax12.set_xlabel("Time (ms)")
ax12.set_xlim(0, times2[-1])
ax12.set_ylim(min(voltages2) - 5, max(voltages2) + 5)
ax12.set_title("Membrane Potential Over Time for Neuron 2")
ax12.legend()

plt.tight_layout()
plt.savefig('neuron-membrane-potential.png')
plt.show()

# Plot current over time for both neurons
fig2, (ax21, ax22) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot 1: Neuron 1 Current
ax21.plot(times1, current1, lw=2, label="Neuron 1 Current")
ax21.set_ylabel("Current")
ax21.set_xlim(0, times1[-1])
ax21.set_ylim(min(current1) - 5, max(current1) + 5)
ax21.set_title("Current Over Time for Neuron 1")
ax21.legend()

# Plot 2: Neuron 2 Current
ax22.plot(times2, current2, lw=2, label="Neuron 2 Current")
ax22.set_ylabel("Current")
ax22.set_xlabel("Time (ms)")
ax22.set_xlim(0, times2[-1])
ax22.set_ylim(min(current2) - 5, max(current2) + 5)
ax22.set_title("Current Over Time for Neuron 2")
ax22.legend()

plt.tight_layout()
plt.savefig('neuron-currents.png')
plt.show()

#
# def update(frame):
#     line11.set_data(times1[:frame], voltages1[:frame])
#     line12.set_data(times2[:frame], voltages2[:frame])
#     line21.set_data(times1[:frame], current1[:frame])
#     line22.set_data(times2[:frame], current2[:frame])
#     return line11, line12, line21, line22


# ani = animation.FuncAnimation(
#     fig1, update, frames=len(times1), interval=20, blit=True
# )

