
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from matplotlib import colors
from qiskit.circuit import QuantumCircuit
from qiskit_qulacs.qulacs_estimator import QulacsEstimator
from helpers import *
from Sampler import Sampler
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector, PauliList
from tqdm import tqdm


cardinality = 5

number_of_samples = 100
#unique_causal_matrices = get_unique_matrices(cardinality)[1]



C_baseline_samp = Sampler(cardinality, method="classical", cargs = {"link_move":True, "relation_move":True})
c_output_chain = C_baseline_samp.sample(s = np.ones(C_baseline_samp.q), num_samples = 1000, sample_frequency=100, T_therm=1000, accept_all_legitimate_moves=True)
print("Average height of chain: ", np.mean(c_output_chain["heights"]))
print("Average ordering fraction of chain: ", np.mean(c_output_chain["ordering_fractions"]))
print("Average number of relations of chain: ", np.mean(c_output_chain["num_relations"]))
print("Average number of minimal elements of chain: ", np.mean(c_output_chain["minimal_elements"]))



Qsamp = Sampler(cardinality, method="quantum", qargs = {"gammas":[0.9, 0, 0.1], "t":10})
q_output_chain = Qsamp.sample(s = np.ones(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)
q_output_antichain = Qsamp.sample(s = np.zeros(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)

Csamp = Sampler(cardinality, method="classical", cargs = {"link_move":True, "relation_move":True})
c_output_chain = Csamp.sample(s = np.ones(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)
c_output_antichain = Csamp.sample(s = np.zeros(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)

sweepsize = 2*cardinality**3
generic_sample_index_sweep = q_output_chain["sample_index"]/sweepsize
#print(c_output)
fig, axs = plt.subplots(2)
fig.suptitle("Height vs Sample Index")

axs[0].set_title("Quantum")
axs[0].plot(generic_sample_index_sweep, q_output_chain["heights"], label="chain")
axs[0].plot(generic_sample_index_sweep, q_output_antichain["heights"], label="antichain")


axs[1].set_title("Classical")
axs[1].plot(generic_sample_index_sweep, c_output_chain["heights"], label="chain")
axs[1].plot(generic_sample_index_sweep, c_output_antichain["heights"], label="antichain")
axs[0].legend()



fig, axs = plt.subplots(2)
fig.suptitle("ordering_fractions vs Sample Index")

axs[0].set_title("Quantum")
axs[0].plot(generic_sample_index_sweep, q_output_chain["ordering_fractions"], label="chain")
axs[0].plot(generic_sample_index_sweep, q_output_antichain["ordering_fractions"], label="antichain")


axs[1].set_title("Classical")
axs[1].plot(generic_sample_index_sweep, c_output_chain["ordering_fractions"], label="chain")
axs[1].plot(generic_sample_index_sweep, c_output_antichain["ordering_fractions"], label="antichain")

axs[0].legend()





fig, axs = plt.subplots(2)
fig.suptitle("Number of relations vs Sample Index")

axs[0].set_title("Quantum")
axs[0].plot(generic_sample_index_sweep, q_output_chain["num_relations"], label="chain")
axs[0].plot(generic_sample_index_sweep, q_output_antichain["num_relations"], label="antichain")


axs[1].set_title("Classical")
axs[1].plot(generic_sample_index_sweep, c_output_chain["num_relations"], label="chain")
axs[1].plot(generic_sample_index_sweep, c_output_antichain["num_relations"], label="antichain")
axs[0].legend()





fig, axs = plt.subplots(2)
fig.suptitle("Number of minimal elements vs Sample Index")

axs[0].set_title("Quantum")
axs[0].plot(generic_sample_index_sweep, q_output_chain["minimal_elements"], label="chain")
axs[0].plot(generic_sample_index_sweep, q_output_antichain["minimal_elements"], label="antichain")


axs[1].set_title("Classical")
axs[1].plot(generic_sample_index_sweep, c_output_chain["minimal_elements"], label="chain")
axs[1].plot(generic_sample_index_sweep, c_output_antichain["minimal_elements"], label="antichain")
axs[0].legend()



plt.show()
