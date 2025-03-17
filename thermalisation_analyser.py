
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
import scipy as sp

def cumulative_plot(sweeps, values ,mean = None, label = None, ax = None):
    if ax == None:
        fig, ax = plt.subplots()
        
    if isinstance(mean, float):
        ax.plot([np.min(sweeps), np.max(sweeps)], [mean,mean], color = "k", linestyle = "--", label = "Mean")
    if isinstance(mean, tuple) or isinstance(mean, list):
        mean, std = mean
        ax.plot([np.min(sweeps), np.max(sweeps)], [mean,mean], color = "k", linestyle = "--", label = "Mean")
        ax.fill_between(sweeps, mean - std, mean + std, alpha=0.2)

    cumulative_values = np.cumsum(values) / np.arange(1, len(values) + 1)  
    std_err = np.array([sp.stats.sem(values[:i+1]) for i in range(len(values))])
    ax.plot(sweeps, cumulative_values,  label = label)
    
    
    
    ax.fill_between(sweeps, cumulative_values - std_err, cumulative_values + std_err, alpha=0.2)





def cumulative_plots(output_chains, observables, labels, means = None, sweepsize = 1):
    fig, axs = plt.subplots(len(observables)//2,2)
    fig.suptitle("Cumulative plots of some observables")
    for j, strin in enumerate(observables):
        ax = axs[j%2, j//2]
        ax.set_ylabel(strin)
        ax.set_xlabel("Sweeps")
        
        for i, output_chain in enumerate(output_chains):
            if means != None and i == 0:
                cumulative_plot(output_chain["sample_index"]/sweepsize, output_chain[strin], mean = means[j], label = labels[i], ax = ax)
            else:
                cumulative_plot(output_chain["sample_index"]/sweepsize, output_chain[strin], label = labels[i],ax = ax)
                
                
    handles, labels = axs[0,0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())  
    plt.show()




cardinality = 5
sweepsize = 2*cardinality**3
print("sweepsize: ", sweepsize)
number_of_samples = 1000
#unique_causal_matrices = get_unique_matrices(cardinality)[1]


C_baseline_samp = Sampler(cardinality, method="classical", cargs = {"link_move":True, "relation_move":True})
c_output_chain = C_baseline_samp.sample(s = np.ones(C_baseline_samp.q), num_samples = 100, sample_frequency=sweepsize*10, T_therm=1000, accept_all_legitimate_moves=True)




mean_height = np.mean(c_output_chain["heights"])
err_height = sp.stats.bootstrap((c_output_chain["heights"],), np.mean, n_resamples=1000, confidence_level=0.99, method='percentile').standard_error

mean_ordering_fraction = np.mean(c_output_chain["ordering_fractions"])
err_ordering_fraction = sp.stats.bootstrap((c_output_chain["ordering_fractions"],), np.mean, n_resamples=1000, confidence_level=0.99, method='percentile').standard_error

mean_num_relations = np.mean(c_output_chain["num_relations"])
bootstrap_num_relations = sp.stats.bootstrap((c_output_chain["num_relations"],), np.mean, n_resamples=1000, confidence_level=0.99, method='percentile')
err_num_relations = bootstrap_num_relations.standard_error

mean_num_minimal_elements = np.mean(c_output_chain["minimal_elements"])
err_num_minimal_elements = sp.stats.bootstrap((c_output_chain["minimal_elements"],), np.mean, n_resamples=1000, confidence_level=0.99, method='percentile').standard_error

print("Average height of chain: ", mean_height, "+/- ", err_height)
print("Average ordering fraction of chain: ", mean_ordering_fraction, "+/- ", err_ordering_fraction)
print("Average number of relations of chain: ",  mean_num_relations, "+/- ", err_num_relations)
print("Average number of minimal elements of chain: ", mean_num_minimal_elements, "+/- ", err_num_minimal_elements)






Qsamp = Sampler(cardinality, method="quantum", qargs = {"gammas":[0.9, 0, 0.1], "t":10})
q_output_chain = Qsamp.sample(s = np.ones(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)
q_output_antichain = Qsamp.sample(s = np.zeros(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)

Csamp = Sampler(cardinality, method="classical", cargs = {"link_move":True, "relation_move":True})
c_output_chain = Csamp.sample(s = np.ones(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)
c_output_antichain = Csamp.sample(s = np.zeros(Qsamp.q), num_samples = number_of_samples, sample_frequency=1, T_therm=0, accept_all_legitimate_moves=True)

observables = ["heights", "ordering_fractions", "num_relations", "minimal_elements"]
labels = ["classical_chain", "quantum_chain", "classical_antichain", "quantum_antichain"]
output_chains = [c_output_chain, q_output_chain, c_output_antichain, q_output_antichain] 
means = [(mean_height, err_height), (mean_ordering_fraction, err_ordering_fraction), (mean_num_relations, err_num_relations), (mean_num_minimal_elements, err_num_minimal_elements)]
cumulative_plots(output_chains, observables, labels = labels, means = means, sweepsize = sweepsize)

    
#cumulative_plot(c_output_chain["sample_index"]/sweepsize, c_output_chain["heights"], mean = (mean_height, std_height), label = "Classical chain")
#cumulative_plot(q_output_chain["sample_index"]/sweepsize, q_output_chain["heights"], label = "Quantum chain")
#cumulative_plot(c_output_antichain["sample_index"]/sweepsize, c_output_antichain["heights"], label = "Classical antichain")
#cumulative_plot(q_output_antichain["sample_index"]/sweepsize, q_output_antichain["heights"], label = "Quantum antichain")
plt.show()


"""
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
"""