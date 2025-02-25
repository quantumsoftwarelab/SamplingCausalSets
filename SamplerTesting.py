
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

def plot_BD_action(sampler):
    """
    Plots the cost function for the transitive closure Hamiltonian.
    $H_{TC} = \sum_{i<j<k} C_{ij}C_{jk}(C_{ik} \oplus 1)$
    """
    _, unique_causal_matrix = get_unique_matrices(sampler.n)
    # The line `labels = ["".join(str(i) for i in list(np.frombuffer(mat,
    # dtype=np.int32).reshape(sampler.n, sampler.n)[np.triu_indices(sampler.n, 1)])) for mat in
    # unique_causal_matrix]` is creating labels for each unique causal matrix. Let's break it down:
    labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape( sampler.n,  sampler.n)[np.triu_indices( sampler.n, 1)])) for mat in  unique_causal_matrix]
    
    
    costs = np.zeros(len(unique_causal_matrix))
    costs_first_order_smearing = np.zeros(len(unique_causal_matrix))
    costs_first_order_taylor = np.zeros(len(unique_causal_matrix))
    for i, mat in enumerate( unique_causal_matrix):
        mat_ = np.frombuffer(mat, dtype=np.int32).reshape( sampler.n,  sampler.n)
        costs[i] =  calculate_action(mat_)
        costs_first_order_smearing[i] =  calculate_action(mat_, first_order_smearing = True)
        costs_first_order_taylor[i] =  calculate_action(mat_, first_order_smearing = True, first_order_taylor = True)
    
    argsort = np.argsort(labels)
    labels = np.array(labels)[argsort]
    costs = costs[argsort]
    costs_first_order_smearing = costs_first_order_smearing[argsort]
    costs_first_order_taylor = costs_first_order_taylor[argsort]
    
    expectation_values = np.array( analyse_BD_action_Hamiltonian())[argsort]
    
    plt.plot(labels, np.array(expectation_values), label = "H$_{BD_\epsilon}$ Expectation value")
    plt.plot(labels, costs, label = "BD$_\epsilon$ action", alpha = 0.5)
    plt.plot(labels, costs_first_order_smearing, label = "BD$_\epsilon$ action first order", alpha = 0.5)
    plt.plot(labels, costs_first_order_taylor, label = "BD$_\epsilon$ action first order taylor", alpha = 0.5)
    plt.xlabel("Causal matrix")
    plt.ylabel("Smeared BD action")
    plt.title("Approximation of smeared BD action")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def analyse_BD_action_Hamiltonian(sampler):
    expectation_values = []
    sampler.define_BD_circuit()
    
    op =  SparsePauliOp( sampler.Pauli_List,  sampler.coeffs_list)
    labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape( sampler.n, sampler.n)[np.triu_indices( sampler.n, 1)])) for mat in  unique_causal_matrix]
    for s in labels:
        qc = QuantumCircuit( sampler.q)
        for i, x in enumerate(s):
            if x == "1":
                qc.x(i)
                
        
        
        # Generate random parameter values for the circuit
        params = np.random.rand(qc.num_parameters)

        # Create a SparsePauliOp observable
        obs = op

        # Initialize QulacsEstimator
        qulacs_estimator = QulacsEstimator()

        # Run the estimation job with the circuit, observable, and parameters
        job = qulacs_estimator.run(qc, obs, params)

        # Get the result of the job
        result = job.result()

        # Retrieve the expectation value from the result
        expectation_value = result.values[0]

        # Print the expectation value
        #print("Expectation value:", expectation_value)
        
        #H = SparsePauliOp()
        #Hmat = H.to_matrix()
        #eig = np.linalg.eigvals(Hmat)
        #s
        # print(eig)
        expectation_values.append(expectation_value)
    return expectation_values

def analyse_output_bitstrings(sampler, repeats = 100, energy_ordering = True, title = "Transition Matrix", plot_grid = False, plot_cumulative = False):
    _, unique_causal_matrix = get_unique_matrices(sampler.n)
    
    
    
    mats = [np.frombuffer(mat, dtype=np.int32).reshape( sampler.n,  sampler.n) for mat in  unique_causal_matrix]
    print(" ")
    print("Number of unique causal matrices: ", len(mats))
    labels = ["".join(str(i) for i in mat[np.triu_indices( sampler.n, 1)]) for mat in mats]
    costs = np.zeros((2** sampler.q,2** sampler.q))
    BD_action = np.zeros(2** sampler.q)
    first_loop = True
    #for r in tqdm(range(repeats), desc="Repeats"):
    for s_pos, s in tqdm(enumerate(labels)):
        s_prime_list =  sampler.proposal(s, multiple = repeats)
        
        s_int = int(s, 2)
        
        for s_prime in s_prime_list:
            s_prime_int = int(s_prime, 2)
            costs[s_int, s_prime_int] += 1

        if first_loop:
            BD_action[s_int] =  calculate_action(mats[s_pos])
    first_loop = False
    

    
    
    
    
    
    
    if energy_ordering:
        BD_sorted_args = np.argsort(BD_action)
        sorted_BD = BD_action[BD_sorted_args]
        costs = costs[BD_sorted_args,:]
        costs = costs[:,BD_sorted_args]
    else:
        pass
    #print(np.max(costs))
    #print(costs)
    
    #reorder and normalise
    _costs = np.flipud(costs)/repeats
    
    cmap = colors.LinearSegmentedColormap.from_list('red_white', ['white', 'red'], N=256)
    
    
    
    if energy_ordering:
        non_zero_bd_index = np.where(sorted_BD != 0)[0][0]
        
        #plt.xticks(ticks=np.arange(len(sorted_BD)/5), labels=np.round(sorted_BD, 2), rotation=90)
        #plt.yticks(ticks=np.arange(len(sorted_BD)/5), labels=np.round(sorted_BD, 2))
        plt.xticks([])
        plt.yticks([])
        plt.ylim(non_zero_bd_index, 2** sampler.q)
        
        plt.imshow(_costs, extent=[0, 2** sampler.q, 0, 2** sampler.q], cmap=cmap, interpolation='nearest', norm=colors.LogNorm())
        
    else:
        plt.imshow(_costs, extent=[0, 2** sampler.q, 0, 2** sampler.q], cmap=cmap, interpolation='nearest', norm=colors.LogNorm())
    
    
    plt.colorbar(label='Transition Counts', norm=colors.LogNorm())
    #plt.colorbar(label='Transition Counts')
    plt.xlabel('Proposed Configuration (s\')')
    plt.ylabel('Initial Configuration (s)')
    plt.title(title)
    
    
    s_int_list = [int(s, 2) for s in labels]
    all_ints = np.arange(0, 2** sampler.q)
    
    #print("BD_sorted_args before: ", BD_sorted_args)
    #print("all_ints before: ", all_ints)
    if energy_ordering:
        all_ints = all_ints[BD_sorted_args]
        

        plt.plot([non_zero_bd_index, 2** sampler.q], [non_zero_bd_index,non_zero_bd_index] ,color='blue', linestyle='--', label='Non-zero BD')
        plt.plot([non_zero_bd_index,non_zero_bd_index], [non_zero_bd_index, 2**sampler.q] ,color='blue', linestyle='--', label='Non-zero BD')


    #print("all_ints after: ", all_ints)
    
    forbidden_count = 0
    for s_pos, s_int in enumerate(all_ints):
        if s_int not in s_int_list:
            for pos_i, i in enumerate(all_ints):
                if costs[pos_i, s_pos] > 0:
                #    plt.scatter(s_int + 0.5, i + 0.5, marker='x', color='blue')
                    forbidden_count += costs[pos_i, s_pos]
                #plt.scatter(s_pos + 0.5, pos_i + 0.5, marker='x', color='blue')
                #plt.scatter(pos_i + 0.5, s_pos + 0.5, marker='x', color='blue')
            #print("not allowed integer: ", s_int)"""
            
    print(" ")
    print(" --------------------- ")
    print(title)
    BD_transitions = np.zeros((2** sampler.q,2** sampler.q))
    for i in range(2** sampler.q):
        for j in range(2** sampler.q):
            BD_transitions[i,j] = np.abs(sorted_BD[i] - sorted_BD[j])
    
    if energy_ordering:
        total_bd_transition_cost = np.sum((BD_transitions[non_zero_bd_index:,non_zero_bd_index:] * costs[non_zero_bd_index:,non_zero_bd_index:]))
    else:
        total_bd_transition_cost = np.sum(BD_transitions * costs)
        print("THIS IS WRONG AS INCLUDES TRANSITIONS TO NON-CAUSAL MATRICES")
    total_transitions = np.sum(costs)
    
    self_transitions = np.sum(np.diag(costs))
    print(" ")
    print("Average BD transition cost: ", total_bd_transition_cost/(total_transitions-forbidden_count-self_transitions))
    print(" ")
    print("Frequency of self transitions: ", self_transitions / total_transitions)
    print("Frequency of forbidden transitions: ", forbidden_count / total_transitions)
    
    if plot_grid:
        plt.show()
    else:
        plt.close()
    fig, ax = plt.subplots()
    weighted_BD_transitions = np.repeat(BD_transitions[non_zero_bd_index:,non_zero_bd_index:].flatten(), costs[non_zero_bd_index:,non_zero_bd_index:].flatten().astype(int)).flatten()
    costs_no_self = np.copy(costs)
    np.fill_diagonal(costs_no_self, 0)
    weighted_BD_transitions_ignoring_self_transitions = np.repeat(BD_transitions[non_zero_bd_index:,non_zero_bd_index:].flatten(), costs_no_self[non_zero_bd_index:,non_zero_bd_index:].flatten().astype(int)).flatten()
    ax.hist(weighted_BD_transitions, bins=50, density=True, cumulative=True, histtype='step', label='Cumulative Probability')
    ax.hist(weighted_BD_transitions_ignoring_self_transitions, bins=50, density=True, cumulative=True, histtype='step', label='Cumulative Probability (no self transitions)')
    ax.set_xlabel('BD Transitions')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Probability of BD Transitions')
    ax.legend()
    
    if plot_cumulative:
        plt.show()
    else:
        plt.close()
    
    
    print(" --------------------- ")
    print(" ")
    
    return weighted_BD_transitions_ignoring_self_transitions



cardinality = 5

transitions = []
titles = []

title = "Mixing only"
titles.append(title)
Qsamp = Sampler(cardinality, method="quantum", qargs = {"TC":False, "BD":False, "mixing_time":0.1, "t":5})
transitions.append(analyse_output_bitstrings(Qsamp, repeats = 10000, energy_ordering=True, title = title, plot_grid = False, plot_cumulative = False))


title = "Mixing and BD"
titles.append(title)
Qsamp = Sampler(cardinality, method="quantum", qargs = {"TC":False, "BD":True, "mixing_time":0.1, "t":5})
transitions.append(analyse_output_bitstrings(Qsamp, repeats = 10000, energy_ordering=True, title = title, plot_grid = False, plot_cumulative = False))

title = "Mixing and TC"
titles.append(title)
Qsamp = Sampler(cardinality, method="quantum", qargs = {"TC":True, "BD":False, "mixing_time":0.1, "t":5})
transitions.append(analyse_output_bitstrings(Qsamp, repeats = 10000, energy_ordering=True, title = title, plot_grid = False, plot_cumulative = False))

title = "Mixing, TC and BD"
titles.append(title)
Qsamp = Sampler(cardinality, method="quantum", qargs = {"TC":True, "BD":True, "mixing_time":0.1, "t":5})
transitions.append(analyse_output_bitstrings(Qsamp, repeats = 10000, energy_ordering=True, title = title, plot_grid = False, plot_cumulative = False))

for i in range(len(transitions)):
    plt.hist(transitions[i], bins=50, density=True, cumulative=True, histtype='step', label=titles[i])

plt.title('Cumulative Probability of BD Transitions for each hamiltonian')
plt.ylabel('Cumulative Probability')
plt.xlabel('BD Transition energy $\Delta S_{BD_\epsilon}$')

plt.legend()
plt.show()

#plot_BD_action(Qsamp)
