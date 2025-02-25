# Code to constrain the space of all causal sets, Omega to that of transitively closed matrices (i.e. causal matrices) through hamiltonian constraints. 

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_qulacs import QulacsProvider
from qiskit_qulacs.qulacs_backend import QulacsBackend
import math

from collections import defaultdict


from matplotlib import colors
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector, PauliList
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer import StatevectorSimulator
from tqdm import tqdm
from qiskit_qulacs.qulacs_estimator import QulacsEstimator

from helpers import *

class Sampler:
    
    def __init__(self, n:int, method: str = "quantum", qargs: dict = {}):
        """
        Initialises the Sampler class.
        
        Parameters:
        n (int): The cardinality of the causal set.
        method (str): The method to use for sampling. Options are "quantum" and "classical".
        qargs (dict): A dictionary of arguments to pass to the quantum, including the follwoing:
            TC (bool): If True, add Transitive Closure to Hamiltonian. This restricts output bitstrings to 
                those that represent causal matrices Default is True.
            BD (bool): If True, add BD action approximation to Hamiltonian. Default is True.
            mixing_time (float): The time parameter for the mixing circuit. Default is 0.1. 0 if no mixing.
                This will be depreciated and replaced by gamma parameters for TC, BD, and mixing
            t (int): The number of time steps to evolve the Hamiltonian. Default is 5.
        """
        
        self.n = n # number of elements in the causal set
        
        if method == "quantum":
            self.q = int(n*(n-1)/2) # number of qubits (upper triangular matrix)
            self.base_BD_circuit = self.define_BD_circuit()
            self.base_TC_circuit = self.define_TC_circuit()
            #self.mixing_circ = self.build_mixing_circuit().decompose()
            self.proposal = self.quantum_proposal
            
            if not qargs:
                print("qargs dictionary is empty. Proceeding with default values.")
            
            if "TC" in qargs:
                self.TC = qargs["TC"]
                if type(self.TC) != bool:
                    raise ValueError("TC must be a boolean value.")
            else:
                self.TC = True
                
            if "BD" in qargs:
                self.BD = qargs["BD"]
                if type(self.BD) != bool:
                    raise ValueError("BD must be a boolean value.")
            else:
                self.BD = True
            
            if "mixing_time" in qargs:
                self.mixing_time = qargs["mixing_time"]
                if type(self.mixing_time) != float:
                    if type(self.mixing_time) != int:
                        if type(self.mixing_time) != tuple:
                            raise ValueError("mixing_time must be a float corresponding to the mixing time or a tuple representing a range of mixing times from which to sample.") 
                elif type(self.mixing_time) == tuple: 
                    raise ValueError("Tuple mixing times not yet implemented.")   
            else:
                self.mixing_time = 0.1
                
                
            if "t" in qargs:
                self.t = qargs["t"]
                if type(self.t) != int:
                    raise ValueError("t must be an integer value.")
                else:
                    self.t = 5
                
            

            
        elif method == "classical":
            raise ValueError("Classical method not yet implemented.")
            self.proposal = self.classical_proposal
            
        else:
            raise ValueError("Invalid method. Choose 'quantum' or 'classical'.")



    def define_BD_circuit(self,  epsilon: float = 0.1, evo_time:1 = 10)-> QuantumCircuit:
        """
        Define a quantum circuit for time evolution of the approximated BD action.
            
        This function constructs a quantum circuit based on the BD model by defining various 
        Pauli operators and their corresponding time. The circuit is built by iterating over 
        pairs and triples of indices, applying the appropriate operators, and adding them 
        to the circuit. The Pauli operators and their coefficients are stored in the instance 
        variables `Pauli_List` and `coeffs_list` respectively.
        
        Parameters:
        epsilon (float): A small parameter used to smear the action over subgraphs. Default is 0.1.
        evo_time (int): The total evolution time for the circuit. Default is 10. 
            EVO TIME NEEDS FIXED. MUST BE NORMALISED ETC. ANALYTICALLY (*NOT DONE YET")
        
        Returns:
        QuantumCircuit: The constructed quantum circuit after applying the evolution operators.

        """
        
        
        
        # define the map from the relation between ith and jth element
        # to the index of the corresponding qubit in the quantum circuit
        basis = get_upper_triangular_basis(self.n)
        
        
        # Using operator like this is probably very inefficient re. matrix exponential. But will work for now.
        # Using sparse pauli op is definitely better
        
        pauli_list =[]
        coeffs = []
        circuit = QuantumCircuit(self.q)
        
        # cardinality term 
        # Before outer loop
        evo_time_prime_N = 2*epsilon*evo_time*self.n
        Z_string = np.zeros(self.q)
        operator = Pauli((Z_string, np.zeros(self.q), 0))
        pauli_list.append(operator)
        coeffs.append(evo_time_prime_N/evo_time)
        circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_N)
        
        for i in range(self.n):
            for k in range(i+1, self.n):                
                
                # constant term, inside outer loop
                #-2epsilon**2 (1+epsilon)
                evo_time_prime_const = 2*epsilon**2*(1)*evo_time
                Z_string = np.zeros(self.q)
                operator = Pauli((Z_string, np.zeros(self.q), 2))
                pauli_list.append(operator)
                coeffs.append(evo_time_prime_const/evo_time)
                circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_const)
                
                # single ik term
                # 2epsilon**2 Z_ik (1)
                evo_time_prime_one_body = 2*epsilon**2*(1)*evo_time
                Z_string = np.zeros(self.q)
                Z_string[basis[i,k]] = 1
                operator = Pauli((Z_string, np.zeros(self.q), 0))
                pauli_list.append(operator)
                coeffs.append(evo_time_prime_one_body/evo_time)
                circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_one_body)
                
                
                # Inner loop (over all j for which i<j<k)
                evo_time_prime_inner = (4/8)*(epsilon**3)*evo_time
                
                for j in range(i+1, k):
                    # Inner loop constant term
                    # 1
                    Z_string_0 = np.zeros(self.q)
                    operator_0 = Pauli((Z_string_0, np.zeros(self.q), 0))
                    
                    
                    # single body terms
                    # -Zij
                    Z_string_1 = np.zeros(self.q)
                    Z_string_1[basis[i,j]] = 1
                    operator_1 = Pauli((Z_string_1, np.zeros(self.q), 2))
                    
                    # -Zjk
                    Z_string_2 = np.zeros(self.q)
                    Z_string_2[basis[j,k]] = 1
                    operator_2 = Pauli((Z_string_2, np.zeros(self.q), 2))
                    
                    #-Zik
                    Z_string_3 = np.zeros(self.q)
                    Z_string_3[basis[i,k]] = 1
                    operator_3 = Pauli((Z_string_3, np.zeros(self.q), 2))
                    
                    
                    # Two body terms
                    #Zij Zjk
                    Z_string_4 = np.zeros(self.q)
                    Z_string_4[basis[i,j]] = 1
                    Z_string_4[basis[j,k]] = 1
                    operator_4 = Pauli((Z_string_4, np.zeros(self.q), 0))
                    
                    #Zik Zij
                    Z_string_5 = np.zeros(self.q)
                    Z_string_5[basis[i,j]] = 1
                    Z_string_5[basis[i,k]] = 1
                    operator_5 = Pauli((Z_string_5, np.zeros(self.q), 0))
                    
                    # ZikZjk
                    Z_string_6 = np.zeros(self.q)
                    Z_string_6[basis[j,k]] = 1
                    Z_string_6[basis[i,k]] = 1
                    operator_6 = Pauli((Z_string_6, np.zeros(self.q), 0))
                    
                    # Three body term
                    # -ZikZjkZij
                    Z_string_7 = np.zeros(self.q)
                    Z_string_7[basis[i,j]] = 1
                    Z_string_7[basis[j,k]] = 1
                    Z_string_7[basis[i,k]] = 1
                    operator_7 = Pauli((Z_string_7, np.zeros(self.q), 2))
                    
                    
                    #operator_list = [operator_0, operator_1, operator_2, operator_3, operator_4, operator_5, operator_6, operator_7]
                    operator_list = [operator_0,operator_1, operator_2, operator_3, operator_4, operator_5, operator_6, operator_7]
                    for operator in operator_list:
                        circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_inner)
                        pauli_list.append(operator)
                        
                        coeffs.append(evo_time_prime_inner/evo_time)
                        
                        
        self.Pauli_List = PauliList(pauli_list)
        self.coeffs_list = coeffs
        return circuit.decompose()

    def define_TC_circuit(self)-> QuantumCircuit:
        """
        Defines a quantum circuit to exhibit TC (Transitive closure) using a combination of one-body, two-body, 
        and three-body Pauli operators. 

        Parameters:
            None
        
        Returns:
            QuantumCircuit: The constructed quantum circuit after decomposition.
        """
        
        
        
        basis = get_upper_triangular_basis(self.n)
        
        
        #Using operator like this is probably very inefficient re. matrix exponential. But will work for now.
        # Using sparse pauli op is definately better
        circuit = QuantumCircuit(self.q)
        evo_time = 0.5
        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    
                    
                    
                    
                    
                    one_body_terms = [[i,j],[j,k],[i,k]]
                    one_body_signs = [2,2,0]
                    # One body terms
                    for l in range(3):
                        
                        Z_string = np.zeros(self.q)
                        Z_string[basis[one_body_terms[l][0],one_body_terms[l][1]]] = 1
                        operator = Pauli((Z_string, np.zeros(self.q), one_body_signs[l]))
                        circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time)
                    
                    two_body_terms = [[[i,j],[j,k]],[[i,j],[i,k]],[[j,k],[i,k]]]
                    two_body_signs = [0,2,2]
                    # Two body terms
                    for l in range(3):
                        Z_string = np.zeros(self.q)
                        #print(" ")
                        #print(two_body_terms[l][0][0])
                        Z_string[basis[two_body_terms[l][0][0],two_body_terms[l][0][1]]] = 1
                        Z_string[basis[two_body_terms[l][1][0],two_body_terms[l][1][1]]] = 1
                        operator = Pauli((Z_string, np.zeros(self.q), two_body_signs[l]))
                        circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time)
                    
                    
                    
                    #three body terms
                    Z_string = np.zeros(self.q)
                    Z_string[basis[i,j]] = 1
                    Z_string[basis[j,k]] = 1
                    Z_string[basis[i,k]] = 1
                    operator = Pauli((Z_string, np.zeros(self.q), 0))
                    circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time)
        return circuit.decompose()


    def define_mixing_circuit(self, mixing_time: float = 0.1) -> QuantumCircuit:
        # simple X mixer as in Layden
        circuit = QuantumCircuit(self.q)
        for l in range(self.q):         
            Z_string = np.zeros(self.q)
            X_string = np.zeros(self.q)
            X_string[l] = 1
            operator = Pauli((Z_string, X_string, 0))
            circuit = self.add_evo_to_circuit(circuit, operator, time = mixing_time)

        return circuit.decompose()



    def add_evo_to_circuit(self, circuit: QuantumCircuit, op:Pauli, time: float = 0.2):
        evo = PauliEvolutionGate(op, time=time)
        circuit.append(evo, range(self.q))
        return circuit

    def quantum_proposal(self, s: str, multiple: int = 1) -> str | list:
        """
        Propose a new configuration based on the given bitstring.
        This function generates a new bitstring configuration by taking a single shot of a quantum circuit.
        The quantum circuit is time evolution of a Hamiltonian that (optionally) includes mixing, 
        transitive closure, and BD action terms.
        
        Parameters:
        s (str): The initial bitstring configuration.
        
        multiple (int): The number of samples to generate. Default is 1. More than one can be useful for analysing the proposal.
        
        
        Returns:
        str or list: If `multiple` is 1, returns a single new bitstring configuration.
            If `multiple` is greater than 1, returns a list of new bitstring configurations.
        """
        mixing_circ = self.define_mixing_circuit(mixing_time = self.mixing_time)
        
        
        #set initial state
        qc = QuantumCircuit(self.q)
        for i, x in enumerate(s):
            if x == "1":
                qc.x(i)
                
        #If doing transitive closure
        if self.TC:
            qc.compose(self.base_TC_circuit, inplace = True)
        
        # If doing BD action
        if self.BD:
            qc.compose(self.base_BD_circuit, inplace = True)
        
        
        
        for t in range(self.t):
            #Always mix
            qc.compose(mixing_circ, inplace = True)
            if self.TC:
                qc.compose(self.base_TC_circuit, inplace = True)
            if self.BD:
                qc.compose(self.base_BD_circuit, inplace = True)

        # Use Qiskit-Qulacs to run the circuit
        backend = QulacsBackend()

        result = backend.run(qc, shots = multiple).result()

        if multiple == 1:
            s_prime = list(result.data()["counts"].keys())[0]
            return s_prime[::-1]
        elif multiple > 1:
            s_primes_strings = result.data()["counts"].keys()
            s_primes_counts = list(result.data()["counts"].values())
            
            s_primes_list = []
            for i, s_prime in enumerate(s_primes_strings):
                for _ in range(s_primes_counts[i]):
                    s_primes_list.append(s_prime[::-1])
            return s_primes_list
        
    def sample(self, s = None, num_samples = 100, sample_frequency = 100, T_therm = 100):
        """
        Samples the space of all causal sets, Omega, by using Quantum proposal.
        
        Returns:
        dict: A dictionary with causal matrices as keys and their counts as values.
        """
        if s is None:
            s_mat = np.ones((self.n, self.n), dtype=np.int32)
            s = "".join(str(i) for i in s_mat[np.triu_indices(self.n, 1)])
        elif type(s) == np.ndarray:
            s_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
        else:
            print("error with initial state")
        unique_causal_matrices = defaultdict(int)

        
        
        acceptance_count = 0
        self_move_count = 0
        
        
        
        
        steps = num_samples * sample_frequency + T_therm +1
        start_time = time.time()
        for step in range(steps):
            
            
            s_prime = self.proposal(s)#self.quantum_proposal(s, TC=True, BD = True, mixing_time = 0.1)
            
            s_prime_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_prime_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s_prime]
            
            #s_prime_mat = np.frombuffer(s_prime.tostring(), dtype=np.int32).reshape(self.n, self.n)
            
            if not self.is_causal_matrix(s_prime_mat):
                pass
            elif s_prime == s:
                self_move_count += 1
            else:
                s = s_prime
                s_mat = s_prime_mat
                acceptance_count +=1
            
            if step > T_therm and step % sample_frequency == 0:
                # Convert the matrix to a string representation to use as a dictionary key
                
                matrix_str = s_mat.tobytes()
                unique_causal_matrices[matrix_str] += 1
        
        end_time = time.time()
        print("Time taken: ", end_time - start_time, " (per step: ", (end_time - start_time)/steps, ", per sample ", (end_time - start_time)/num_samples, ")")
        print("acceptance rate: ", acceptance_count/steps)
        print("self move rate: ", self_move_count/steps)
        return unique_causal_matrices
    


"""




n = 4

Qsamp = QuantumSampler(n)

num_samples = 100
sample_frequency = 2* n#n**3
T_therm = sample_frequency *2


sampler = Qsamp



color = "coral"
label = None
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


uniques = sampler.sample(num_samples = num_samples, sample_frequency=sample_frequency, T_therm = T_therm)
matrix_labels = list(uniques.keys())

heights = []
num_relations = []
minimal_elements = []
ordering_fractions = []
for string in matrix_labels:
    matrix = np.frombuffer(string, dtype=np.int32).reshape(n, n)
    heights.append(sampler.height(matrix))
    num_relations.append(sampler.num_relations(matrix))
    
    minimal_elements.append(sampler.minimal_elements(matrix))
    ordering_fractions.append(sampler.ordering_fraction(matrix))
    
    

# For num_relations
counts_i = np.zeros((n*(n-1))//2+1)
for num_relation in num_relations:
    counts_i[num_relation] += 1
counts_i = np.array(counts_i)
T = np.sum(counts_i)
freq = counts_i/T
error = np.array([np.sqrt((l * (1-l))/(T-1)) for l in freq])  #√ f (1 − f )/(T − 1
ax2.errorbar(np.arange(0,len(counts_i),1), freq, yerr = error, color=color, linewidth = 0, elinewidth = 1,marker = "o",markersize = 3, label=label)

# For  heights
counts_i = np.zeros(n+1)
for height in heights:
    counts_i[height] += 1
counts_i = np.array(counts_i)
T = np.sum(counts_i)
freq = counts_i/T
error = np.array([np.sqrt((l * (1-l))/(T-1)) for l in freq])  #√ f (1 − f )/(T − 1
ax1.errorbar(np.arange(0,len(counts_i),1), freq, yerr = error, color=color, linewidth = 0, elinewidth = 1,marker = "o",markersize = 3, label=label)


    
    
    
    
    
    
    

exact_data_height = np.array([[1, 8.57e-9], [2, 0.051], [3, 0.61], [4, 0.40], 
                    [5, 0.05], [6, 0.0031], [7, 0.000090], [8, 0.0000013]])

if n ==9:
    ax1.plot(exact_data_height[:,0], exact_data_height[:,1], color = "k", linewidth = 0, marker = "o", label='Exact Data')


ax1.set_yscale('log')
ax1.set_xlabel('Height')
ax1.set_ylabel('Frequency (Normalized)')
ax1.set_title('Scatter Plot of Heights of Unique Causal Matrices')
ax1.legend()


ax2.set_yscale('log')
ax2.set_xlabel('Number of relations')
ax2.set_ylabel('Frequency (Normalized)')
ax2.set_title('Scatter Plot of Heights of Unique Causal Matrices')

plt.show()



samples = np.arange(0,len(matrix_labels)* sample_frequency,sample_frequency)
fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(10, 5))

ax2_1.plot(samples, heights)
ax2_2.plot(samples, ordering_fractions)
ax2_1.set_xlabel('Steps')
ax2_1.set_ylabel('Height')
ax2_1.set_title('Height of Unique Causal Matrices')

ax2_2.set_xlabel('Steps')
ax2_2.set_ylabel('Ordering Fraction')
ax2_2.set_title('Ordering Fraction of Unique Causal Matrices')

plt.show()"""