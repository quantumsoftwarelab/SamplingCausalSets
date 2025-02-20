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


class QuantumSampler:
    
    def __init__(self, n):
        """
        Initialises the class.
        
        Parameters:
        n (int): The size of the causal set.
        """
        
        self.n = n # number of elements in the causal set
        self.q = int(n*(n-1)/2) # number of qubits (upper triangular matrix)
        self.base_BD_circuit = self.define_BD_circuit()
        self.base_TC_circuit = self.define_TC_circuit()
        #self.mixing_circ = self.build_mixing_circuit().decompose()


    def plot_BD_action(self):
        """
        Plots the cost function for the transitive closure Hamiltonian.
        $H_{TC} = \sum_{i<j<k} C_{ij}C_{jk}(C_{ik} \oplus 1)$
        """
        self.get_unique_matrices()
        labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape(self.n, self.n)[np.triu_indices(self.n, 1)])) for mat in self.unique_causal_matrix]
        
        
        costs = np.zeros(len(self.unique_causal_matrix))
        costs_first_order_smearing = np.zeros(len(self.unique_causal_matrix))
        costs_first_order_taylor = np.zeros(len(self.unique_causal_matrix))
        for i, mat in enumerate(self.unique_causal_matrix):
            mat_ = np.frombuffer(mat, dtype=np.int32).reshape(self.n, self.n)
            costs[i] = self.calculate_action(mat_)
            costs_first_order_smearing[i] = self.calculate_action(mat_, first_order_smearing = True)
            costs_first_order_taylor[i] = self.calculate_action(mat_, first_order_smearing = True, first_order_taylor = True)
        
        argsort = np.argsort(labels)
        labels = np.array(labels)[argsort]
        costs = costs[argsort]
        costs_first_order_smearing = costs_first_order_smearing[argsort]
        costs_first_order_taylor = costs_first_order_taylor[argsort]
        
        expectation_values = np.array(self.analyse_BD_action_Hamiltonian())[argsort]
        epsilon = 0.1
        
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
    
    
    def analyse_BD_action_Hamiltonian(self):
        expectation_values = []
        self.define_BD_circuit()
        
        op =  SparsePauliOp(self.Pauli_List, self.coeffs_list)
        labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape(self.n, self.n)[np.triu_indices(self.n, 1)])) for mat in self.unique_causal_matrix]
        for s in labels:
            qc = QuantumCircuit(self.q)
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
    

    def calc_interval_abundances(self, causal_matrix):


        adj_mat = causal_matrix
        past_mat = causal_matrix.T



        rel_abundances = np.zeros(self.n+1)
        for i, val_i in enumerate(adj_mat):
            for j, val_j in enumerate(adj_mat):
                if i == j:
                    rel_abundances[0] += 1
                elif adj_mat[i,j] ==1:
                    #number of matching 1's in row of adj matrix
                    #from Naive action algorithm in cunningham 2018
                    #only want to search in the future of j... so only top (or bottom depending on setup) half of matrix
                    k = len(np.where((adj_mat[i,:]==past_mat[j,:])&(adj_mat[i,:] ==1))[0])
                    rel_abundances[k+1] +=1
        #print("adj_mat: ", adj_mat)
        #print("rel_abundances: ", rel_abundances)
        return rel_abundances.astype('i')

    def calculate_action(self, causal_matrix, smeared = True, stdim = 2, epsilon = 0.1, first_order_smearing = False, first_order_taylor = False):
        c = self.calc_interval_abundances(causal_matrix)
        a = 0
        
        if smeared:
            if first_order_smearing:
                if first_order_taylor:
                    eps1 = epsilon / (1.0 - epsilon)
                    for i in range(0, c[0] - 1):
                        ni = float(c[i + 1])
                        if stdim == 2:
                            a += ni * (1.0-i*epsilon)
                        elif stdim == 4:
                            a += ni * (1.0-i*epsilon)
                        #print("i: ", i)
                        #print("a contribution from i: ", ni * (1.0-i*epsilon))

                    #print("a before factor: ", a)
                    if stdim == 2:
                        a= 2.0 * epsilon * (c[0] - 2.0 * epsilon * a)
                    elif stdim == 4:
                        a= (4.0 / math.sqrt(6.0)) * (math.sqrt(epsilon) * c[0] - math.pow(epsilon, 1.5) * a)
                else:
                    eps1 = epsilon / (1.0 - epsilon)
                    for i in range(0, c[0] - 1):
                        ni = float(c[i + 1])
                        if stdim == 2:
                            a += ni * math.pow(1.0 - epsilon, i) 
                        elif stdim == 4:
                            a += ni * math.pow(1.0 - epsilon, i) 

                    if stdim == 2:
                        a= 2.0 * epsilon * (c[0] - 2.0 * epsilon * a)
                    elif stdim == 4:
                        a= (4.0 / math.sqrt(6.0)) * (math.sqrt(epsilon) * c[0] - math.pow(epsilon, 1.5) * a)
                
            else:               
                eps1 = epsilon / (1.0 - epsilon)
                for i in range(0, c[0] - 1):
                    ni = float(c[i + 1])
                    if stdim == 2:
                        a += ni * math.pow(1.0 - epsilon, i) * (1.0 - 2.0 * eps1 * i + 0.5 * eps1 * eps1 * i * (i - 1.0))
                    elif stdim == 4:
                        a += ni * math.pow(1.0 - epsilon, i) * (1.0 - 9.0 * eps1 * i + 8.0 * eps1 * eps1 * i * (i - 1.0) - (4.0 / 3.0) * eps1 * eps1 * eps1 * i * (i - 1.0) * (i - 2.0))

                if stdim == 2:
                    a= 2.0 * epsilon * (c[0] - 2.0 * epsilon * a)
                elif stdim == 4:
                    a= (4.0 / math.sqrt(6.0)) * (math.sqrt(epsilon) * c[0] - math.pow(epsilon, 1.5) * a)
        else:
            if stdim == 2:
                a = 2.0 * (c[0] - 2.0 * (c[1] - 2.0 * c[2] + c[3]))
            elif stdim == 4:
                a = (4.0 / math.sqrt(6.0)) * (c[0] - c[1] + 9.0 * c[2] - 16.0 * c[3] + 8.0 * c[4])
                
        #print("a: ", a)
        return a
    
    

    def transitive_closure(self,a):
        # copied from https://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
        n = len(a)
        m = np.copy(a)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    m[i][j] = m[i][j] or (m[i][k] and m[k][j])
        return m
    
    def is_causal_matrix(self, matrix):
        """
        Checks if a given upper triangular binary matrix corresponds to the causal matrix of a causal set.
        
        Parameters:
        matrix (np.ndarray): The upper triangular binary matrix to check.
        
        Returns:
        bool: True if the matrix corresponds to a causal matrix, False otherwise.
        """
        
        m = self.transitive_closure(matrix)
        if np.all(m == matrix):
            return True
        else:
            return False
    
    def get_unique_matrices(self):
        
        num_unique = 2**((self.n**2-self.n)//2)
        
        unique_matrices = set()
        for bits in product([0, 1], repeat=(self.n * (self.n - 1)) // 2):
            matrix = np.zeros((self.n, self.n), dtype = np.int32)
            upper_tri_indices = np.triu_indices(self.n, 1)
            matrix[upper_tri_indices] = bits
            unique_matrices.add(matrix.tobytes())
            dtype_ = matrix.dtype

            
        #print(f"Number of unique matrices: {len(unique_matrices)}")
        #if len(unique_matrices) != num_unique:
        #    #print("Something is wrong")



        unique_causal_matrix = set()



        for unique_matrix in unique_matrices:
            matrix = np.frombuffer(unique_matrix, dtype = dtype_).reshape(self.n,self.n)

            unique_matrices.add(matrix.tobytes())
            
        

            if self.is_causal_matrix(matrix):
                unique_causal_matrix.add(matrix.tobytes())


        self.unique_matrices = unique_matrices
        self.unique_causal_matrix = unique_causal_matrix






    def get_basis(self, n):
        q = np.zeros((n,n), dtype = int)
        
        count = int(0)
        for i in range(n):
            for j in range(i+1, n):
                q[i,j] = count
                count += 1
        return q




    def define_BD_circuit(self,  epsilon = 0.1, evo_time = 10):
        n = self.n
        basis = self.get_basis(n)
        
        
        #Using operator like this is probably very inefficient re. matrix exponential. But will work for now.
        # Using sparse pauli op is definately better
        
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
        
        for i in range(n):
            for k in range(i+1, n):
                #print("i,k: ", i,k)
                
                
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
                    #print("i,k,j: ", i,k,j)
                    
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


    def define_TC_circuit(self):
        basis = self.get_basis(self.n)
        
        
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

    def add_evo_to_circuit(self, circuit, op, time = 0.2):
        evo = PauliEvolutionGate(op, time=time)
        circuit.append(evo, range(self.q))
        return circuit
    
    
    def build_mixing_circuit(self, mixing_time = 0.1):
        circuit = QuantumCircuit(self.q)
        # One body terms
        for l in range(self.q):
            #print("hiii")
            
            Z_string = np.zeros(self.q)
            X_string = np.zeros(self.q)
            X_string[l] = 1
            operator = Pauli((Z_string, X_string, 0))
            circuit = self.add_evo_to_circuit(circuit, operator, time = mixing_time)

        return circuit.decompose()
    
    def propose_new_configuration(self, s, TC=True, BD = True, mixing_time = 0.1, multiple = 1):
        # function that proposes a new bitstring, with optional Hamiltonian contributions to:
        # Restrict the proposals to causal matrices
        # Restrict the proposals to matrices of similar bd action
        mixing_circ = self.build_mixing_circuit(mixing_time = mixing_time)
        time = 5
        
        
        #set initial state
        qc = QuantumCircuit(self.q)
        for i, x in enumerate(s):
            if x == "1":
                qc.x(i)
                
        #If doing transitive closure
        if TC:
            qc.compose(self.base_TC_circuit, inplace = True)
        
        # If doing BD action
        if BD:
            qc.compose(self.base_BD_circuit, inplace = True)
        
        
        
        for t in range(time):
            #Always mix
            qc.compose(mixing_circ, inplace = True)
            if TC:
                qc.compose(self.base_TC_circuit, inplace = True)
            if BD:
                qc.compose(self.base_BD_circuit, inplace = True)
        
        #qc.save_statevector(label = 'test', pershot = False)
        #qc.measure_all()
        #print(qc.draw())
        
        


        # Use Qiskit-Qulacs to run the circuit
        #backend = QulacsProvider().get_backend("qulacs_simulator")
        #backend = StatevectorSimulator()#BasicSimulator()
        backend = QulacsBackend()

        

        result = backend.run(qc, shots = multiple).result()
        #self.statevector = result.get_statevector()

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
        
        
    def analyse_output_bitstrings(self, TC=True, BD = True, mixing_time = 0.1, repeats = 100, energy_ordering = True, title = "Transition Matrix"):
        self.get_unique_matrices()
        
        mats = [np.frombuffer(mat, dtype=np.int32).reshape(self.n, self.n) for mat in self.unique_causal_matrix]
        labels = ["".join(str(i) for i in mat[np.triu_indices(self.n, 1)]) for mat in mats]
        costs = np.zeros((2**self.q,2**self.q))
        BD_action = np.zeros(2**self.q)
        first_loop = True
        #for r in tqdm(range(repeats), desc="Repeats"):
        for s_pos, s in enumerate(labels):
            s_prime_list = self.propose_new_configuration(s, TC=TC, BD = BD, mixing_time = mixing_time, multiple = repeats)
            
            s_int = int(s, 2)
            
            for s_prime in s_prime_list:
                s_prime_int = int(s_prime, 2)
                costs[s_int, s_prime_int] += 1

            if first_loop:
                BD_action[s_int] = self.calculate_action(mats[s_pos])
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
        
        _costs = np.flipud(costs)
        
        cmap = colors.LinearSegmentedColormap.from_list('red_white', ['white', 'red'], N=256)
        
        
        
        if energy_ordering:
            non_zero_bd_index = np.where(sorted_BD != 0)[0][0]
            
            #plt.xticks(ticks=np.arange(len(sorted_BD)/5), labels=np.round(sorted_BD, 2), rotation=90)
            #plt.yticks(ticks=np.arange(len(sorted_BD)/5), labels=np.round(sorted_BD, 2))
            plt.xticks([])
            plt.yticks([])
            plt.ylim(non_zero_bd_index, 2**self.q)
            
            plt.imshow(_costs, extent=[0, 2**self.q, 0, 2**self.q], cmap=cmap, interpolation='nearest', norm=colors.LogNorm())
            
        else:
            plt.imshow(_costs, extent=[0, 2**self.q, 0, 2**self.q], cmap=cmap, interpolation='nearest', norm=colors.LogNorm())
        
        
        plt.colorbar(label='Transition Counts', norm=colors.LogNorm())
        #plt.colorbar(label='Transition Counts')
        plt.xlabel('Proposed Configuration (s\')')
        plt.ylabel('Initial Configuration (s)')
        plt.title(title)
        
        
        s_int_list = [int(s, 2) for s in labels]
        all_ints = np.arange(0, 2**self.q)
        
        #print("BD_sorted_args before: ", BD_sorted_args)
        #print("all_ints before: ", all_ints)
        if energy_ordering:
            all_ints = all_ints[BD_sorted_args]
            

            plt.plot([non_zero_bd_index, 2**self.q], [non_zero_bd_index,non_zero_bd_index] ,color='blue', linestyle='--', label='Non-zero BD')
            plt.plot([non_zero_bd_index,non_zero_bd_index], [non_zero_bd_index, 2**self.q] ,color='blue', linestyle='--', label='Non-zero BD')


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
        print(title)
        BD_transitions = np.zeros((2**self.q,2**self.q))
        for i in range(2**self.q):
            for j in range(2**self.q):
                BD_transitions[i,j] = np.abs(sorted_BD[i] - sorted_BD[j])
        
        if energy_ordering:
            total_bd_transition_cost = np.sum((BD_transitions[non_zero_bd_index:,non_zero_bd_index:] * costs[non_zero_bd_index:,non_zero_bd_index:]))
        else:
            total_bd_transition_cost = np.sum(BD_transitions * costs)
            print("THIS IS WRONG AS INCLUDES TRANSITIONS TO NON-CAUSAL MATRICES")
            
        print(" ")
        print("Total BD transition cost: ", total_bd_transition_cost)
        print(" ")
        print("Total number of self transitions: ", np.sum(np.diag(costs)))
        print("Total forbidden transitions: ", forbidden_count)
        print("Out of a total of: ", np.sum(costs))
        
        
        plt.show()
        
        weighted_BD_transitions = np.repeat(BD_transitions[non_zero_bd_index:,non_zero_bd_index:].flatten(), costs[non_zero_bd_index:,non_zero_bd_index:].flatten().astype(int)).flatten()
        #print("sum of all 0 in weighted BD transitions: ", np.sum(weighted_BD_transitions == 0))
        #print("BD_transitions[non_zero_bd_index:,non_zero_bd_index:]", BD_transitions[non_zero_bd_index:,non_zero_bd_index:])
        # Remove diagonal elements (self-transitions)
        costs_no_self = np.copy(costs)
        np.fill_diagonal(costs_no_self, 0)
        weighted_BD_transitions_ignoring_self_transitions = np.repeat(BD_transitions[non_zero_bd_index:,non_zero_bd_index:].flatten(), costs_no_self[non_zero_bd_index:,non_zero_bd_index:].flatten().astype(int)).flatten()
        #BD_transitions = np.abs(BD_action[:, None] - BD_action[None, :]).flatten()
        plt.hist(weighted_BD_transitions, bins=50, density=True, cumulative=True, histtype='step', label='Cumulative Probability')
        plt.hist(weighted_BD_transitions_ignoring_self_transitions, bins=50, density=True, cumulative=True, histtype='step', label='Cumulative Probability (no self transitions)')

        plt.xlabel('BD Transitions')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Probability of BD Transitions')
        plt.legend()
        plt.show()
        
        return 

    def sample(self, s = None, num_samples = 100, sample_frequency = 100, T_therm = 100):
        """
        Samples the space of all causal sets, Omega, by using Henson's relation move
        
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
            
            
            s_prime = self.propose_new_configuration(s, TC=True, BD = True, mixing_time = 0.1)
            
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
    
    def num_relations(self, matrix):
        """
        Calculates the number of relations in a given causal matrix.
        
        Parameters:
        matrix (np.ndarray): The causal matrix to calculate the number of relations of.
        
        Returns:
        int: The number of relations in the causal matrix.
        """
        return np.sum(matrix)
    
    
    def height(self, matrix):
        """
        Calculates the height of a given causal matrix. Ie. the length of the longest chain of relations.
        
        Parameters:
        matrix (np.ndarray): The causal matrix to calculate the height of.
        
        Returns:
        int: The height of the causal matrix.
        """
        n = len(matrix)
        
        longest_path_ending_each = np.zeros(n)
        # for every pair i < j
        for i in range(1, n):
            current_longest_parent = -1
            for j in range(0, i): # search for longest path to a past connection
                
                if matrix[j,i] == 1:
                    length_current_parent = longest_path_ending_each[j]
                    current_longest_parent = max(current_longest_parent, length_current_parent)
            longest_path_ending_each[i] = current_longest_parent + 1
        
        longest_chain = int(np.max(longest_path_ending_each)) +1
        #+1 as height is the numbe rof nodes, not relations
        return longest_chain

    def ordering_fraction(self, matrix):
        # The ordering fraction r is the fraction of pairs of elements
        # which are related
        
        #R/(N choose 2) # R is the number of relations
        R = np.sum(matrix)
        n = len(matrix)
        
        return R/((n*(n-1))/2)
    
    def minimal_elements(self, matrix):
        # Counts the number of minimal elements in the causal set 
        # (elements with no incoming relations)
        
        n = len(matrix)
        minimal_elements = 0
        for i in range(n):
            if np.sum(matrix[:,i]) == 0:
                minimal_elements += 1
        return minimal_elements


Qsamp = QuantumSampler(5)
#BD_ham.propose_new_configuration("000000", TC=True, BD = True, mixing_time = 0.1)
Qsamp.analyse_output_bitstrings(TC=False, BD = False, mixing_time = 0.1, repeats = 10000, energy_ordering=True, title = "Mixing only")
#Qsamp.analyse_output_bitstrings(TC=False, BD = True, mixing_time = 0.1, repeats = 100, energy_ordering=True, title = "Mixing and BD")
#Qsamp.analyse_output_bitstrings(TC=True, BD = False, mixing_time = 0.1, repeats = 100, energy_ordering=True, title = "Mixing and TC")
Qsamp.analyse_output_bitstrings(TC=True, BD = True, mixing_time = 0.1, repeats = 10000, energy_ordering=True, title = "Mixing, TC and BD")


Qsamp.plot_BD_action()


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