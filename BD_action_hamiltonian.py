# Code to constrain the space of all causal sets, Omega to that of transitively closed matrices (i.e. causal matrices) through hamiltonian constraints. 


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_qulacs import QulacsProvider
from qiskit_qulacs.qulacs_backend import QulacsBackend
import math


from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector, PauliList
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer import StatevectorSimulator
from tqdm import tqdm
from qiskit_qulacs.qulacs_estimator import QulacsEstimator


class BDActionHamiltonian:
    
    def __init__(self, n):
        """
        Initialises the class.
        
        Parameters:
        n (int): The size of the causal set.
        """
        
        self.n = n # number of elements in the causal set
        self.q = int(n*(n-1)/2) # number of qubits (upper triangular matrix)
        self.base_circuit = self.define_BD_circuit()
        self.base_circuit = self.define_TC_circuit()


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
        
        plt.plot(labels, np.array(expectation_values), label = "Expectation values (+constant but not all yet)")
        plt.plot(labels, costs, label = "BD action", alpha = 0.5)
        plt.plot(labels, costs_first_order_smearing, label = "BD action first order", alpha = 0.5)
        plt.plot(labels, costs_first_order_taylor, label = "BD action first order taylor", alpha = 0.5)
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
        if len(unique_matrices) != num_unique:
            #print("Something is wrong")



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




    def define_BD_circuit(self,  epsilon = 0.1, evo_time = 0.5):
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
    
    
    def propose_new_configuration(self, s, TC=True, BD = True):
        # function that proposes a new bitstring, with optional Hamiltonian contributions to:
        # Restrict the proposals to causal matrices
        # Restrict the proposals to matrices of similar bd action
        
        time = 5
        mixing_circ = self.build_mixing_circuit().decompose()
        
        #set initial state
        qc = QuantumCircuit(self.q)
        for i, x in enumerate(s):
            if x == "1":
                qc.x(i)
        if TC:
            qc.compose(self.base_circuit, inplace = True)
        
        
        for t in range(time):
            qc.compose(mixing_circ, inplace = True)
            if TC:
                qc.compose(self.base_circuit, inplace = True)
        
        #qc.save_statevector(label = 'test', pershot = False)
        #qc.measure_all()
        #print(qc.draw())
        
        


        # Use Qiskit-Qulacs to run the circuit
        #backend = QulacsProvider().get_backend("qulacs_simulator")
        #backend = StatevectorSimulator()#BasicSimulator()
        backend = QulacsBackend()

        

        result = backend.run(qc).result()
        self.statevector = result.get_statevector()


        #result = backend.run(qc, shots=20000).result()   
        
        
        
        #counts =  result.get_counts()
        #print("counts:",counts)
        #plot_histogram(counts)
        #plt.show()
        #self.statevector = result.data(0)['test']#np.asarray(result.get_statevector())
        #print("statevector: ", self.statevector)
        return None#counts
        
    

    




BD_ham = BDActionHamiltonian(5)
#BD_ham.plot_BD_action()