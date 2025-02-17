# Code to constrain the space of all causal sets, Omega to that of transitively closed matrices (i.e. causal matrices) through hamiltonian constraints. 


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_qulacs import QulacsProvider
from qiskit_qulacs.qulacs_backend import QulacsBackend



from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer import StatevectorSimulator
from tqdm import tqdm
class TransitiveClosureHamiltonian:
    
    def __init__(self, n):
        """
        Initialises the class.
        
        Parameters:
        n (int): The size of the causal set.
        """
        
        self.n = n # number of elements in the causal set
        self.q = int(n*(n-1)/2) # number of qubits (upper triangular matrix)
        self.base_circuit = self.define_TC_circuit(n)
        #     H_{TC} = \sum_{i<j<k} C_{ij}C_{jk}(C_{ik} \oplus 1)

    def plot_cost_function(self):
        """
        Plots the cost function for the transitive closure Hamiltonian.
        $H_{TC} = \sum_{i<j<k} C_{ij}C_{jk}(C_{ik} \oplus 1)$
        """
        self.get_unique_matrices()
        labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape(self.n, self.n)[np.triu_indices(self.n, 1)])) for mat in self.unique_matrices]

        
        costs = np.zeros(len(self.unique_matrices))
        for i, mat in enumerate(self.unique_matrices):
            costs[i] = self.classical_cost_function(mat)
        
        argsort = np.argsort(labels)
        labels = np.array(labels)[argsort]
        costs = costs[argsort]
        #plt.bar(labels, costs)
        #plt.xlabel('Unique Matrices')
        #plt.ylabel('Cost')
        #plt.title('Cost Function for Transitive Closure Hamiltonian')
        
        marker_array = np.zeros(len(self.unique_matrices))
        for i, mat in enumerate(self.unique_matrices):
            if mat not in self.unique_causal_matrix:
                marker_array[i] = 1
        marked_integers = []
        label_count = 0
        for i, marker in enumerate(marker_array[argsort]):
            if marker == 1:
                marked_integers.append(i)
                if label_count == 0:
                    plt.plot(i,0, marker = "x", color = "red", label = "Forbidden causal matrices", linewidth = 0)
                    label_count+=1
                else:
                    plt.plot(i,0, marker = "x", color = "red", linewidth = 0)
        self.marked_integers = marked_integers
        
        print("number of unique upper triangular binary matrices", len(self.unique_matrices))
        print("number of unique causal matrices", len(self.unique_causal_matrix))
        self.frac_allowed = len(self.unique_causal_matrix)/len(self.unique_matrices)
        print("Fraction of allowed matrices: ", self.frac_allowed)
    
    def classical_cost_function(self, string):
        """$H_{TC} = \sum_{i<j<k} C_{ij}C_{jk}(C_{ik} \oplus 1)$"""

        matrix = np.frombuffer(string, dtype = np.int32).reshape(self.n,self.n)
        cost = np.sum([matrix[i,j]*matrix[j,k]*(1-matrix[i,k]) for i in range(self.n) for j in range(i+1, self.n) for k in range(j+1, self.n)])
        return cost
    
    
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
            print("Something is wrong")



        unique_causal_matrix = set()



        for unique_matrix in unique_matrices:
            matrix = np.frombuffer(unique_matrix, dtype = dtype_).reshape(self.n,self.n)

            unique_matrices.add(matrix.tobytes())
            
        

            if self.is_causal_matrix(matrix):
                unique_causal_matrix.add(matrix.tobytes())


        self.unique_matrices = unique_matrices
        self.unique_causal_matrix = unique_causal_matrix



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

    def sample(self, num_samples = 100, sample_frequency = 100, T_therm = 100, link_move = True, relation_move = True):
        """
        Samples the space of all causal sets, Omega, by using Henson's relation move
        
        Returns:
        dict: A dictionary with causal matrices as keys and their counts as values.
        """
        
        unique_causal_matrices = defaultdict(int)

        #list of moves that can be selected
        moves = []
        if link_move:
            moves.append(0)
        if relation_move:
            moves.append(1)
        if len(moves) == 0:
            print("Must have either or both link and relation moves, not neither")
            
        acceptance = 0
        steps = num_samples * sample_frequency + T_therm +1
        for step in range(steps):
            
            #select move
            move = np.random.choice(moves)
            if move == 0:
                self.link_move()
            elif move == 1:
                self.relation_move()


            if self.last_move is not None: # If a move was made
                # Check if it is definitely a causal matrix
                check = self.is_causal_matrix(self.causal_matrix)
                if not check:
                    print("Not a causal matrix, iteration: ", _)
                    print("Last move: ", self.last_move)
                    break
                acceptance +=1
            
            if step > T_therm and step % sample_frequency == 0:
                # Convert the matrix to a string representation to use as a dictionary key
                matrix_str = self.causal_matrix.tobytes()
                unique_causal_matrices[matrix_str] += 1
        
        #print("acceptance rate: ", acceptance/steps)
        return unique_causal_matrices
    
    





    def get_basis(self, n):
        q = np.zeros((n,n), dtype = int)
        
        count = int(0)
        for i in range(n):
            for j in range(i+1, n):
                q[i,j] = count
                count += 1
        return q





    def define_TC_circuit(self, n):
        basis = self.get_basis(n)
        
        
        #Using operator like this is probably very inefficient re. matrix exponential. But will work for now.
        # Using sparse pauli op is definately better
        circuit = QuantumCircuit(self.q)
        evo_time = 0.5
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    
                    
                    
                    
                    
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
    
    def build_mixing_circuit(self):
        circuit = QuantumCircuit(self.q)
        # One body terms
        for l in range(self.q):
            #print("hiii")
            
            Z_string = np.zeros(self.q)
            X_string = np.zeros(self.q)
            X_string[l] = 1
            operator = Pauli((Z_string, X_string, 0))
            circuit = self.add_evo_to_circuit(circuit, operator, time = 0.1)

        return circuit
    
    
    def TC_quantum(self,s, TC = True):
        # function that starts with a causal set, proposes another causal set with high probability
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
    
    
        
    def compute_overlaps(self, statevector, forbidden_states):
        marked_forbidden_states = []
        
        for state in forbidden_states:
            marked_state = np.zeros(len(statevector), dtype = int)
            marked_state[state] = 1
            marked_forbidden_states.append(marked_state)
        

        
        overlaps = [np.abs(np.vdot(phi, statevector)) for phi in marked_forbidden_states]
        return overlaps

    def total_forbidden_probability(self, statevector, forbidden_states):
        overlaps = self.compute_overlaps(statevector, forbidden_states)
        return sum(overlaps)







def get_allowed_probs(n):
    H = TransitiveClosureHamiltonian(n)


    H.plot_cost_function()
    H_no_TC = TransitiveClosureHamiltonian(n)

    total_probs = np.zeros(2**H.q)
    total_probs_no_TC = np.zeros(2**H.q)
    total_probs_own = 0
    total_probs_own_no_TC = 0


    little_endian_key = []
    for i in range(2**H.q):
        little_endian_key.append(int(bin(i)[2:].zfill(H.q)[::-1],2))
    little_endian_key = np.array(little_endian_key)

    print(" ")
    print("Number of qubits: ", H.q)
    print("Total number to process: ", 2**H.q)
    for s in tqdm(product("01", repeat=H.q), desc="Processing states"):
        s = "".join(s)
        s_int = int(s, 2)
        
        if not s_int in H.marked_integers:
            counts = H.TC_quantum(s)
            
            
            
            #remove own and normalise
            temp_statevector = np.copy(H.statevector) #.probabilities())
            temp_statevector = abs(temp_statevector**2)

            temp_statevector = temp_statevector[little_endian_key]
            total_probs_own += temp_statevector[s_int]
            temp_statevector[s_int] = 0
            
            total_probs += temp_statevector
            
            
            
            
            counts_no_TC = H_no_TC.TC_quantum(s, TC=False)
            
            
            temp_statevector_no_TC = np.copy((H_no_TC.statevector))  #.probabilities())
            temp_statevector_no_TC = abs(temp_statevector_no_TC**2) # make probabilities
            temp_statevector_no_TC = temp_statevector_no_TC[little_endian_key] # Reorder because qiskit is stupid
            total_probs_own_no_TC += temp_statevector_no_TC[s_int] # Add own state to own state sum 
            temp_statevector_no_TC[s_int] = 0 # remove own state probability from the probability vector
            
            total_probs_no_TC += temp_statevector_no_TC




    #labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape(n, n)[np.triu_indices(H.n, 1)])) for mat in H.unique_matrices]
    get_bin = lambda x, n: format(x, 'b').zfill(n)

    labels = [get_bin(i,H.q) for i in range(2**H.q)]

    norm = sum([np.sum(total_probs),total_probs_own])

    total_probs_own = total_probs_own/norm
    total_probs_own_no_TC = total_probs_own_no_TC/norm

    total_probs = total_probs/norm
    total_probs_no_TC = total_probs_no_TC/norm

    plt.bar(labels,total_probs, label = "With transitive closure (hamiltonian)", alpha = 0.5, color = "coral")
    plt.bar(labels,total_probs_no_TC , label = "Without transitive closure (hamiltonian)", alpha = 0.5, color = "k")

    plt.bar(["Own State"], [total_probs_own], alpha=0.5, color="coral")
    plt.bar(["Own State"], [total_probs_own_no_TC], alpha=0.5, color="k")

    plt.xticks(rotation=45)
    plt.legend()

    plt.title("Total measurment probability of each output state")
    plt.ylabel("Total measurement probability")
    if n==4:
        plt.show()
    else:
        plt.close()
    plt.close()


    tfp = H.total_forbidden_probability(total_probs, 
                                        H.marked_integers)
    print("Total forbidden probability with TC: ", tfp)
    tfp_notc = H_no_TC.total_forbidden_probability(total_probs_no_TC, H.marked_integers)
    print("Total forbidden probability without TC: ", tfp_notc)
        


    # Calculate and print the probability of own states
    probability_own_state_with_TC = total_probs_own / (sum(total_probs)+total_probs_own)
    print("Probability of own state with TC: ", probability_own_state_with_TC)

    probability_own_state_without_TC = total_probs_own_no_TC / (sum(total_probs_no_TC)+total_probs_own_no_TC)
    print("Probability of own state without TC: ", probability_own_state_without_TC)



    total_allowed_probability_with_TC = 1 - tfp - probability_own_state_with_TC
    print("Total allowed probability with TC: ", total_allowed_probability_with_TC)

    total_allowed_probability_without_TC = 1 - tfp_notc - probability_own_state_without_TC
    print("Total allowed probability without TC: ", total_allowed_probability_without_TC)
    
    
    
    return tfp, tfp_notc, 1-H.frac_allowed, probability_own_state_with_TC, probability_own_state_without_TC, total_allowed_probability_with_TC, total_allowed_probability_without_TC


tfp_list = []
tfp_notc_list = []
frac_allowed_list = []
pos_list = []
pos_notc_list = []
tap_list = []
tap_notc_list = []

n_list = [3,4,5]
for n in n_list:
    print("Causal set size: ", n)
    tfp, tfp_notc, frac_allowed, pos, pos_notc, tap, tap_notc= get_allowed_probs(n)
    tfp_list.append(tfp)
    tfp_notc_list.append(tfp_notc)
    frac_allowed_list.append(frac_allowed)
    pos_list.append(pos)
    pos_notc_list.append(pos_notc)
    
    tap_list.append(tap)
    tap_notc_list.append(tap_notc)
    
plt.close()


tfp_list = np.array(tfp_list)
tfp_notc_list = np.array(tfp_notc_list)
pos_list = np.array(pos_list)
pos_notc_list = np.array(pos_notc_list)
frac_allowed_list = np.array(frac_allowed_list)



#plt.plot(n_list, tfp_list, label = "With TC")
#plt.plot(n_list, tfp_notc_list, label = "Without TC")

#plt.plot(n_list, pos_list, label = "Probability of own state with TC")
#plt.plot(n_list, pos_notc_list, label = "Probability of own state without TC")

#plt.plot(n_list, frac_allowed_list, label = "Fraction forbidden")

plt.plot(n_list, tap_list, label = "Total allowed probability with TC")
plt.plot(n_list, tap_notc_list, label = "Total allowed probability without TC")

plt.plot(n_list, tap_list+pos_list, label = "Total allowed probability with TC inc. own state")
plt.plot(n_list, tap_notc_list+pos_notc_list, label = "Total allowed probability without TC inc. own state")

plt.xlabel("Causal set size")
plt.ylabel("Total forbidden probability")
plt.title("Total probability of proposing forbidden states")

plt.legend()
plt.show()