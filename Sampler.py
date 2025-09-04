
import time
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_qulacs.qulacs_backend import QulacsBackend
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
import math
from typing import List
import numpy.typing as npt
from tqdm import tqdm
from qiskit_qulacs.qulacs_estimator import QulacsEstimator
from helpers import transitive_closure, transitive_reduction, is_causal_matrix, calculate_action, get_unique_matrices, transitive_closure, get_upper_triangular_basis, num_relations, height, ordering_fraction, minimal_elements, is_critical_pair, is_suitable_pair, is_linked

class Sampler:
    """
    Sampler class for generating samples of causal sets using quantum or classical methods.
    This class provides functionality to sample causal sets via quantum circuits or classical moves,
    supporting both uniform and BD-action-based sampling. 
    
    Attributes
    ----------
    n : int
        Number of elements in the causal set.
    dt : float
        Time step for quantum evolution.
    verbose : bool
        Verbosity flag.
    epsilon : float
        BD action approximation parameter.
    dimension : int
        Dimension of the causal set.
    q : int
        Number of qubits (upper triangular matrix size).
    method : str
        Sampling method ("quantum" or "classical").
    proposal : callable
        Proposal function for generating new configurations.
    gamma_TC, gamma_BD, gamma_mixing : float
        Hamiltonian coefficients for quantum sampling.
    alpha_TC, alpha_BD : float
        Normalization parameters for Hamiltonian terms.
    moves : list
        List of allowed classical moves.
    Methods
    -------
    allocate_gammas_from_ranges()
        Samples gamma values from specified ranges.
    allocate_t_from_range()
        Samples t (mixing steps) from specified range.
    define_BD_circuit(gamma_BD, epsilon=0.1, trivial_terms=False, return_circuit=True)
        Constructs quantum circuit for BD action evolution.
    analyse_BD_action_Hamiltonian()
        Analyzes BD Hamiltonian expectation values for unique causal sets.
    get_alpha_BD()
        Computes normalization parameter for BD action.
    get_alpha_TC()
        Computes normalization parameter for TC action.
    define_TC_circuit(gamma_TC, trivial_terms=False, return_circuit=True)
        Constructs quantum circuit for transitive closure evolution.
    define_mixing_circuit(gamma_mix)
        Constructs quantum circuit for mixing term.
    add_evo_to_circuit(circuit, op, time=0.2)
        Adds time evolution gate to a quantum circuit.
    quantum_proposal(s, multiple=1, output_statevector=False)
        Proposes new configuration(s) using quantum circuit evolution.
    classical_proposal(s, multiple=1)
        Proposes new configuration(s) using classical moves.
    link_move(s)
        Performs a link move on the causal set.
    relation_move(s)
        Performs a relation move on the causal set.
    sample_uniform(s=None, num_samples=100, sample_frequency=100, T_therm=100, observables=[...])
        Samples causal sets uniformly using the chosen proposal method.
    sample_BD(T, s=None, num_samples=100, sample_frequency=100, T_therm=100, observables=[...])
        Samples causal sets using BD action and Metropolis-Hastings acceptance.
    BD_acceptance_probability(s_old, s_new, T)
        Computes acceptance probability for BD action.
    accept_reject(s_old, s_new, T)
        Performs Metropolis-Hastings acceptance-rejection step.
    Notes
    -----
    - Quantum sampling requires Qiskit-Qulacs backend and appropriate Hamiltonian configuration.
    - Classical sampling supports link and relation moves as described in causal set literature.
    - Observables supported include ordering fraction, height, number of relations, minimal elements, and BD action.
    - Only dimensions 2 and 4 are supported for BD action."""
    

    def __init__(self, n:int, method: str = "quantum", qargs: dict = {}, cargs :dict = {}, verbose: bool = True, epsilon: float = 0.1, dimension: int = 2):
        """
        Initialises the Sampler class.
        
        Parameters:
        n (int): The cardinality of the causal set.
        method (str): The method to use for sampling. Options are "quantum" and "classical".
        qargs (dict): A dictionary of arguments to pass to the quantum, including the follwoing:
            Note that you provide EITHER gammas or gamma ratios. Either work, but both are provided for ease of use.
            
            gammas (list[float]): [gamma_TC, gamma_BD, gamma_mix]: Gamma parameters that govern the magnitude of the hamiltonian terms.
                gamma_TC: If nonzero, add Transitive Closure to Hamiltonian.
                gamma_BD: If nonzero, add BD action approximation to Hamiltonian.
                gamma_mix = (1- gamma_TC - gamma_BD) If nonzero, add the mixing term to the Hamiltonian. Can be inferred from 'gamma_TC' and `gamma_BD' if not provided
                If gamma mix is not provided, it will be set to 1 - gamma_TC - gamma_BD.
                Cannot be sampled from a range of values for each.
            gamma_ratios (list[float or tuple]): [gamma_TC_ratio, gamma_BD_ratio]
                gamma_TC_ratio: If nonzero, add Transitive Closure to Hamiltonian.
                gamma_BD_ratio: If nonzero, add BD action approximation to Hamiltonian.
                If ratios are provided, gammas are set to: [gamma_TC, gamma_BD, gamma_mix] = [TC_gamma_ratio, float((1-TC_gamma_ratio)*gamma_BD_ratio), float((1-TC_gamma_ratio)*(1-gamma_BD_ratio))]
                This makes it easier to sample from different possible gammas, so you may provide a tuple of floats representing a range from which to samplef or each ratio.
        cargs (dict): A dictionary of arguments to pass to the classical sampler, including the following:
            link_move (bool): If True, allow link moves. Default is False.
            relation_move (bool): If True, allow relation moves. Default is False.
        verbose (bool): If True, print verbose output. Default is True.
        epsilon (float): A small parameter used in the BD action approximation. Default is 0.1.
        dimension (int): The dimension of the causal set. Default is 2.
        """
        self.n = n # number of elements in the causal set
        self.dt = 0.8 # time step for the evolution same as Layden
        self.verbose = verbose
        self.epsilon = epsilon # small parameter used in the BD action approximation
        self.dimension = dimension
        self.q = int(n*(n-1)/2) # number of qubits (upper triangular matrix)
        self.method = method
        
        self.Q = None

        
        
        if self.dimension != 2 and self.dimension != 4:
            raise ValueError("Dimension not supported, BD action is currently only defined for 2d and 4d")
        
        if method == "quantum":
            #self.base_BD_circuit = self.define_BD_circuit()
            #self.base_TC_circuit = self.define_TC_circuit()
            #self.mixing_circ = self.build_mixing_circuit().decompose()
            self.proposal = self.quantum_proposal
            
            
            
            
            if not qargs:
                if self.verbose :
                    print("qargs dictionary is empty. Proceeding with default values.")
            
            if "gammas" in qargs and "gamma_ratios" in qargs:
                raise ValueError("You cannot provide both gammas and gamma_ratios. Please provide one or the other.")
            
            if "gammas" in qargs:
                self.gamma_ranges = False
                if type(qargs["gammas"]) != list:
                        if type(qargs["gammas"]) != np.ndarray:
                            raise ValueError("Gammas must be a list or array of floats or integers.")
                for val in qargs["gammas"]:
                    if type(val) != float:
                        if type(val) != int:
                                raise ValueError("Each gamma must be a float or integer.")
                if len(qargs["gammas"]) == 3:
                        
                    if not np.isclose(float(np.sum(qargs["gammas"])),1):
                        raise ValueError("Gammas must sum to 1 if providing gamma_mix")
                    else:
                        self.gamma_TC = qargs["gammas"][0]
                        self.gamma_BD = qargs["gammas"][1]
                        self.gamma_mixing = qargs["gammas"][2]
                        
                elif len(qargs["gammas"]) == 2:
                    if np.sum(qargs["gammas"]) > 1:
                        raise ValueError("Gammas must sum to less than or equal to 1 if not providing gamma_mix")
                    else:
                        self.gamma_TC = qargs["gammas"][0]
                        self.gamma_BD = qargs["gammas"][1]
                        self.gamma_mixing = 1 - self.gamma_TC - self.gamma_BD
                
                self.alpha_TC = self.get_alpha_TC()
                self.alpha_BD = self.get_alpha_BD()
                
            elif "gamma_ratios" in qargs:
                if type(qargs["gamma_ratios"]) != list:
                        if type(qargs["gamma_ratios"]) != np.ndarray:
                            raise ValueError("Gamma ratios must be a list or array of floats, integers or tuples.")
                for val in qargs["gamma_ratios"]:
                    if type(val) != float:
                            if type(val) != tuple:
                                if type(val) != np.floating:
                                    raise ValueError("Each gamma ratio must be a float or tuple.")
                            
                            
                if len(qargs["gamma_ratios"]) != 2:
                    raise ValueError("If providing gamma ratios, you must provide a list of two floats or tuples.")
                        
                elif len(qargs["gamma_ratios"]) == 2:
                    self.TC_gamma_ratio = qargs["gamma_ratios"][0]
                    self.BD_gamma_ratio = qargs["gamma_ratios"][1]
                    if type(self.TC_gamma_ratio) == tuple and type(self.BD_gamma_ratio) == tuple:
                        self.gamma_ranges = True
                        self.gamma_TC = None
                        self.gamma_BD = None
                        self.gamma_mixing = None
                        
                        
                        if self.verbose:
                            tc_, bd_, mix_ = self.allocate_gammas_from_ranges()
                            print("Allocating gamma from ranges, test gammas, TC: ", tc_, "BD: ", bd_, "Mixing: ", mix_)
                    elif bool(type(self.TC_gamma_ratio) == tuple) != bool(type(self.BD_gamma_ratio) == tuple):
                        raise ValueError("If providing gamma ranges, both TC and BD gamma ratios must be tuples or floats, not a mixture.")
                    else:
                        self.gamma_ranges = False
                        self.gamma_TC = self.TC_gamma_ratio
                        self.gamma_BD = float((1-self.TC_gamma_ratio)*self.BD_gamma_ratio)
                        self.gamma_mixing = float((1-self.TC_gamma_ratio)*(1-self.BD_gamma_ratio))
                        if self.verbose:
                            print("Gamma ratios provided, setting gammas to: TC: ", self.gamma_TC, "BD: ", self.gamma_BD, "Mixing: ", self.gamma_mixing)
                


                self.alpha_TC = self.get_alpha_TC()
                self.alpha_BD = self.get_alpha_BD()
                
            else:
                raise ValueError("Gammas or gamma ratios must be specified.")
            
            
                
            if "t" in qargs:
                t = qargs["t"]
                if type(t) == int:
                    self.is_t_a_range = False
                    self.t = qargs["t"]
                elif type(t) == tuple:
                    self.is_t_a_range = True
                    self.t_range = qargs["t"]
                    self.t = None
                else:
                    raise ValueError("t must be an integer or tuple.")
            else:
                print("t not provided in qargs, defaulting to 5.")
                self.is_t_a_range = False
                self.t = 5
            if self.verbose:
                print("------------------------------------------------------------")
                print("Starting quantum algorithm with the following parameters:")
                print("gamma_TC: ", self.gamma_TC)
                print("gamma_BD: ", self.gamma_BD)
                print("gamma_mixing: ", self.gamma_mixing)
                print("t: ", self.t)
            
        elif method == "classical":
            if self.verbose:
                print("cargs recieved : ", cargs)
            #raise ValueError("Classical method not yet implemented.")
            self.proposal = self.classical_proposal
            #list of moves that can be selected
            self.moves = []
            
            for key in cargs:
                if key not in ["link_move", "relation_move"]:
                    raise ValueError(f"Invalid classical move key: '{key}'. Only 'link_move' and 'relation_move' are allowed.")
            
            if len(cargs) == 0:
                if self.verbose:
                    print("cargs dictionary is empty. Proceeding with default (both link and relation move).")
                self.moves = [0,1]
            
            if "link_move" in cargs:
                if type(cargs["link_move"]) != bool:
                    if self.verbose:
                        print("Link_move must be a boolean value. Defaulting to False.")
                else:
                    if cargs["link_move"]:
                        self.moves.append(0)
            if "relation_move" in cargs:
                if type(cargs["relation_move"]) != bool:
                    if self.verbose:
                        print("Relation_move must be a boolean value. Defaulting to False.")
                else:
                    if cargs["relation_move"]:
                        self.moves.append(1)
            
        
            if len(self.moves) == 0:
                print("Must have either or both link and relation moves, not neither")
                
            if self.verbose:
                print("------------------------------------------------------------")
                print("Starting classical algorithm with the following parameters:")
                print("Link move: ", cargs["link_move"])
                print("Relation move: ", cargs["relation_move"])
            
        else:
            raise ValueError("Invalid method. You gave: '" + str(method) + "'. Please choose 'quantum' or 'classical'.")

    def allocate_gammas_from_ranges(self):
        """
        Allocate gamma values from ranges (provided in qargs).

        Raises:
            ValueError: If gamma ranges are not provided.

        Returns:
            Tuple[float, float, float]: The allocated gamma values for TC, BD, and mixing.
        """

        # Check if gamma ranges are provided
        if not self.gamma_ranges:
            raise ValueError("Gamma ranges not provided. Cannot sample from ranges.")

        # Sample gamma values from the provided ranges
        TC_gamma_ratio = np.random.uniform(self.TC_gamma_ratio[0], self.TC_gamma_ratio[1])
        BD_gamma_ratio = np.random.uniform(self.BD_gamma_ratio[0], self.BD_gamma_ratio[1])
        
        # Turn ratios into explicit values for each gamma
        gamma_TC = TC_gamma_ratio
        gamma_BD = float((1-TC_gamma_ratio)*BD_gamma_ratio)
        gamma_mixing = float((1-TC_gamma_ratio)*(1-BD_gamma_ratio))
        
        #print("self.TC_gamma_ratio: ", self.TC_gamma_ratio)
        #print("TC_gamma_ratio", TC_gamma_ratio)
        return gamma_TC, gamma_BD, gamma_mixing

    def allocate_t_from_range(self):

        """Allocate t values from ranges (provided in qargs).

        Raises:
            ValueError: If t ranges are not provided.

        Returns:
            int: The allocated t value.
        """
        if not self.is_t_a_range:
            raise ValueError("t range not provided. Cannot sample from ranges. BUG.")
        
        t = np.random.randint(self.t_range[0], self.t_range[1])
        return t


    def analyse_BD_action_Hamiltonian(self):
        
        """
        Analyze the action of the BD Hamiltonian, according to the Hamiltonian defined in define_BD_circuit.
        Cycles through each unique causal set, and computes the expectation values of the Hamiltonian, which can later be compared with exact calculated values.

            
        """
        unique_matrices, unique_causal_matrix  = get_unique_matrices(self.n) 
        expectation_values = []
        self.define_BD_circuit(gamma_BD= 1, trivial_terms=True)
        
        op =  SparsePauliOp(self.BD_Pauli_List, self.BD_coeffs_list)
        labels = ["".join(str(i) for i in list(np.frombuffer(mat, dtype=np.int32).reshape(self.n, self.n)[np.triu_indices(self.n, 1)])) for mat in unique_causal_matrix]
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
            expectation_values.append(expectation_value)
        return expectation_values
    
    
    def get_alpha_BD(self):
        """
        Calculate the alpha parameter for the BD action Hamiltonian. 
        This is done according to the expression in the appendix of the paper.
        """

        norm_analytic = np.sqrt(4*self.epsilon**4*self.q)

        alpha_BD = np.sqrt(self.q)/norm_analytic
        return alpha_BD
    
    def get_alpha_TC(self):
        
        """
        Calculate the alpha parameter for the TC action.
        This is done according to the expression in the appendix of the paper
        """
        

        lowest_energy_level = -math.comb(self.n, 3)/8
        magnitude_of_lowest_energy_level = abs(lowest_energy_level)
        #print("Lowest energy level: ", lowest_energy_level)
        #print("normalised lowest energy level: ", E_0/magnitude_of_lowest_energy_level)
        alpha_TC = np.sqrt(self.q)/magnitude_of_lowest_energy_level
        return alpha_TC


    def define_BD_circuit(self, gamma_BD: float,  epsilon: float = 0.1, trivial_terms:bool = False, return_circuit: bool = True)-> QuantumCircuit| None:
        """
        # If return_circuit is False, this function is being used to initialise alpha_BD
        Define a quantum circuit for time evolution of the approximated BD action.
            
        This function constructs a quantum circuit based on the BD model by defining various 
        Pauli operators and their corresponding time. The circuit is built by iterating over 
        pairs and triples of indices, applying the appropriate operators, and adding them 
        to the circuit. The Pauli operators and their coefficients are stored in the instance 
        variables `Pauli_List` and `coeffs_list` respectively.
        
        Parameters:
        gamma_BD (float): The gamma parameter for the BD action.
        epsilon (float): A small parameter used in the BD action approximation.
        trivial_terms (bool): If True, include trivial terms in the circuit. Default is False.
        return_circuit (bool): If True, return the constructed quantum circuit. If False, only calculate alpha_BD.
        
        
        Returns:
        QuantumCircuit: The constructed quantum circuit after applying the evolution operators.

        """
        
        if gamma_BD == 0 and return_circuit is True:
            raise ValueError("gamma_BD must be nonzero.")
        
        if return_circuit:
            evo_time = gamma_BD*self.alpha_BD
        else:
            evo_time = 1
        
        # define the map from the relation between ith and jth element
        # to the index of the corresponding qubit in the quantum circuit
        basis = get_upper_triangular_basis(self.n)
        
        
        # Using operator like this is probably very inefficient re. matrix exponential. But will work for now.
        # Using sparse pauli op is definitely better
        
        pauli_list =[]
        coeffs = []
        circuit = QuantumCircuit(self.q)
        if self.dimension != 2 and self.dimension != 4:
            raise ValueError("Dimension not supported by define BD circuit")
        
        if trivial_terms:
            # cardinality term 
            # Before outer loop
            if self.dimension == 2:
                evo_time_prime_N = 2*epsilon*evo_time*self.n
            elif self.dimension == 4:
                evo_time_prime_N = (4/np.sqrt(6))*np.sqrt(epsilon)*evo_time*self.n
            Z_string = np.zeros(self.q)
            operator = Pauli((Z_string, np.zeros(self.q), 0))
            pauli_list.append(operator)
            coeffs.append(evo_time_prime_N/evo_time)
            circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_N)
        
        for i in range(self.n):
            for k in range(i+1, self.n):                
                
                if trivial_terms:
                    if self.dimension == 2:
                        evo_time_prime_const = 2*epsilon**2*(1)*evo_time
                    elif self.dimension == 4:
                        evo_time_prime_const = (2/np.sqrt(6))*(epsilon**(3/2))*evo_time
                    
                    Z_string = np.zeros(self.q)
                    operator = Pauli((Z_string, np.zeros(self.q), 2))
                    pauli_list.append(operator)
                    coeffs.append(evo_time_prime_const/evo_time)
                    circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_const)
                
                # single ik term
                # 2epsilon**2 Z_ik (1)
                if self.dimension == 2:
                    evo_time_prime_one_body = 2*epsilon**2*(1)*evo_time
                elif self.dimension == 4:
                    evo_time_prime_one_body = (2/np.sqrt(6))*(epsilon**(3/2))*evo_time
                Z_string = np.zeros(self.q)
                Z_string[basis[i,k]] = 1
                operator = Pauli((Z_string, np.zeros(self.q), 0))
                pauli_list.append(operator)
                coeffs.append(evo_time_prime_one_body/evo_time)
                circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_one_body)
                
                
                # Inner loop (over all j for which i<j<k)
                if self.dimension == 2:
                    evo_time_prime_inner = (3/2)*(epsilon**3)*evo_time
                elif self.dimension == 4:
                    evo_time_prime_inner = (5/np.sqrt(6))*(epsilon**(5/2))*evo_time
                
                
                for j in range(i+1, k):
                    if trivial_terms:
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
                    if trivial_terms:
                        operator_list = [operator_0, operator_1, operator_2, operator_3, operator_4, operator_5, operator_6, operator_7]

                    else:
                        operator_list = [operator_1, operator_2, operator_3, operator_4, operator_5, operator_6, operator_7]
                    for operator in operator_list:
                        circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time_prime_inner)
                        pauli_list.append(operator)
                        
                        coeffs.append(evo_time_prime_inner/evo_time)

                        
        
        # Don't 100% need these, just for figuring out the alphas
        self.BD_Pauli_List = PauliList(pauli_list)
        self.BD_coeffs_list = coeffs
        
        if return_circuit:
            return circuit.decompose()
    
    def define_TC_circuit(self , gamma_TC: float, trivial_terms:bool = False, return_circuit:bool = True)-> QuantumCircuit | None:
        """
        Defines a quantum circuit to exhibit TC (Transitive closure) using a combination of one-body, two-body, 
        and three-body Pauli operators. 

        # Actually makes the circuit. Next iteration of this code should just take the pauli terms and coeffs (not including gamma) 
        # and use this to contruct the HSIM circuit for arb. gamma
        

        Parameters:
            gamma_TC (float): The coefficient for the Hamiltonian term corresponding to TC.
            trivial_terms (bool): If True, include the trivial terms in the Hamiltonian  NOT IMPLIMENTED. Default is False.
            return_circuit (bool): If True, return the quantum circuit. If False, return None. Default is True.
        
        Returns:
            QuantumCircuit: The constructed quantum circuit after decomposition.
        """
        if trivial_terms:
            raise ValueError("Trivial terms not yet implemented.")

        if gamma_TC == 0 and return_circuit is True:
            raise ValueError("gamma_TC must be nonzero.")
        
        if return_circuit:
            evo_time = gamma_TC*self.alpha_TC
        else:
            evo_time = 1
            
        basis = get_upper_triangular_basis(self.n)
        
        
        # Using operator like this is probably very inefficient re. matrix exponential. But will work for now.
        # Using sparse pauli op is definately better
        circuit = QuantumCircuit(self.q)
        pauli_list = []
        count = 0
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
                        pauli_list.append(operator)
                    
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
                        pauli_list.append(operator)
                    
                    
                    #three body terms
                    Z_string = np.zeros(self.q)
                    Z_string[basis[i,j]] = 1
                    Z_string[basis[j,k]] = 1
                    Z_string[basis[i,k]] = 1
                    operator = Pauli((Z_string, np.zeros(self.q), 0))
                    circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time)
                    pauli_list.append(operator)
                    
                    
                    
                    count +=1
        self.TC_Pauli_List = PauliList(pauli_list)
        if return_circuit:
            return circuit.decompose()

    def define_mixing_circuit(self, gamma_mix: float) -> QuantumCircuit:
        
        """
        Define a quantum circuit for time evolution of the mixing term.

        Parameters:
            gamma_mix (float): The coefficient for the Hamiltonian term corresponding to mixing.

        Returns:
            QuantumCircuit: The constructed quantum circuit after applying the mixing.
        """
        
        
        evo_time = gamma_mix
        
        # simple X mixer as in Layden
        circuit = QuantumCircuit(self.q)
        for l in range(self.q):         
            Z_string = np.zeros(self.q)
            X_string = np.zeros(self.q)
            X_string[l] = 1
            operator = Pauli((Z_string, X_string, 0))
            circuit = self.add_evo_to_circuit(circuit, operator, time = evo_time)

        return circuit.decompose()

    def add_evo_to_circuit(self, circuit: QuantumCircuit, op:Pauli, time: float = 0.2)-> QuantumCircuit:
        """
        
        Adds a time evolution gate (of a hgiven operator) to a quantum circuit.
        
        Parameters:
            circuit (QuantumCircuit): The quantum circuit to add the time evolution gate to.
            op (Pauli): The Pauli operator to evolve.
            time (float): The time to evolve the operator for. Default is 0.2.
            
        Returns:
            QuantumCircuit: The quantum circuit with the time evolution gate appended.
        """
        
        evo = PauliEvolutionGate(op, time=time)
        circuit.append(evo, range(self.q))
        return circuit

    def quantum_proposal(self, s: str, multiple: int = 1, output_statevector: bool = False) -> str | list | npt.NDArray:
        """
        Propose a new configuration based on the given bitstring.
        This function generates a new bitstring configuration by taking a single shot of a quantum circuit.
        The quantum circuit is time evolution of a Hamiltonian that (optionally) includes mixing, 
        transitive closure, and BD action terms.
        
        Parameters:
            s (str): The initial bitstring configuration.
            multiple (int): The number of samples to generate. Default is 1. More than one can be useful for analysing the proposal.
            output_statevector (bool): Whether to return the output statevector instead of the bitstring. Default is False.

        Returns:
            str or list: If `multiple` is 1, returns a single new bitstring configuration.
                If `multiple` is greater than 1, returns a list of new bitstring configurations.
        """
        if self.gamma_ranges:
            self.gamma_TC, self.gamma_BD, self.gamma_mixing = self.allocate_gammas_from_ranges()
            #print("chosen gammas: ", self.gamma_TC, self.gamma_BD, self.gamma_mixing)
        if self.is_t_a_range:
            self.t = self.allocate_t_from_range()
            #print("chosen t: ", self.t)
            
        mixing_circ = self.define_mixing_circuit(gamma_mix= self.gamma_mixing)
        if self.gamma_TC > 0:
            TC_circuit = self.define_TC_circuit(gamma_TC = self.gamma_TC)
        if self.gamma_BD > 0:            
            BD_circuit = self.define_BD_circuit(gamma_BD = self.gamma_BD)
        
        #set initial state
        qc = QuantumCircuit(self.q)
        for i, x in enumerate(s):
            if x == "1":
                qc.x(i)
                
        #If doing transitive closure
        if self.gamma_TC > 0:
            qc.compose(TC_circuit, inplace = True)
        
        # If doing BD action
        if self.gamma_BD> 0:
            qc.compose(BD_circuit, inplace = True)
        

        
        for t in range(self.t):
            #Always mix
            qc.compose(mixing_circ, inplace = True)
            if self.gamma_TC>0:
                qc.compose(TC_circuit, inplace = True)
            if self.gamma_BD>0:
                qc.compose(BD_circuit, inplace = True)

        
        # Use Qiskit-Qulacs to run the circuit
        backend = QulacsBackend()

        #qc = self.pm.run(qc)
        
        if output_statevector:
            # Get the statevector of the circuit
            statevector = backend.run(qc).result().get_statevector()
            # Reorder the statevector from little-endian to big-endian
            num_qubits = int(np.log2(len(statevector)))
            statevector = statevector.reshape([2] * num_qubits).transpose(range(num_qubits - 1, -1, -1)).reshape(-1)
            return statevector
        else:
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


    def classical_proposal(self, s: str, multiple: int = 1) -> str | list:
        """
        Generates a new state or a list of new states by applying random moves 
        to the input state `s`. The moves are selected from a predefined set 
        of possible moves.
        Args:
            s (str): The input state to which the moves will be applied.
            multiple (int, optional): The number of new states to generate. 
                Defaults to 1. If `multiple` is 1, a single new state is returned; 
                otherwise, a list of new states is returned.
        Returns:
            str | list: A single new state (if `multiple` is 1) or a list of new 
            states (if `multiple` > 1).
        Raises:
            ValueError: If an invalid move is selected from the predefined set 
            of moves.
        """
        
        
        #select move
        s_primes = []
        for m in range(multiple):
            move = np.random.choice(self.moves)
            if move == 0:
                s_prime = self.link_move(s)
            elif move == 1:
                s_prime = self.relation_move(s)
            if multiple == 1:
                return s_prime
            else:
                s_primes.append(s_prime)
        return s_primes
    
    def link_move(self, s:str) -> str:
        """Perform a link move operation on a binary string `s` representing the upper 
        triangular part of a causal matrix. The function modifies the causal matrix 
        by either removing or adding a link between two randomly selected nodes, while
        ensuring that the resulting matrix maintains transitive closure. As described in Henson paper
        
        Args:
            s (str): A binary string representing the upper triangular part of the 
                causal matrix.
        Returns:
            str: A binary string representing the updated upper triangular part of 
                the causal matrix after the link move operation.
        
        """
        
        s_mat = np.zeros((self.n, self.n), dtype=np.int32)
        s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
        
        # pick two random elements i and j
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)
        
        # make sure i != j
        while i == j:
            j = np.random.randint(0, self.n)
            
        y = max(i,j)
        x = min(i,j)
        
        
        if is_linked(x,y, s_mat):
            
            link_matrix = transitive_reduction(s_mat)
            link_matrix[x,y] = 0
            s_mat = transitive_closure(link_matrix)

            
        elif is_suitable_pair(x,y, s_mat):
            #print("suitable pair")

            
            #self.causal_matrix[x,y] = 1 # Relate x and y
            
            link_matrix = transitive_reduction(s_mat)
            
            if link_matrix[x,y] != 0:
                print("suitable link error")   

            link_matrix[x,y] = 1
            
            
            s_mat = transitive_closure(link_matrix)
            
            
            
            link_matrix_new = transitive_reduction(s_mat)
            
            if link_matrix_new[x,y] != 1:
                print("link move no change error")
                print("x,y: ", x,y)
                print(link_matrix)
                print(link_matrix_new)            
            
        else:
            pass
        
        return "".join(str(bit) for bit in s_mat[np.triu_indices(self.n, 1)])

    def relation_move(self, s:str):
        """
        Perform a relation move on a binary string representation of an upper triangular matrix.
        This function modifies the binary string `s` by randomly selecting two indices (i, j),
        ensuring they are distinct, and then determining whether to update the corresponding
        entry in the upper triangular matrix based on specific conditions.
        Args:
            s (str): A binary string representing the upper triangular part of an n x n matrix,
                excluding the diagonal. The length of the string should be `n * (n - 1) / 2`.
        Returns:
            str: A modified binary string representing the updated upper triangular matrix.
        
        """
        

        
        s_mat = np.zeros((self.n, self.n), dtype=np.int32)
        s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
        # pick two random elements i and j
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)
        
        # make sure i != j
        while i == j:
            j = np.random.randint(0, self.n)
            
        y = max(i,j)
        x = min(i,j)
        
        
        
        if is_linked(x,y, s_mat):
            #print("linked")
            s_mat[x,y] = 0
        elif is_critical_pair(x,y,s_mat):
            #print("critical pair")
            s_mat[x,y] = 1
        else:
            #print("not critical or linked pair")
            pass
        return "".join(str(bit) for bit in s_mat[np.triu_indices(self.n, 1)])
        
        
    def sample_uniform(self, s:str = None, num_samples:int = 100, sample_frequency:int = 100, T_therm:int = 100,  observables: List[str] = ["ordering_fraction", "height", "num_relations", "minimal_elements", "BD_action"]):
        """
        Samples the space of all causal sets, Omega, by using either a quantum or classical proposal.
        
        Parameters:
            s (str): The initial state of the system. If None, the initial state is set to the all-ones configuration.
            num_samples (int): The number of samples to generate. Default is 100.
            sample_frequency (int): The frequency at which samples are generated. Default is 100.
            T_therm (int): The number of thermalisation steps. Default is 100.
            observables (list): A list of observables to calculate. Default is ["ordering_fraction", "height", "num_relations", "minimal_elements"].
        
        
        Returns:
            dict: A dictionary with causal matrices as keys and their counts as values.
        """
        
        
        if s is None:
            s_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s = "1" * (self.n * (self.n - 1) // 2)
            s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
        elif type(s) == np.ndarray:
            s_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
            s = "".join(str(bit) for bit in s_mat[np.triu_indices(self.n, 1)])
        else:
            print("error with initial state")
        
        bitstring_chain = []
        sample_index = []
        
        
        acceptance_count = 0
        self_move_count = 0
        
        

        
        if "ordering_fraction" in observables:
            ordering_fractions_list = []
        if "height" in observables:
            heights_list = []
        if "num_relations" in observables:
            num_relations_list = []
        if "minimal_elements" in observables:
            minimal_elements_list = []
        if "BD_action" in observables:
            BD_action_list = [] 
        
        # Initial oservables
        bitstring_chain.append(int(s, 2))
        sample_index.append(0)
        
        if "ordering_fraction" in observables:
            ordering_fractions_list.append(ordering_fraction(s_mat))
        if "height" in observables:
            heights_list.append(height(s_mat))
        if "num_relations" in observables:
            num_relations_list.append(num_relations(s_mat))
        if "minimal_elements" in observables:
            minimal_elements_list.append(minimal_elements(s_mat))
        if "BD_action" in observables:
            BD_action_list.append(calculate_action(causal_matrix = s_mat, smeared = True, stdim = self.dimension, epsilon = self.epsilon, first_order_smearing = False, first_order_taylor = False))
        
        
        
        
        
        steps = num_samples * sample_frequency + T_therm +1
        start_time = time.time()
        for step in tqdm(range(1,steps)):
            s_prime = self.proposal(s)#self.quantum_proposal(s, TC=True, BD = True, mixing_time = 0.1)
            
            s_prime_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_prime_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s_prime]
            
            if not is_causal_matrix(s_prime_mat):
                pass
            elif s_prime == s:
                self_move_count += 1
            else:
                s = s_prime
                s_mat = s_prime_mat
                acceptance_count +=1



            
            if step > T_therm and step % sample_frequency == 0:
                # Convert the matrix to a string representation to use as a dictionary key
                
                
                bitstring_chain.append(int(s, 2))
                sample_index.append(step)
                
                if "ordering_fraction" in observables:
                    ordering_fractions_list.append(ordering_fraction(s_mat))
                if "height" in observables:
                    heights_list.append(height(s_mat))
                if "num_relations" in observables:
                    num_relations_list.append(num_relations(s_mat))
                if "minimal_elements" in observables:
                    minimal_elements_list.append(minimal_elements(s_mat))
                if "BD_action" in observables:
                    BD_action_list.append(calculate_action(causal_matrix = s_mat, smeared = True, stdim = self.dimension, epsilon = self.epsilon, first_order_smearing = False, first_order_taylor = False))
                    
        end_time = time.time()
        if self.verbose:
            print("Time taken: ", end_time - start_time, " (per step: ", (end_time - start_time)/steps, ", per sample ", (end_time - start_time)/num_samples, ")")
            print("acceptance rate: ", acceptance_count/steps)
            print("self move rate: ", self_move_count/steps)
        
        
        results = {"bitstring_chain": bitstring_chain, "sample_index": np.array(sample_index)}
        if "ordering_fraction" in observables:
            results["ordering_fractions"] = np.array(ordering_fractions_list)
        if "height" in observables:
            results["heights"] = np.array(heights_list)
        if "num_relations" in observables:
            results["num_relations"] = np.array(num_relations_list)
        if "minimal_elements" in observables:
            results["minimal_elements"] = np.array(minimal_elements_list)
        if "BD_action" in observables:
            results["BD_action"] = np.array(BD_action_list)
        return results
    
    def sample_BD(self, T:float, s:str = None, num_samples:int = 100, sample_frequency:int = 100, T_therm:int = 100,  observables: List[str] = ["ordering_fraction", "height", "num_relations", "minimal_elements", "BD_action"]):
        """
        Samples the space of all causal sets, Omega, by using Quantum proposal.
        
        Parameters:
            T (float): The temperature parameter for the simulation.
            s (str): The initial state of the system. If None, the initial state is set to the all-ones matrix.
            num_samples (int): The number of samples to generate. Default is 100.
            sample_frequency (int): The frequency at which samples are generated. Default is 100.
            T_therm (int): The number of thermalisation steps. Default is 100.
            observables (list): A list of observables to calculate. Default is ["ordering_fraction", "height", "num_relations", "minimal_elements"].

        
        Returns:
            dict: A dictionary with causal matrices as keys and their counts as values.
        """

        
        if s is None:
            s_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s = "1" * (self.n * (self.n - 1) // 2)
            s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
        elif type(s) == np.ndarray:
            s_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s]
            s = "".join(str(bit) for bit in s_mat[np.triu_indices(self.n, 1)])
        else:
            print("error with initial state")
        
        bitstring_chain = []
        sample_index = []
        
        
        
        acceptance_count = 0
        self_move_count = 0
        forbidden_count = 0
        
        
        if "ordering_fraction" in observables:
            ordering_fractions_list = []
        if "height" in observables:
            heights_list = []
        if "num_relations" in observables:
            num_relations_list = []
        if "minimal_elements" in observables:
            minimal_elements_list = []
        if "BD_action" in observables:
            BD_action_list = [] 
        
        # Initial oservables
        bitstring_chain.append(int(s, 2))
        sample_index.append(0)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        s_old = calculate_action(causal_matrix = s_mat, smeared = True, stdim = self.dimension, epsilon = self.epsilon, first_order_smearing = False, first_order_taylor = False)#
        if "ordering_fraction" in observables:
            ordering_fractions_list.append(ordering_fraction(s_mat))
        if "height" in observables:
            heights_list.append(height(s_mat))
        if "num_relations" in observables:
            num_relations_list.append(num_relations(s_mat))
        if "minimal_elements" in observables:
            minimal_elements_list.append(minimal_elements(s_mat))
        if "BD_action" in observables:
            BD_action_list.append(s_old)
        
        
        
        
        
        steps = num_samples * sample_frequency + T_therm +1
        start_time = time.time()
        for step in tqdm(range(1,steps)):
            s_prime = self.proposal(s)#self.quantum_proposal(s, TC=True, BD = True, mixing_time = 0.1)
            
            s_prime_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_prime_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s_prime]
            

            # accept or reject according to netropolis hastings
            s_new = calculate_action(causal_matrix = s_prime_mat, smeared = True, stdim = self.dimension, epsilon = self.epsilon, first_order_smearing = False, first_order_taylor = False)



            if not is_causal_matrix(s_prime_mat):
                forbidden_count +=1
            elif s_prime == s:
                self_move_count += 1
            else:
                
                
                accept = self.accept_reject(s_old, s_new, T)
                if accept:
                    s = s_prime
                    s_mat = s_prime_mat
                    s_old = s_new 
                    acceptance_count +=1       
                else:
                    pass
                            
                
            if step > T_therm and step % sample_frequency == 0:
                # Convert the matrix to a string representation to use as a dictionary key
                
                
                bitstring_chain.append(int(s, 2))
                sample_index.append(step)
                
                if "ordering_fraction" in observables:
                    ordering_fractions_list.append(ordering_fraction(s_mat))
                if "height" in observables:
                    heights_list.append(height(s_mat))
                if "num_relations" in observables:
                    num_relations_list.append(num_relations(s_mat))
                if "minimal_elements" in observables:
                    minimal_elements_list.append(minimal_elements(s_mat))
                if "BD_action" in observables:
                    BD_action_list.append(calculate_action(causal_matrix = s_mat, smeared = True, stdim = self.dimension, epsilon = self.epsilon, first_order_smearing = False, first_order_taylor = False))
                    
        end_time = time.time()
        
        if self.verbose:
            print("Time taken: ", end_time - start_time, " (per step: ", (end_time - start_time)/steps, ", per sample ", (end_time - start_time)/num_samples, ")")
            print("acceptance rate: ", acceptance_count/steps)
            print("self move rate: ", self_move_count/steps)
            print("forbidden rate: ", forbidden_count/steps)
        
        
        results = {"bitstring_chain": bitstring_chain, "sample_index": np.array(sample_index)}
        
        
        results["acceptance_rate"] = acceptance_count/steps
        results["self_move_rate"] = self_move_count/steps
        results["forbidden_rate"] = forbidden_count/steps
        
        
        if "ordering_fraction" in observables:
            results["ordering_fractions"] = np.array(ordering_fractions_list)
        if "height" in observables:
            results["heights"] = np.array(heights_list)
        if "num_relations" in observables:
            results["num_relations"] = np.array(num_relations_list)
        if "minimal_elements" in observables:
            results["minimal_elements"] = np.array(minimal_elements_list)
        if "BD_action" in observables:
            results["BD_action"] = np.array(BD_action_list)
        return results
    
    
    def BD_acceptance_probability(self, s_old, s_new, T):
        """
        Calculate the acceptance probability for a Metropolis-Hastings step in the BD action.

        Parameters:
            s_old (float): The action value of the old configuration.
            s_new (float): The action value of the new configuration.
            T (float): The temperature parameter.

        Returns:
            float: The acceptance probability.
        """

        delta = s_old - s_new
        if delta < 0:
            return np.exp(delta/ T)
        else:
            return 1.0
        
    def accept_reject(self, s_old, s_new, T):
        """
        Perform the acceptance-rejection step for the Metropolis-Hastings algorithm.

        Parameters:
            s_old (float): The action value of the old configuration.
            s_new (float): The action value of the new configuration.
            T (float): The temperature parameter.

        Returns:
            bool: True if the new configuration is accepted, False otherwise.
        """
        p = self.BD_acceptance_probability(s_old, s_new, T)
        u = np.random.uniform(0, 1)
        
        return u < p
    
    
    

