# Code to constrain the space of all causal sets, Omega to that of transitively closed matrices (i.e. causal matrices) through hamiltonian constraints. 

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_qulacs import QulacsProvider
from qiskit_qulacs.qulacs_backend import QulacsBackend
import math
from helpers import *

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
    
    def __init__(self, n:int, method: str = "quantum", epsilon: float = 0.1, qargs: dict = {}, cargs :dict = {}):
        """
        Initialises the Sampler class.
        
        Parameters:
        n (int): The cardinality of the causal set.
        method (str): The method to use for sampling. Options are "quantum" and "classical".
        qargs (dict): A dictionary of arguments to pass to the quantum, including the follwoing:
            gammas (list[float]]): [gamma_TC, gamma_BD, gamma_mix]: Gamma parameters that govern the magnitude of the hamiltonian terms.
                gamma_TC: If nonzero, add Transitive Closure to Hamiltonian.
                gamma_BD: If nonzero, add BD action approximation to Hamiltonian.
                gamma_mix = (1- gamma_TC - gamma_BD) If nonzero, add the mixing term to the Hamiltonian. Can be inferred from 'gamma_TC' and `gamma_BD' if not provided
        cargs (dict): A dictionary of arguments to pass to the classical sampler, including the following:
            link_move (bool): If True, allow link moves. Default is False.
            relation_move (bool): If True, allow relation moves. Default is False.
        t (int): The number of time steps to evolve the Hamiltonian. Default is 5.
        """
        self.n = n # number of elements in the causal set
        self.epsilon = epsilon # small parameter used to smear the action over subgraphs
        self.dt = 0.8 # time step for the evolution same as Layden
        
        
        self.q = int(n*(n-1)/2) # number of qubits (upper triangular matrix)
        
        
        
        if method == "quantum":
            #self.base_BD_circuit = self.define_BD_circuit()
            #self.base_TC_circuit = self.define_TC_circuit()
            #self.mixing_circ = self.build_mixing_circuit().decompose()
            self.proposal = self.quantum_proposal
            
            
            
            
            if not qargs:
                print("qargs dictionary is empty. Proceeding with default values.")
            
            
            if "gammas" in qargs:
                if type(qargs["gammas"]) != list:
                    if type(qargs["gammas"]) != tuple:
                        if type(qargs["gammas"]) != np.ndarray:
                            raise ValueError("Gammas must be a list, tuple or array of floats or tuples.")
                for val in qargs["gammas"]:
                    if type(val) != float:
                        if type(val) != int:
                            if type(val) != tuple:
                                raise ValueError("Gammas must be floats, integers or tuples of floats representing a range from which to sample.")
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
                

                #self.alpha_mix = 
                self.alpha_TC = self.get_alpha_TC()
                self.alpha_BD = self.get_alpha_BD()
            else:
                raise ValueError("Gammas must be specified.")
            
            print("gamma_TC: ", self.gamma_TC)
            print("gamma_BD: ", self.gamma_BD)
            print("gamma_mixing: ", self.gamma_mixing)
            
            
            """
            if "TC" in qargs:
                self.TC = qargs["TC"]
                if type(self.TC) != float:
                    raise ValueError("TC must be a float value between 0 and 1 (inclusive).")
            else:
                self.TC = 0.
                
            if "BD" in qargs:
                self.BD = qargs["BD"]
                if type(self.BD) != float:
                    raise ValueError("BD must be a float value between 0 and 1 (inclusive).")
            else:
                self.BD = 0.
            
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
            """
                
            if "t" in qargs:
                self.t = qargs["t"]
                if type(self.t) != int:
                    raise ValueError("t must be an integer value.")
                else:
                    self.t = 5
        
        elif method == "classical":
            #raise ValueError("Classical method not yet implemented.")
            self.proposal = self.classical_proposal
            #list of moves that can be selected
            self.moves = []
            
            if len(cargs) == 0:
                print("cargs dictionary is empty. Proceeding with default (both link and relation move).")
                self.moves = [0,1]
            
            if "link_move" in cargs:
                if type(cargs["link_move"]) != bool:
                    print("Link_move must be a boolean value. Defaulting to False.")
                else:
                    if cargs["link_move"]:
                        self.moves.append(0)
            if "relation_move" in cargs:
                if type(cargs["relation_move"]) != bool:
                    print("Relation_move must be a boolean value. Defaulting to False.")
                else:
                    if cargs["relation_move"]:
                        self.moves.append(1)
        
            if len(self.moves) == 0:
                print("Must have either or both link and relation moves, not neither")
            
        else:
            raise ValueError("Invalid method. You gave: '" + str(method) + "'. Please choose 'quantum' or 'classical'.")



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
        epsilon (float): A small parameter used to smear the action over subgraphs. Default is 0.1.
        evo_time (int): The total evolution time for the circuit. Default is 10. 
            EVO TIME NEEDS FIXED. MUST BE NORMALISED ETC. ANALYTICALLY (*NOT DONE YET")
        
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
        
        if trivial_terms:
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
                
                if trivial_terms:
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
                evo_time_prime_inner = (3/2)*(epsilon**3)*evo_time
                
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
        
    def get_alpha_BD(self):
        """
        Calculate the alpha parameter for the BD action.
        """
        
        self.define_BD_circuit(gamma_BD = self.gamma_BD, epsilon = self.epsilon, return_circuit=False)
        
        # Qemcmc alpha self.alpha = np.sqrt(self.num_spins) / np.sqrt( sum([J[i][j]**2 for i in range(self.num_spins) for j in range(i)]) + sum([h[j]**2 for j in range(self.num_spins)])  )
        #np.sqrt( sum([J[i][j]**2 for i in range(self.num_spins) for j in range(i)]) + sum([h[j]**2 for j in range(self.num_spins)])  )
        
        coeffs_list = np.copy(self.BD_coeffs_list)
        Pauli_List = copy.copy(self.BD_Pauli_List)

        
        matrix = SparsePauliOp(Pauli_List, np.array(coeffs_list)).to_matrix()
        norm = np.linalg.norm(matrix)
        #print("norm: ", norm)
        norm_manual = 0
        for i in coeffs_list:
            norm_manual += (i)**2
        norm_manual = np.sqrt(norm_manual)
        
        # Fixing it so that the single Z terms are all combined properly
        # Previously, two different terms, both Z acting on same qubits were treated the same.
        sign_list = np.zeros(len(coeffs_list))
        for i in range(len(coeffs_list)):

            sign = Pauli_List[i].to_label()[0]
            if sign == "-":
                sign_list[i] = 1
                #print("before fixing: ", Pauli_List[i].to_label())
                Pauli_List[i] = -1*Pauli_List[i]
                #print("after fixing: ", Pauli_List[i].to_label())
        
        #print("pauli list: ", Pauli_List)
        #print("sign_list: ", sign_list)
        #print("coeffs list: ", coeffs_list)
        
        new_PL = []
        new_coeffs = []
        for i, p in enumerate(Pauli_List):
            if p not in new_PL:
                new_PL.append(p)
                if sign_list[i] == 1:
                    new_coeffs.append(-1*coeffs_list[i])
                else:
                    new_coeffs.append(coeffs_list[i])
            else:
                index = new_PL.index(p)
                if sign_list[i] == 1:
                    new_coeffs[index] -= coeffs_list[i]
                else:
                    new_coeffs[index] += coeffs_list[i]
        
        #print("new PL: ", new_PL)
        #print("new coeffs: ", new_coeffs)
        
        norm_manual = 0
        for i in new_coeffs:
            norm_manual += (i)**2
        norm_manual = np.sqrt(norm_manual)
        
        matrix = SparsePauliOp(new_PL, np.array(new_coeffs)).to_matrix()
        norm = np.linalg.norm(matrix)
        
        #print("norm 2: ", norm)
        print("alpha_BD denominator: ", norm_manual)
        print("alpha_BD : ", np.sqrt(self.n)/norm_manual)
        alpha_BD = np.sqrt(self.n)/norm_manual
        return alpha_BD
        
    
    def get_alpha_TC(self):
        
        """Calculate the alpha parameter for the TC action."""
        
        self.define_TC_circuit(gamma_TC = self.gamma_TC, return_circuit=False)
        
        Pauli_List = copy.copy(self.TC_Pauli_List)
        coeffs_list = np.ones(len(self.TC_Pauli_List))*(1/8)
        
        #print("Pauli_List: ", Pauli_List)
        #print("coeffs_list: ", coeffs_list)

        
        matrix = SparsePauliOp(Pauli_List, np.array(coeffs_list)).to_matrix()
        norm = np.linalg.norm(matrix)
        norm_manual = 0
        for i in coeffs_list:
            norm_manual += (i)**2
        norm_manual = np.sqrt(norm_manual)
        #print("norm manual: ", norm_manual)
        
        
        sign_list = np.zeros(len(coeffs_list))
        for i in range(len(coeffs_list)):

            sign = Pauli_List[i].to_label()[0]
            if sign == "-":
                sign_list[i] = 1
                Pauli_List[i] = -1*Pauli_List[i]
        
        
        new_PL = []
        new_coeffs = []
        for i, p in enumerate(Pauli_List):
            if p not in new_PL:
                new_PL.append(p)
                if sign_list[i] == 1:
                    new_coeffs.append(-1*coeffs_list[i])
                else:
                    new_coeffs.append(coeffs_list[i])
            else:
                index = new_PL.index(p)
                if sign_list[i] == 1:
                    new_coeffs[index] -= coeffs_list[i]
                else:
                    new_coeffs[index] += coeffs_list[i]
        
        #print("new PL: ", new_PL)
        #print("new coeffs: ", new_coeffs)
        
        norm_manual = 0
        for i in new_coeffs:
            norm_manual += (i)**2
        norm_manual = np.sqrt(norm_manual)
        
        matrix = SparsePauliOp(new_PL, np.array(new_coeffs)).to_matrix()
        norm = np.linalg.norm(matrix)
        
        #print("norm 2: ", norm)
        print("alpha_BD denominator: ", norm_manual)
        print("alpha_BD : ", np.sqrt(self.n)/norm_manual)
        alpha_BD = np.sqrt(self.n)/norm_manual
        return alpha_BD
        

    def define_TC_circuit(self , gamma_TC: float, trivial_terms:bool = False, return_circuit:bool = True)-> QuantumCircuit | None:
        """
        Defines a quantum circuit to exhibit TC (Transitive closure) using a combination of one-body, two-body, 
        and three-body Pauli operators. 

        # Actually makes the circuit. Next iteration of this code should just take the pauli terms and coeffs (not including gamma) 
        # and use this to contruct the HSIM circuit for arb. gamma


        Parameters:
            gamma_TC (float): The coefficient for the Hamiltonian term corresponding to TC.
            trivial_terms (bool): If True, include the trivial terms in the Hamiltonian. Default is False.
            return_circuit (bool): If True, return the quantum circuit. If False, return None. Default is True.
        
        Returns:
            QuantumCircuit: The constructed quantum circuit after decomposition.
        """

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
        mixing_circ = self.define_mixing_circuit(gamma_mix= self.gamma_mixing)
        if self.gamma_TC > 0:
            TC_circuit = self.define_TC_circuit(gamma_TC = self.gamma_TC)
        if self.gamma_BD > 0:            
            BD_circuit = self.define_BD_circuit(gamma_BD = self.gamma_BD, epsilon = self.epsilon)
        
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
    
    def link_move(self, s) -> str:
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

            
            """
            # Find all pairs k, l between incpast(x) and incfut(y)
            # Remove relations
            
            self.causal_matrix[x,y] = 0 # Unrelate x and y
            
            for k in range(0, x+1):
                if self.causal_matrix[k,x] == 1: # If k is incpast(x):
                    for l in range(y, self.n):
                        if self.causal_matrix[y,l] == 1: # If l is incfut(y):   
                            self.causal_matrix[k,l] = 0 # Unrelate all k (incpast(x)) and l (incfut(y))
            
            # restore every element by transitivity (where relations are inferred by elements other than x and y)
            # Think that since it is incfut and incpast, 
            # we can just do full transitive closure, although this is innefficient
            self.causal_matrix = self.transitive_closure(self.causal_matrix)
            
            """
            
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
            
            
            """
            Doing exactly as in the paper doesnt work as it doesnt escape anti-chains
            original_hamm = np.sum(self.causal_matrix)
            
            # add relation between all incpast(x) and incfut(y)
            for k in range(0, x+1):
                if self.causal_matrix[k,x] == 1: # If k is incpast(x):
                    
                    for l in range(y, self.n):
                        if self.causal_matrix[y,l] == 1: # If l is incfut(y): 
                            
                            self.causal_matrix[k,l] = 1 # Relate all k (incpast(x)) and l (incfut(y))
            end_hamm = np.sum(self.causal_matrix)
            
            print("hamming weight change: ", end_hamm - original_hamm)"""
            
        else:
            #print("not suitable or linked pair")
            pass
        #return s_mat
        return "".join(str(bit) for bit in s_mat[np.triu_indices(self.n, 1)])
            
    def relation_move(self, s):
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
            



        
    def sample(self, s = None, num_samples = 100, sample_frequency = 100, T_therm = 100, accept_all_legitimate_moves = False, observables = ["ordering_fraction", "height", "num_relations", "minimal_elements"]):
        """
        Samples the space of all causal sets, Omega, by using Quantum proposal.
        
        Parameters:
            s (str): The initial state of the system. If None, the initial state is set to the all-ones matrix.
            num_samples (int): The number of samples to generate. Default is 100.
            sample_frequency (int): The frequency at which samples are generated. Default is 100.
            T_therm (int): The number of thermalisation steps. Default is 100.
            accept_all_legitimate_moves (bool): If True, accept all legitimate moves. Default is False.
                If False: Use BD action sampling algorithm.
            observables (list): A list of observables to calculate. Default is ["ordering_fraction", "height", "num_relations", "minimal_elements"].
        
        Returns:
            dict: A dictionary with causal matrices as keys and their counts as values.
        """
        if s is None:
            s_mat = np.ones((self.n, self.n), dtype=np.int32)
            s = "".join(str(i) for i in s_mat[np.triu_indices(self.n, 1)])
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
        
        
        
        
        
        steps = num_samples * sample_frequency + T_therm +1
        start_time = time.time()
        for step in tqdm(range(1,steps)):
            s_prime = self.proposal(s)#self.quantum_proposal(s, TC=True, BD = True, mixing_time = 0.1)
            
            s_prime_mat = np.zeros((self.n, self.n), dtype=np.int32)
            s_prime_mat[np.triu_indices(self.n, 1)] = [int(bit) for bit in s_prime]
            
            #s_prime_mat = np.frombuffer(s_prime.tostring(), dtype=np.int32).reshape(self.n, self.n)
            # If sampling form \Omega uniformly(ish)
            if accept_all_legitimate_moves:

                if not is_causal_matrix(s_prime_mat):
                    pass
                elif s_prime == s:
                    self_move_count += 1
                else:
                    s = s_prime
                    s_mat = s_prime_mat
                    acceptance_count +=1
            else: # If sampling from BD action
                raise NotImplementedError("Not yet implemented BD action sampling algorithm.")
            
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
                    
        end_time = time.time()
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
        
        return results
    

