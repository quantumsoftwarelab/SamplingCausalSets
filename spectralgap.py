
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import scipy as sp
import joblib
from typing import List

from Sampler import Sampler
from helpers import make_basis, transitive_closure, get_unique_causal_bitstrings, transitive_reduction, is_causal_matrix, calculate_action, get_unique_matrices, transitive_closure, get_upper_triangular_basis, num_relations, height, ordering_fraction, minimal_elements, is_critical_pair, is_suitable_pair, is_linked,is_incpast, is_incfut




class SpectralGap:
    """
    A class to calculate the spectral gap, acceptance, and proposal matrices for a given sampler.

    This class provides methods to analyze the properties of a Markov chain defined by a sampler,
    focusing on the spectral gap which characterizes the convergence rate of the chain.

    Attributes:
        sampler (Sampler): The sampler object defining the Markov chain.
        temp (float): The temperature parameter for the simulation.
        results_dir (str): The directory to save and load results.
    """

    def __init__(self, sampler: Sampler, temp: float, results_dir: str):
        """
        Initializes the SpectralGap class.

        Args:
            sampler (Sampler): The sampler object.
            temp (float): The temperature for the simulation. Only relevant if doing bd action sampling
            results_dir (str): The directory to store results.
        """
        self.sampler: Sampler = sampler
        self.temp: float = temp
        self.t_str: str = str(np.format_float_positional(temp, precision=2, unique=False, fractional=False, trim='k')).replace(".", "_")
        self.results_dir: str = results_dir
        
        # Pre-compute unique matrices and bitstrings for the given cardinality n
        _, _ = get_unique_matrices(sampler.n)
        _, self.unique_causal_bitstrings = get_unique_causal_bitstrings(sampler.n)
        
        
    def find_acceptance_matrix(self, uniform: bool = False) -> np.ndarray:
        """
        Calculates or loads the acceptance matrix A.

        The acceptance matrix A[i, j] contains the probability of accepting a move from state i to state j.
        If `uniform` is True, it returns a matrix of ones. Otherwise, it calculates the matrix based on
        the Benincasa-Dowker action and the given temperature, caching the result for future use.

        Args:
            uniform (bool, optional): If True, returns a uniform acceptance matrix. Defaults to False.

        Returns:
            np.ndarray: The acceptance matrix.
        
        Raises:
            ValueError: If the sampler dimension is not 2 or 4.
        """
        if uniform:
            # For a uniform sampler, all moves are accepted
            num_causal_bitstrings = len(self.unique_causal_bitstrings)
            return np.ones((num_causal_bitstrings, num_causal_bitstrings), dtype=np.float128)

        # Determine the filepath for the cached acceptance matrix based on sampler dimension and temperature
        if self.sampler.dimension == 2:
            A_filepath = os.path.join(self.results_dir, "A_mats_2d", f"A_mat_{self.sampler.n}_{self.t_str}.pkl")
        elif self.sampler.dimension == 4:
            A_filepath = os.path.join(self.results_dir, "A_mats_4d", f"A_mat_{self.sampler.n}_{self.t_str}.pkl")
        else:
            raise ValueError("Only 2D and 4D samplers are supported for now in find_acceptance_matrix")
        
        # Try to load the pre-calculated matrix
        try:
            with open(A_filepath, 'rb') as fileObj:
                A = pickle.load(fileObj)
            return A
        except FileNotFoundError:
            pass  # If not found, calculate it
        
        # Calculate the acceptance matrix from scratch
        num_cs = len(self.unique_causal_bitstrings)
        A = np.zeros((num_cs, num_cs))
        
        # Pre-calculate the smeared Benincasa-Dowker actions for all unique causal structures
        smeared_BD_actions = np.zeros(num_cs)
        basis = make_basis(self.sampler.n)
        for i, string in enumerate(self.unique_causal_bitstrings):
            mat = np.zeros((self.sampler.n, self.sampler.n), dtype=np.int32)
            for count, pair in enumerate(basis):
                mat[pair[0], pair[1]] = int(string[count])
            smeared_BD_actions[i] = calculate_action(mat, epsilon=self.sampler.epsilon, stdim=self.sampler.dimension)
            
        # Populate the acceptance matrix
        for i in range(num_cs):
            for j in range(num_cs):
                if i != j:
                    A[i, j] = self.sampler.BD_acceptance_probability(smeared_BD_actions[i], smeared_BD_actions[j], self.temp)
                else:
                    A[i, j] = 1.0  # A move to the same state is always accepted
        
        # Cache the newly calculated matrix
        os.makedirs(os.path.dirname(A_filepath), exist_ok=True)
        with open(A_filepath, 'wb') as fileObj:
            pickle.dump(A, fileObj)
            
        return A

    def find_proposal_matrix_brute_force(self, multiple: int = 10) -> np.ndarray:
        """
        Calculates the proposal matrix Q using a brute-force sampling method.

        This method iterates through each unique causal bitstring, generates multiple proposals,
        and counts the transitions to other causal bitstrings. Moves to non-causal structures
        are counted as self-loops.

        Args:
            multiple (int, optional): The number of proposals to generate for each state. Defaults to 10.

        Returns:
            np.ndarray: The proposal matrix Q.
        """
        num_cs = len(self.unique_causal_bitstrings)
        Q = np.zeros((num_cs, num_cs))
        
        # Create a mapping from bitstring to index for quick lookups
        bitstring_to_index = {s: i for i, s in enumerate(self.unique_causal_bitstrings)}
        
        for i, s in tqdm(enumerate(self.unique_causal_bitstrings), total=num_cs, desc="Processing causal bitstrings brute force"):
            # Generate proposals for the current state
            output_list = self.sampler.proposal(s, multiple=multiple)
            forbidden_moves = 0
            
            # Count transitions
            for bitstring in output_list:
                bitstring_str = "".join(map(str, bitstring))
                j = bitstring_to_index.get(bitstring_str)
                if j is not None:
                    Q[i, j] += 1
                else:
                    forbidden_moves += 1
            
            # Add forbidden moves to the diagonal
            Q[i, i] += forbidden_moves
            
            # Normalize the row to get probabilities
            row_sum = np.sum(Q[i, :])
            if row_sum > 0:
                Q[i, :] /= row_sum
                
        return Q

    def get_individual_q_matrix(self, causal_indices: List[int]) -> np.ndarray:
        """
        Calculates a single proposal matrix Q for a quantum sampler.

        This method is designed to be used with joblib for parallel computation, especially
        when estimating the proposal matrix through repeated sampling (e.g., for jackknife).

        Args:
            causal_indices (List[int]): Indices of causal bitstrings in the full list of bitstrings.

        Returns:
            np.ndarray: A single calculated proposal matrix Q.
        """
        num_cs = len(self.unique_causal_bitstrings)
        Q = np.zeros((num_cs, num_cs))
        
        for i, s in enumerate(self.unique_causal_bitstrings):
            # Get the statevector from the quantum proposal
            statevector = self.sampler.quantum_proposal(s, multiple=0, output_statevector=True)
            prob_vector = np.absolute(statevector)**2
            
            # Filter for probabilities corresponding to causal states
            prob_causal_vector = prob_vector[causal_indices] 
            
            
            # Assign transition probabilities
            Q[i, :] = prob_causal_vector #
            
            # The diagonal element includes the probability of staying in the same state
            # plus the summed probability of transitioning to any non-causal state.
            prob_non_causal = np.sum(prob_vector) - np.sum(prob_causal_vector)
            Q[i, i] += prob_non_causal
            
        return Q
    
    def find_proposal_matrix(self, causal_indices: List[int], jackknife: bool = False,
                        how_many_samples_if_sampling: int = 10, save_matrix: bool = False) -> np.ndarray:
        """
        Calculates or loads the proposal matrix Q.

        This is a central method that handles various ways of obtaining the proposal matrix:
        - Loading from a cached file if available.
        - Calculating for a classical sampler by analyzing possible moves.
        - Calculating for a quantum sampler by simulating the quantum circuit.
        - Estimating via sampling for quantum samplers with parameter ranges (e.g., for jackknife).

        Args:
            causal_indices (List[int]): Indices of causal bitstrings.
            jackknife (bool, optional): Whether to prepare for jackknife resampling. Defaults to False.
            how_many_samples_if_sampling (int, optional): Number of samples for estimation. Defaults to 10.
            save_matrix (bool, optional): Whether to cache the resulting matrix. Defaults to False.

        Returns:
            np.ndarray: The proposal matrix Q.
        
        Raises:
            ValueError: If the classical sampler has no moves enabled.
        """
        # Return the matrix if it's already stored in the sampler
        if self.sampler.Q is not None:
            return self.sampler.Q

        # Construct filepath for caching
        if save_matrix:
            if self.sampler.method == "quantum":
                if not self.sampler.gamma_ranges:
                    filename = f"{self.sampler.method}_{self.sampler.n}_{self.sampler.dimension}_{self.sampler.gamma_TC}_{self.sampler.gamma_BD}_{self.sampler.gamma_mixing}.pkl"
                else:
                    filename = f"{self.sampler.method}_{self.sampler.n}_{self.sampler.dimension}_range_{self.sampler.TC_gamma_ratio[0]}_{self.sampler.TC_gamma_ratio[1]}_{self.sampler.BD_gamma_ratio[0]}_{self.sampler.BD_gamma_ratio[1]}.pkl"
            else:
                moves_str = "both" if 0 in self.sampler.moves and 1 in self.sampler.moves else ("link" if 0 in self.sampler.moves else "relation")
                filename = f"{self.sampler.method}_{self.sampler.n}_{self.sampler.dimension}_{moves_str}.pkl"
            
            Q_filepath = os.path.join(self.results_dir, "Q_mats", filename)
            os.makedirs(os.path.dirname(Q_filepath), exist_ok=True)
            
            # Load from cache if it exists
            if os.path.exists(Q_filepath):
                print("Loading cached Q matrix from:", Q_filepath, flush=True)
                with open(Q_filepath, "rb") as f:
                    Q = pickle.load(f)
                self.sampler.Q = np.copy(Q)
                return Q
        
        
        basis = make_basis(self.sampler.n)
        num_cs = len(self.unique_causal_bitstrings)

        if self.sampler.method == "quantum":
            # --- Quantum Sampler Logic ---
            if self.sampler.is_t_a_range or self.sampler.gamma_ranges:
                # --- Logic for Quantum Sampler with Parameter Ranges (Requires Sampling) ---
                if jackknife:
                    # Generate multiple Q matrices in parallel for jackknife analysis
                    Q_s = joblib.Parallel(n_jobs=-2)(
                        joblib.delayed(self.get_individual_q_matrix)(causal_indices)
                        for _ in range(how_many_samples_if_sampling)
                    )
                    
                    # Stack matrices and normalize each one
                    Q_final = np.array(Q_s)
                    row_sums = Q_final.sum(axis=2, keepdims=True)
                    Q_final = np.divide(Q_final, row_sums, where=row_sums!=0)

                    self.sampler.Q = np.copy(Q_final)
                else:
                    # Average over multiple samples to get a single estimated Q matrix
                    Q = np.zeros((num_cs, num_cs))
                    for _ in tqdm(range(how_many_samples_if_sampling), desc="Repeated sampling for quantum proposals"):
                        Q += self.get_individual_q_matrix(causal_indices)
                    
                    Q /= how_many_samples_if_sampling
                    row_sums = Q.sum(axis=1, keepdims=True)
                    Q = np.divide(Q, row_sums, where=row_sums!=0)
                    self.sampler.Q = np.copy(Q)

            else:
                # --- Logic for Quantum Sampler with Fixed Parameters ---
                Q = self.get_individual_q_matrix(causal_indices)
                row_sums = Q.sum(axis=1, keepdims=True)
                Q = np.divide(Q, row_sums, where=row_sums!=0)
                self.sampler.Q = np.copy(Q)

        elif self.sampler.method == "classical":
            # --- Classical Sampler Logic ---
            Q_L = np.zeros((num_cs, num_cs))  # For link moves
            Q_R = np.zeros((num_cs, num_cs))  # For relation moves
            
            link_move = 0 in self.sampler.moves
            relation_move = 1 in self.sampler.moves
                
            if not link_move and not relation_move:
                raise ValueError("Classical sampler must have at least one move type (link or relation) enabled.")
            
            bitstring_to_index = {s: i for i, s in enumerate(self.unique_causal_bitstrings)}

            if relation_move:
                # --- Calculate Transitions for Relation Moves ---
                for i, s in tqdm(enumerate(self.unique_causal_bitstrings), total=num_cs, desc="Processing classical (Relation moves)"):
                    s_mat = np.zeros((self.sampler.n, self.sampler.n), dtype=np.int32)
                    for count, pair in enumerate(basis):
                        s_mat[pair[0], pair[1]] = int(s[count])
                    
                    # Check for possible single-bit flip transitions (add/remove relation)
                    for k, c in enumerate(s):
                        flipped_char = "0" if c == "1" else "1"
                        s_j_str = s[:k] + flipped_char + s[k+1:]
                        
                        j = bitstring_to_index.get(s_j_str)
                        if j is not None:
                            pair = basis[k]
                            if c == "1" and is_linked(pair[0], pair[1], s_mat):
                                Q_R[i, j] += 1
                            elif c == "0" and is_critical_pair(pair[0], pair[1], s_mat):
                                Q_R[i, j] += 1

            if link_move:
                # --- Calculate Transitions for Link Moves ---
                for i, s in tqdm(enumerate(self.unique_causal_bitstrings), total=num_cs, desc="Processing classical (Link moves)"):
                    s_mat = np.zeros((self.sampler.n, self.sampler.n), dtype=np.int32)
                    for count, pair in enumerate(basis):
                        s_mat[pair[0], pair[1]] = int(s[count])
                    
                    for x in range(self.sampler.n):
                        for y in range(x + 1, self.sampler.n):
                            s_proposed_mat = None
                            if is_linked(x, y, s_mat):
                                # --- Unlinking Move ---
                                s_proposed_mat = np.copy(s_mat)
                                s_proposed_mat[x, y] = 0
                                for a in range(self.sampler.n):
                                    for b in range(a + 1, self.sampler.n):
                                        if s_mat[a, b] and is_incpast(a, x, s_mat) and is_incfut(b, y, s_mat):
                                            s_proposed_mat[a, b] = 0
                                
                            elif is_suitable_pair(x, y, s_mat):
                                # --- Linking Move ---
                                s_proposed_mat = np.copy(s_mat)
                                s_proposed_mat[x, y] = 1
                                for a in range(self.sampler.n):
                                    for b in range(a + 1, self.sampler.n):
                                        if is_incpast(a, x, s_proposed_mat) and is_incfut(b, y, s_proposed_mat):
                                            s_proposed_mat[a, b] = 1
                            
                            if s_proposed_mat is not None:
                                # Convert the resulting matrix back to a bitstring and find its index
                                s_mat_closed = transitive_closure(s_proposed_mat)
                                s_proposed_list = [str(s_mat_closed[pair[0], pair[1]]) for pair in basis]
                                s_proposed_str = "".join(s_proposed_list)
                                
                                j = bitstring_to_index.get(s_proposed_str)
                                if j is not None:
                                    Q_L[i, j] += 1

            # --- Combine and Normalize Classical Q Matrices ---
            if relation_move:
                stay_counts = self.sampler.q - Q_R.sum(axis=1)
                np.fill_diagonal(Q_R, stay_counts)
            if link_move:
                stay_counts = self.sampler.q - Q_L.sum(axis=1)
                np.fill_diagonal(Q_L, stay_counts)

            if relation_move and not link_move:
                Q = Q_R
            elif link_move and not relation_move:
                Q = Q_L
            else: # Both moves are active
                Q = (Q_L + Q_R) / 2
            
            row_sums = Q.sum(axis=1, keepdims=True)
            Q = np.divide(Q, row_sums, where=row_sums!=0)
            self.sampler.Q = np.copy(Q)
        
        # --- Caching and Return ---
        if save_matrix:
            with open(Q_filepath, "wb") as f:
                pickle.dump(self.sampler.Q, f)
        
        return self.sampler.Q
    
    def jackknife_spec_gaps(self, A: np.ndarray, Q_s: np.ndarray) -> tuple[float, float, np.ndarray]:
        """
        Calculates the spectral gap and its standard error using the jackknife resampling method.

        This is used when the proposal matrix Q is estimated from a set of samples (Q_s).
        It computes the spectral gap for each leave-one-out sample to estimate the variance.

        Args:
            A (np.ndarray): The acceptance matrix.
            Q_s (np.ndarray): A stack of proposal matrices, where each is a single sample.

        Returns:
            tuple[float, float, np.ndarray]:
                - The mean of the jackknife spectral gaps.
                - The standard error of the mean (SEM).
                - An analysis of moves from the full sample.
        """
        num_samples_taken = len(Q_s)
        spec_gaps = np.zeros(num_samples_taken)

        # Perform leave-one-out analysis
        for i in tqdm(range(num_samples_taken), desc="Jackknife iterations"):
            # Average all samples except the i-th one
            Q = np.sum(Q_s, axis=0) - Q_s[i]
            Q /= (num_samples_taken - 1)
            
            # Calculate spectral gap for this subsample
            _, delta, _ = self.find_spec_gap(A, Q)
            spec_gaps[i] = delta
        
        # Calculate the spectral gap using the full sample for comparison
        Q_full = np.sum(Q_s, axis=0) / num_samples_taken
        _, actual_spec_gap, moves_analyses = self.find_spec_gap(A, Q_full)
        
        # Compute mean and standard error from the jackknife samples
        mean_gap = np.mean(spec_gaps)
        sem = sp.stats.sem(spec_gaps)
        
        return mean_gap, sem, moves_analyses

    def find_spec_gap(self, A: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Calculates the spectral gap for a given transition matrix P.

        The transition matrix P is constructed from the proposal matrix Q and acceptance matrix A.
        The spectral gap is defined as 1 - |lambda_2|, where lambda_2 is the second largest
        eigenvalue of P.

        Args:
            A (np.ndarray): The acceptance matrix.
            Q (np.ndarray): The proposal matrix.

        Returns:
            tuple[np.ndarray, float, np.ndarray]:
                - The transition matrix P.
                - The calculated spectral gap (delta).
                - An array containing analysis of move types (self, rejected, combined).
        """
        # Construct the transition matrix P = Q * A )
        P = np.multiply(Q, A)
        
        self_probs = np.diag(P).copy()
        
        # Adjust the diagonal of P to account for rejected moves
        # The probability of staying in state i is the sum of the probability of proposing
        # to stay in i and the probabilities of proposing moves to other states that are rejected.
        for i in range(P.shape[0]):
            s = np.sum(P[i, :]) - P[i, i]  # Sum of probabilities of accepted moves to other states
            P[i, i] = 1 - s

        # Analyze the types of moves
        rejected_probs = 1 - np.sum(P, axis=1)
        self_and_rejected_probs = self_probs + rejected_probs
        
        avg_self_moves = np.mean(self_probs)
        avg_rejected_moves = np.mean(rejected_probs)
        avg_self_and_rejected_moves = np.mean(self_and_rejected_probs)
        moves_analyses = np.array([avg_self_moves, avg_rejected_moves, avg_self_and_rejected_moves])

        # Normalize P to ensure rows sum to 1 (making it a valid stochastic matrix)
        row_sums = P.sum(axis=1, keepdims=True)
        P = np.divide(P, row_sums, where=row_sums!=0)

        # Find the eigenvalues of the transition matrix
        e_vals, _ = sp.linalg.eig(P)
        
        # Sort eigenvalues by magnitude in descending order
        sorted_e_vals = np.flip(np.sort(np.abs(e_vals)))
        
        # The spectral gap is 1 minus the second largest eigenvalue magnitude
        delta = 1 - sorted_e_vals[1]
        
        # Sanity check: the largest eigenvalue should be close to 1
        if not np.isclose(sorted_e_vals[0], 1):
            print(f"Warning: Largest eigenvalue is {sorted_e_vals[0]}, not 1.")
            print(f"Method: {self.sampler.method}, Moves: {self.sampler.moves if self.sampler.method == 'classical' else 'N/A'}")
        
        return P, delta, moves_analyses
    
    def analyse_BD_proximity(self, Q: np.ndarray, color: str = "k", label: str = None, num_bins: int = 50) -> None:
        """
        Analyzes and plots the proximity of proposals in terms of Benincasa-Dowker (BD) action difference.

        Instead of Hamming distance, this function quantifies how "close" proposals are by the
        difference in their BD action. It generates a histogram where each bar represents the
        total proposal probability for transitions with a certain BD action difference.

        Args:
            Q (np.ndarray): The proposal matrix.
            color (str, optional): Color for the histogram plot. Defaults to "k".
            label (str, optional): Label for the plot legend. Defaults to None.
            num_bins (int, optional): Number of bins for the histogram. Defaults to 50.
        """
        basis = make_basis(self.sampler.n)
        num_cs = len(self.unique_causal_bitstrings)
        
        # Calculate BD action for each causal structure
        BD_actions = np.zeros(num_cs)
        for i, s in enumerate(self.unique_causal_bitstrings):
            mat = np.zeros((self.sampler.n, self.sampler.n), dtype=np.int32)
            for count, pair in enumerate(basis):
                mat[pair[0], pair[1]] = int(s[count])
            BD_actions[i] = calculate_action(mat, epsilon=self.sampler.epsilon)
        
        # Calculate the matrix of absolute BD action differences between all pairs of states
        BD_action_differences = np.abs(BD_actions[:, np.newaxis] - BD_actions)
        
        # Create bins for the histogram
        bins = np.linspace(0, np.max(BD_action_differences) * 1.001, num_bins)
        
        # Digitize the differences to assign each to a bin
        bin_indexes = np.digitize(BD_action_differences, bins) - 1
        
        # Sum the proposal probabilities (from Q) for each bin
        values = np.zeros(len(bins) - 1)
        for i in range(num_cs):
            for j in range(num_cs):
                bin_index = bin_indexes[i, j]
                if 0 <= bin_index < len(values):
                    values[bin_index] += Q[i, j]

        # Plot the resulting histogram
        plt.stairs(values, bins, alpha=0.7, color=color, label=label)
        plt.title("Histogram of Proposals by BD Action Difference")
        plt.xlabel("BD Action Difference")
        plt.ylabel("Total Proposal Probability")

