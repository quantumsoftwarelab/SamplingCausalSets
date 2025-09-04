import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime
import scipy as sp
import joblib
from scipy.optimize import curve_fit
import pandas as pd
from typing import List
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))

from Sampler import Sampler
from helpers import make_basis, transitive_closure, get_unique_causal_bitstrings, transitive_reduction, is_causal_matrix, calculate_action, get_unique_matrices, transitive_closure, get_upper_triangular_basis, num_relations, height, ordering_fraction, minimal_elements, is_critical_pair, is_suitable_pair, is_linked,is_incpast, is_incfut

from spectralgap import SpectralGap

class SpectralGapAnalyser:
    """
    A class for analyzing spectral gaps and optimizing parameters for causal set samplers.

    This class provides methods to find optimal gamma parameters for quantum samplers,
    compare different sampling methods, and analyze the properties of proposal matrices.
    """
    
    def compare_gammas(self, cardinality: int, temp: float, methods: List[str], qc_args: List[dict],
                        epsilon: float = 0.1, uniform: bool = False, how_many_samples_if_sampling: int = 10,
                        dimension: int = 2) -> tuple[List[float], List[np.ndarray], List[float]]:
        """
        Compares the spectral gaps and move analyses for different quantum and classical sampler configurations.

        Args:
            cardinality (int): The number of elements in the causal set (n).
            temp (float): The temperature parameter for the acceptance matrix.
            methods (List[str]): A list of sampler methods (e.g., "quantum", "classical").
            qc_args (List[dict]): A list of dictionaries containing arguments specific to each sampler method.
            epsilon (float, optional): The epsilon parameter for the Benincasa-Dowker action. Defaults to 0.1.
            uniform (bool, optional): If True, uses a uniform acceptance matrix. Defaults to False.
            how_many_samples_if_sampling (int, optional): Number of samples for quantum proposal matrix estimation. Defaults to 10.
            dimension (int, optional): The spacetime dimension. Defaults to 2.

        Returns:
            tuple[List[float], List[np.ndarray], List[float]]:
                - deltas (List[float]): List of spectral gaps for each method.
                - analyses (List[np.ndarray]): List of move analyses (self, rejected, combined) for each method.
                - delta_errs (List[float]): List of spectral gap errors (from jackknife) for each method.
        """
        # Acceptance matrices are dependent on temperature.
        unique_bitstrings, unique_causal_bitstrings = get_unique_causal_bitstrings(cardinality)
        
        # Sort full list of bitstrings for consistent indexing.
        arg_sorted = np.array(np.argsort(list(unique_bitstrings)), dtype=int)
        unique_bitstrings = np.array(list(unique_bitstrings))[arg_sorted]
        
        # Sort causal bitstrings for consistent indexing.
        arg_sorted_causal = np.array(np.argsort(list(unique_causal_bitstrings)), dtype=int)
        unique_causal_bitstrings = np.array(list(unique_causal_bitstrings))[arg_sorted_causal]

        # Find the index of bitstrings in causal but not in full list
        causal_indices = [i for i, bs in enumerate(unique_bitstrings) if bs in unique_causal_bitstrings]
        
        directory = os.path.join(os.getcwd(), "save_files")
        A_mats_dir = os.path.join(directory, f"A_mats_{dimension}d")
            
        # Initialize a dummy sampler to get the acceptance matrix.
        dummy_sampler = Sampler(cardinality, method="quantum", qargs={"gammas":[0.1, (1-0.1)*0., (1-0.1)*(1-0.)], "t":10}, epsilon=epsilon, verbose=False, dimension=dimension)
        dummy_specgap = SpectralGap(dummy_sampler, temp, A_mats_dir)
        A = dummy_specgap.find_acceptance_matrix(uniform=uniform)
        
        deltas: List[float] = []
        delta_errs: List[float] = []
        analyses: List[np.ndarray] = []

        for i, method in enumerate(methods):
            if method == "quantum":
                sampler = Sampler(cardinality, method="quantum", qargs=qc_args[i], epsilon=epsilon, verbose=False, dimension=dimension)
                specgap = SpectralGap(sampler, temp, directory)
            elif method == "classical":
                sampler = Sampler(cardinality, method="classical", cargs=qc_args[i], epsilon=epsilon, verbose=False, dimension=dimension)
                specgap = SpectralGap(sampler, temp, directory)
            else:
                raise ValueError(f"Method '{method}' not recognised")
            
            if sampler.method == "quantum" and (sampler.is_t_a_range or sampler.gamma_ranges):
                Q_s = specgap.find_proposal_matrix(causal_indices, jackknife=True, how_many_samples_if_sampling=how_many_samples_if_sampling, save_matrix=True)
                delta, delta_err, analys = specgap.jackknife_spec_gaps(A, Q_s)
            else:
                Q = specgap.find_proposal_matrix(causal_indices, save_matrix=True)
                _, delta, analys = specgap.find_spec_gap(A, Q)
                delta_err = None # No error estimation for non-jackknife cases
                
            deltas.append(delta)
            delta_errs.append(delta_err)
            analyses.append(analys)
        return deltas, analyses, delta_errs
    

