
import numpy as np
from itertools import product
import math
from typing import List, Tuple, Union, Dict
import numpy.typing as npt

def calc_interval_abundances(causal_matrix: npt.NDArray ) -> npt.NDArray[np.int32]:
    """
    Calculate interval abundances based on a given causal matrix.
    This function computes the relative abundances of intervals in a causal matrix.
    
    Args:
        causal_matrix (numpy.ndarray): A square matrix representing causal relationships.
    
    Returns:
        numpy.ndarray: An array of relative abundances of intervals, the first element
                    is the cardinality of the causal set, and subsequent elements 
                    represent the number of IOIs (see cunningham 2018)
    """

    n = causal_matrix.shape[0]

    adj_mat = causal_matrix
    past_mat = causal_matrix.T

    rel_abundances = np.zeros(n+1)
    # Loop through all pairs
    for i, val_i in enumerate(adj_mat):
        for j, val_j in enumerate(adj_mat):
            # trivial way to find cardinality
            if i == j:
                rel_abundances[0] += 1
            
            # if i is in the past of j, count the number of elements in interval
            elif adj_mat[i,j] ==1:
                #number of matching 1's in row of adj matrix
                #from Naive action algorithm in cunningham 2018
                #only want to search in the future of j... so only top (or bottom depending on setup) half of matrix
                k = len(np.where((adj_mat[i,:]==past_mat[j,:])&(adj_mat[i,:] ==1))[0])
                rel_abundances[k+1] +=1
    # Return as integer array
    return rel_abundances.astype(np.int32)

def calculate_action(causal_matrix: npt.NDArray, smeared: bool = True, stdim: int = 2, epsilon: float = 0.1, first_order_smearing: bool = False, first_order_taylor: bool = False):
    """
        Calculate the action based on the given causal matrix and parameters.
        
        Args:
            causal_matrix (numpy.ndarray): The causal matrix used to calculate interval abundances.
            smeared (bool, optional): Whether to apply smearing. Default is True.
            stdim (int, optional): The spacetime dimension. Default is 2. (other dimensions not yet implimented)
            epsilon (float, optional): The epsilon parameter for smearing. Default is 0.1.
            first_order_smearing (bool, optional): Whether to use first-order smearing. Default is False.
            first_order_taylor (bool, optional): Whether to use first-order Taylor expansion. Default is False. Infers first_order_smearing=True.
        
        Returns:
            float: The calculated action.
        """
    
    if stdim != 2:
        raise NotImplementedError("Only 2D is currently implemented")
    
    if first_order_taylor:
        if first_order_smearing == False:
            print("Assuming first_order_smearing=True due to first_order_taylor=True")
            first_order_smearing = True
    
    
    c = calc_interval_abundances(causal_matrix)
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
            
    return a



def transitive_closure(a:npt.NDArray) -> npt.NDArray:
    """
    Computes the transitive closure of a given adjacency matrix using Warshall's algorithm.
    The transitive closure of a graph is a reachability matrix that indicates whether there is a path 
    between any pair of vertices in the graph.
    Taken from https://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
    Parameters:
    a (numpy.ndarray): A square adjacency matrix representing the graph. The matrix should be a 2D numpy array 
        where a[i][j] is True if there is an edge from vertex i to vertex j, and False otherwise.
    Returns:
    numpy.ndarray: A square matrix of the same size as the input matrix, where the element at position (i, j) 
        is True if there is a path from vertex i to vertex j, and False otherwise.
    """
    
    
    
    
    n = len(a)
    m = np.copy(a)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                m[i][j] = m[i][j] or (m[i][k] and m[k][j])
    return m


def is_causal_matrix(matrix:npt.NDArray) -> bool:
    """
    Checks if a given upper triangular binary matrix corresponds to the causal matrix of a causal set.
    
    Parameters:
    matrix (np.ndarray): The upper triangular binary matrix to check.
    
    Returns:
    bool: True if the matrix corresponds to a causal matrix, False otherwise.
    """
    
    m = transitive_closure(matrix)
    if np.all(m == matrix):
        return True
    else:
        return False

def get_unique_matrices(n:int) -> Tuple[set, set]:
    """
    Generate unique matrices and unique causal matrices of size n x n.
    This function generates all possible unique upper triangular binary matrices 
    of size n x n, and then filters out those that are causal matrices.
    
    Parameters:
    n (int): The size of the matrices to generate.
    
    Returns:
    Tuple[set, set]: A tuple containing two sets:
        - The first set contains all unique upper triangular binary matrices.
        - The second set contains all unique causal matrices.
    """
    
    
    num_unique = 2**((n**2-n)//2)
    
    unique_matrices = set()
    for bits in product([0, 1], repeat=(n * (n - 1)) // 2):
        matrix = np.zeros((n, n), dtype = np.int32)
        upper_tri_indices = np.triu_indices(n, 1)
        matrix[upper_tri_indices] = bits
        unique_matrices.add(matrix.tobytes())
        dtype_ = matrix.dtype

        
    #print(f"Number of unique matrices: {len(unique_matrices)}")
    if len(unique_matrices) != num_unique:
        raise ValueError(f"Number of unique matrices is not correct. Expected {num_unique}, got {len(unique_matrices)}")


    unique_causal_matrix = set()



    for unique_matrix in unique_matrices:
        matrix = np.frombuffer(unique_matrix, dtype = dtype_).reshape(n,n)

        unique_matrices.add(matrix.tobytes())
        
    

        if is_causal_matrix(matrix):
            unique_causal_matrix.add(matrix.tobytes())


    unique_matrices = unique_matrices
    unique_causal_matrix = unique_causal_matrix
    
    return unique_matrices, unique_causal_matrix






def get_upper_triangular_basis(n: int) -> npt.NDArray:
    """
    
    Generate a basis for the upper triangular part of an n x n matrix.
    Parameters:
        n (int): The cardinality
    
    Returns:
        np.ndarray: The map between the i and j coordinates of the 
            upper triangular part of the matrix and the qubit labelling
    
    """
    
    
    
    q = np.zeros((n,n), dtype = int)
    
    count = int(0)
    for i in range(n):
        for j in range(i+1, n):
            q[i,j] = count
            count += 1
    return q


def num_relations(matrix: npt.NDArray) -> int:
    """
    Calculates the number of relations in a given causal matrix.
    
    Parameters:
    matrix (np.ndarray): The causal matrix to calculate the number of relations of.
    
    Returns:
    int: The number of relations in the causal matrix.
    """
    return np.sum(matrix)


def height(matrix: npt.NDArray) -> int:
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
    #+1 as height is the number of nodes, not relations
    
    return longest_chain

def ordering_fraction(matrix: npt.NDArray) -> float:
    """
    Calculate the ordering fraction of a given matrix.
    The ordering fraction r is the fraction of pairs of elements which are related.
    It is computed as the number of relations (R) divided by the total number of possible pairs (N choose 2).
    
    Parameters:
    matrix (npt.NDArray): A square matrix representing the relations between elements.
    
    Returns:
    float: The ordering fraction of the matrix.
    """
    
    
    
    # The ordering fraction r is the fraction of pairs of elements
    # which are related
    
    #R/(N choose 2) # R is the number of relations
    R = np.sum(matrix)
    n = len(matrix)
    
    return R/((n*(n-1))/2)

def minimal_elements(matrix: npt.NDArray) -> int:
    """
    Counts the number of minimal elements in the causal set.
    A minimal element is defined as an element with no incoming relations,
    i.e., there are no other elements that precede it in the causal set.
    
    Parameters:
    matrix (npt.NDArray): A square adjacency matrix representing the causal set,
        where matrix[i, j] is non-zero if there is a relation from element i to element j.
        
    Returns:
        int: The number of minimal elements in the causal set.
    """
    
    
    
    # Counts the number of minimal elements in the causal set 
    # (elements with no incoming relations)
    
    n = len(matrix)
    minimal_elements = 0
    for i in range(n):
        if np.sum(matrix[:,i]) == 0:
            minimal_elements += 1
    return minimal_elements
