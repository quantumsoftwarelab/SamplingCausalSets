
import numpy as np
from itertools import product
import math
from typing import List, Tuple, Union, Dict
import os
import pickle
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
    
    #if stdim != 2:
    #    raise NotImplementedError("Only 2D is currently implemented")
    
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


def transitive_reduction(a):
    
    # need to first close it... ugh
    m = transitive_closure(a)
    n = a.shape[0]
    for j in range(n):
        for i in range(n):
            if (m[i][j]):
                for k in range(n):
                    if (m[j][k]):
                        m[i][k] = 0
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
    
    
    # Create the directory if it doesn't exist
    filepath = os.getcwd()
    save_folder = "save_files"
    save_path = os.path.join(filepath, save_folder)
    
    
    try:
        unique_matrices = pickle.load(open(os.path.join(save_path, f"unique_matrices_"+str(n)+".pkl"), "rb"))
        unique_causal_matrix = pickle.load(open(os.path.join(save_path, f"unique_causal_matrices_"+str(n)+".pkl"), "rb"))
    except:
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
        
        

        with open(os.path.join(save_path, f"unique_matrices_"+str(n)+".pkl"), "wb") as f:
            pickle.dump(unique_matrices, f)

        with open(os.path.join(save_path, f"unique_causal_matrices_"+str(n)+".pkl"), "wb") as f:
            pickle.dump(unique_causal_matrix, f)
        
    unique_matrices = sorted(unique_matrices, key=lambda x: np.frombuffer(x, dtype=np.int32).reshape(n, n).tolist())
    unique_causal_matrix = sorted(unique_causal_matrix, key=lambda x: np.frombuffer(x, dtype=np.int32).reshape(n, n).tolist())
    
    #print("Unique matrices: ", unique_matrices)
    #print("Unique causal matrices: ", unique_causal_matrix)
    return unique_matrices, unique_causal_matrix

def get_unique_causal_bitstrings(n:int) -> Tuple[npt.NDArray,npt.NDArray]:
    
    unique_matrices, unique_causal_matrices = get_unique_matrices(n)
    
    # Convert unique causal matrices into bitstring representations
    unique_bitstring_causal_matrices = []
    for string in unique_causal_matrices:
        matrix = np.frombuffer(string, dtype=np.int32).reshape((n, n))
        bitstring = ''.join(str(int(matrix[i, j])) for i in range(n) for j in range(i + 1, n))
        unique_bitstring_causal_matrices.append(bitstring)

    # Convert to a numpy array
    unique_bitstring_causal_matrices = np.array(unique_bitstring_causal_matrices)
    
    unique_bitstring_matrices = []
    # Convert unique causal matrices into bitstring representations
    for string in unique_matrices:
        matrix = np.frombuffer(string, dtype=np.int32).reshape((n, n))
        bitstring = ''.join(str(int(matrix[i, j])) for i in range(n) for j in range(i + 1, n))
        unique_bitstring_matrices.append(bitstring)
    
    unique_bitstring_matrices = np.array(unique_bitstring_matrices)
    
    
        # Sort full list of matrices
    arg_sorted = np.array(np.argsort(list(unique_bitstring_matrices), axis=0), dtype=int)
    unique_bitstring_matrices = np.array(list(unique_bitstring_matrices))[arg_sorted]

    # Sort causal matrices
    arg_sorted_causal = np.array(np.argsort(list(unique_bitstring_causal_matrices), axis=0), dtype=int)
    unique_bitstring_causal_matrices = np.array(list(unique_bitstring_causal_matrices))[arg_sorted_causal]
    
    return unique_bitstring_matrices, unique_bitstring_causal_matrices




def calculate_average_action(cardinality:int, causal_matrices: set, stdim: int = 2, epsilon: float = 0.1, Temp: float = 1) -> float:
    """
    Calculate the average action over a set of causal matrices for a particular temperature.
    
    Parameters:
        causal_matrices (set): A set of unique causal matrices.
        smeared (bool, optional): Whether to apply smearing. Default is True.
        stdim (int, optional): The spacetime dimension. Default is 2.
        epsilon (float, optional): The epsilon parameter for smearing. Default is 0.1.
    """
    save_path = os.path.join(os.path.dirname(os.getcwd()), "save_files")
    str_temp = str(Temp).replace(".", "_") 
    
    
    
    try:
        average_action = np.load(os.path.join(save_path, f"average_action_{cardinality}_"+str_temp)+".npy")
        return average_action
    except:
        pass
    causal_bitstrings = causal_matrices.copy()
    causal_matrices = []
    for bitstring in causal_bitstrings:
        matrix = np.zeros((cardinality,cardinality), dtype = np.int32)
        upper_tri_indices = np.triu_indices(cardinality, 1)
        for i in range(len(upper_tri_indices[0])):
            matrix[upper_tri_indices[0][i],upper_tri_indices[1][i]] = bitstring[i]
        causal_matrices.append(matrix)
    
    partition_function = calculate_BD_partition_function(cardinality, causal_matrices, stdim = stdim, epsilon = epsilon, Temp = Temp)
    average_action = 0
    for i, matrix in enumerate(causal_matrices):
        bitstring = causal_bitstrings[i]
        if not is_causal_matrix(matrix):
            raise ValueError("Matrix is not a causal matrix")
        action = calculate_action(matrix, stdim = stdim, epsilon = epsilon)
        mu = calculate_mu(action, partition_function, Temp = Temp)

        average_action += action * mu
    np.save(os.path.join(save_path, f"average_action_{cardinality}_"+str_temp+".npy"), average_action)
    return average_action

def calculate_mu(action, partition_function, Temp):
    return np.exp(-action/Temp)/partition_function
    
def calculate_BD_partition_function(cardinality: int, causal_matrices: set, stdim: int = 2, epsilon: float = 0.1, Temp: float = 1) -> float:
    save_path = os.path.join(os.path.dirname(os.getcwd()), "save_files")
    str_temp = str(Temp).replace(".", "_") 
    q = cardinality*(cardinality-1)//2
    try:
        partition_function = np.load(os.path.join(save_path, f"partition_function_{cardinality}_"+str_temp)+".npy")
        return partition_function
    except:   
        partition_function = 0
        for matrix in causal_matrices:
            
            
            action = calculate_action(matrix, stdim = stdim, epsilon = epsilon)
            partition_function += np.exp(-action/Temp)
        
        np.save(os.path.join(save_path, f"partition_function_{cardinality}_"+str_temp+".npy"), partition_function)
        return partition_function




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


def is_critical_pair(x, y, s_mat):
    n = len(s_mat)
    for k in range(y+1, n):
        if s_mat[y,k] ==1: #k is fut(y)
            if s_mat[x,k] != 1: #k is not fut(x)
                # If there is a k such that k is fut(y) but not fut(x), then x and y are not a critical pair
                return False
    for k in range(0, x):
        if s_mat[k,x] ==1:
            if s_mat[k,y] != 1:
                # If there is a k such that k is past(x) but not past(y), then x and y are not a critical pair
                return False
    #print("causal matrix: ", self.causal_matrix)
    return True

    
def is_suitable_pair(x,y, s_mat):
    
    if s_mat[x,y] == 1:
        # If x and y are related, then not a suitable pair
        return False
    
    n= len(s_mat)
    #print("Suitable pair check, C: ", self.causal_matrix)
    for z in range(0, x+1):#z is incpast(x)
        if s_mat[z,x] ==1 or z == x:
            #print("z: ", z)
            for w in range(y, n):#w is incfut(y)
                if s_mat[y,w] ==1 or w == y:
                    #print("w: ", w)
                    if s_mat[z,w] ==1: 
                        #print("Not suitable pair, z, w: ", z,w)
                        # If there is a z in incpast(x) related to w incfut(y), then not suitable
                        return False
            
    return True


def make_basis(n:int) -> npt.NDArray:
    basis = [(j, k) for j in range(n) for k in range(j+1, n)]
    return basis

def is_incpast(a,x,s_mat):
    # If a is in the inclusive past of x, then return True
    # Else, return false
    if s_mat[a,x] == 1 or a == x:
        return True
    else:
        return False

def is_incfut(b,y,s_mat):
    # If a is in the inclusive future of x, then return True
    # Else, return false
    
    if s_mat[y,b] == 1 or b == y:
        return True
    else:
        return False
    
def is_linked(x, y, s_mat):
    linked = False
    if s_mat[x,y] == 1: # If related
        #for all points k
        for k in range(x,y+1): 
            if s_mat[x,k] == 1 and s_mat[k,y] == 1:
                # If there is a k such that k is past(x) and fut(y), then not a link
                return False
        linked = True # If urelated and nothing in between, then linked
    else: # If unrelated
        linked = False
    return linked