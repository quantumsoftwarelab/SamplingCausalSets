# Code to explore the space of all causal sets, Omega


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product


# Questions:

# What fraction of Link matrices correspond to actual causal sets?
# Same question for causal matrices

# What is the overcounting? (i.e. how many different Link matrices correspond to the same causal set?)



def random_upper_triangular_binary_matrix(n):
    """
    Generates a random upper triangular binary matrix of size n x n.
    
    Parameters:
    n (int): The size of the matrix.
    
    Returns:
    np.ndarray: A random upper triangular binary matrix.
    """
    matrix = np.triu(np.random.randint(0, 2, size=(n, n)), 1)
    return matrix

def is_link_matrix(matrix):
    """
    Checks if a given upper triangular binary matrix corresponds to the link matrix of a causal set.
    
    Parameters:
    matrix (np.ndarray): The upper triangular binary matrix to check.
    
    Returns:
    bool: True if the matrix corresponds to a link matrix, False otherwise.
    """
    
    m = transitive_reduction(matrix)
    if np.all(m == matrix):
        return True
    else:
        return False

def is_causal_matrix(matrix):
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

def transitive_reduction(a):
    """ Transforms a given directed acyclic graph into its minimal equivalent """
    n = len(a)
    m = np.copy(a)

    for i in range(n):
        for j in range(i+1,n):
            
            
            #if i and j are connected
            if m[i,j]:
                
                for k in range(n):
                    #if j and k are connected
                    if m[j,k] and m[i,k]:
                        # unconnect i and k
                        m[i,j] = 0
    return m


def transitive_closure(a):
    #go from a DAG adjacency matrix to a link matrix
    # copied from https://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
    n = len(a)
    m = np.copy(a)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                m[i][j] = m[i][j] or (m[i][k] and m[k][j])
    return m

n_list = [2,3,4,5,6]
num_uniques = []
num_L_list = []
num_C_list = []

for n in n_list:
    

    num_unique = 2**((n**2-n)//2)
    num_uniques.append(num_unique)
    
    unique_matrices = set()
    for bits in product([0, 1], repeat=(n * (n - 1)) // 2):
        matrix = np.zeros((n, n), dtype=int)
        upper_tri_indices = np.triu_indices(n, 1)
        matrix[upper_tri_indices] = bits
        unique_matrices.add(matrix.tobytes())
        dtype_ = matrix.dtype

        
    print(f"Number of unique matrices: {len(unique_matrices)}")
    if len(unique_matrices) != num_unique:
        print("Something is wrong")



    unique_link_matrix = set()
    unique_causal_matrix = set()



    for unique_matrix in unique_matrices:
        matrix = np.fromstring(unique_matrix, dtype = dtype_).reshape(n,n)
        
        TC_matrix = transitive_closure(matrix)
        TR_matrix = transitive_reduction(matrix)
        TC_of_TR_matrix = transitive_closure(TR_matrix)
        
        if np.all(TC_matrix != TC_of_TR_matrix):
            print("something is wronggg")
        
        unique_matrices.add(matrix.tostring())
        
        if is_link_matrix(matrix):
            unique_link_matrix.add(matrix.tostring())

        if is_causal_matrix(matrix):
            unique_causal_matrix.add(matrix.tostring())


    """
    print("Unique Causal Matrices")
    for m in unique_causal_matrix:
        print((np.fromstring(m, dtype = np.int32)).reshape(n,n))

    print("Unique Link Matrices")
    for m in unique_link_matrix:
        print((np.fromstring(m, dtype = np.int32)).reshape(n,n))"""

    print(f"Number of unique matrices found: {len(unique_matrices)}")
    print(f"Number of unique link matrices found: {len(unique_link_matrix)}")
    print(f"Number of unique causal matrices found: {len(unique_causal_matrix)}")
    
    num_L_list.append(len(unique_link_matrix))
    num_C_list.append(len(unique_causal_matrix))



percent_C_list = [num_C / num_unique * 100 for num_C, num_unique in zip(num_C_list, num_uniques)]
percent_L_list = [num_L / num_unique * 100 for num_L, num_unique in zip(num_L_list, num_uniques)]

plt.yscale("log")
#plt.plot(n_list, num_uniques, label="Number of unique matrices")
plt.plot(n_list, percent_C_list, label="Percent of unique causal matrices")
plt.plot(n_list, percent_L_list, label="Percent of unique link matrices")
plt.legend()
plt.show()

