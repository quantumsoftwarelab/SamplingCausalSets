# Code to brute force sample Omega, by generating all possible upper triangular binary matrices of size n x n, and checking if they correspond to link/causal matrices.


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product


class naive_sampling:
    
    def __init__(self, n):
        """
        Initialises the brute force sampling class.
        
        Parameters:
        n (int): The size of the causal set.
        """
        
        self.n = n
        

    
    
    def is_link_matrix(self, matrix):
        """
        Checks if a given upper triangular binary matrix corresponds to the link matrix of a causal set.
        
        Parameters:
        matrix (np.ndarray): The upper triangular binary matrix to check.
        
        Returns:
        bool: True if the matrix corresponds to a link matrix, False otherwise.
        """
    
        m = self.transitive_reduction(matrix)
        if np.all(m == matrix):
            return True
        else:
            return False

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

    def transitive_reduction(self, a):
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


    def transitive_closure(self,a):
        #go from a DAG adjacency matrix to a link matrix
        # copied from https://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
        n = len(a)
        m = np.copy(a)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    m[i][j] = m[i][j] or (m[i][k] and m[k][j])
        return m

    def get_unique_matrices(self):
        
        num_unique = 2**((self.n**2-self.n)//2)
        
        unique_matrices = set()
        for bits in product([0, 1], repeat=(self.n * (self.n - 1)) // 2):
            matrix = np.zeros((self.n, self.n), dtype=int)
            upper_tri_indices = np.triu_indices(self.n, 1)
            matrix[upper_tri_indices] = bits
            unique_matrices.add(matrix.tobytes())
            dtype_ = matrix.dtype

            
        print(f"Number of unique matrices: {len(unique_matrices)}")
        if len(unique_matrices) != num_unique:
            print("Something is wrong")



        unique_link_matrix = set()
        unique_causal_matrix = set()



        for unique_matrix in unique_matrices:
            matrix = np.fromstring(unique_matrix, dtype = dtype_).reshape(self.n,self.n)

            unique_matrices.add(matrix.tostring())
            
            if self.is_link_matrix(matrix):
                unique_link_matrix.add(matrix.tostring())

            if self.is_causal_matrix(matrix):
                unique_causal_matrix.add(matrix.tostring())


        return unique_matrices, unique_link_matrix, unique_causal_matrix
