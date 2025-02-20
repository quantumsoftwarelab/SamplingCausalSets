
# Code to naively sample Omega, by rejecting local moves that do not result in a link/causal matrix


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from collections import defaultdict
import seaborn as sns
import time

class Henson_sampling:
    
    def __init__(self, n):
        """
        Initialises the classical sampling class. Recreates the algorithm of Henson in the "onset of the asymptotic regime" paper.
        
        Parameters:
        n (int): The size of the causal set.
        """
        
        self.n = n
        
        self.causal_matrix = self.generate_causal_matrix()
        
    def link_move(self):
        # pick two random elements i and j
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)
        
        # make sure i != j
        while i == j:
            j = np.random.randint(0, self.n)
            
        y = max(i,j)
        x = min(i,j)
        
        
        self.last_move = None
        
        if self.is_linked(x,y):
            #print("linked")
            # print("linked")
            #self.causal_matrix[x,y] = 0
            self.last_move = "Link: unlinked"
            
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
            
            link_matrix = self.transitive_reduction(self.causal_matrix)
            link_matrix[x,y] = 0
            self.causal_matrix = self.transitive_closure(link_matrix)

            
        elif self.is_suitable_pair(x,y):
            #print("suitable pair")
            self.last_move = "Link: Suitable pair"
            
            #self.causal_matrix[x,y] = 1 # Relate x and y
            
            link_matrix = self.transitive_reduction(self.causal_matrix)
            
            if link_matrix[x,y] != 0:
                print("suitable link error")   

            link_matrix[x,y] = 1
            
            
            self.causal_matrix = self.transitive_closure(link_matrix)
            
            
            
            link_matrix_new = self.transitive_reduction(self.causal_matrix)
            
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
            self.last_move = None#"Link: No change"
            pass
        
        
        
        
    def relation_move(self):
        # pick two random elements i and j
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)
        
        # make sure i != j
        while i == j:
            j = np.random.randint(0, self.n)
            
        y = max(i,j)
        x = min(i,j)
        
        
        self.last_move = None
        
        if self.is_linked(x,y):
            #print("linked")
            self.causal_matrix[x,y] = 0
            self.last_move = "Relation: unlinked"
        elif self.is_critical_pair(x,y):
            #print("critical pair")
            self.causal_matrix[x,y] = 1
            self.last_move = "Relation: Critical pair"
        else:
            #print("not critical or linked pair")
            self.last_move = None#"Relation: No change"
            pass
            

    
    def is_linked(self, x, y):
        linked = False
        if self.causal_matrix[x,y] == 1: # If related
            sum = 0
            for k in range(x,y+1): 
                sum += self.causal_matrix[x,k]*self.causal_matrix[k,y]
            if sum == 0: # If there is no element in past(y) and fut(x), relation must be a link
                linked =  True
        return linked
    
    def is_critical_pair(self, x, y):
        
        for k in range(y+1, self.n):
            if self.causal_matrix[y,k] ==1: #k is fut(y)
                if self.causal_matrix[x,k] != 1: #k is not fut(x)
                    # If there is a k such that k is fut(y) but not fut(x), then x and y are not a critical pair
                    return False
        for k in range(0, x):
            if self.causal_matrix[k,x] ==1:
                if self.causal_matrix[k,y] != 1:
                    # If there is a k such that k is past(x) but not past(y), then x and y are not a critical pair
                    return False
        #print("causal matrix: ", self.causal_matrix)
        return True
    
        
    def is_suitable_pair(self, x,y):
        
        #print("Suitable pair check, C: ", self.causal_matrix)
        for z in range(0, x+1):#z is incpast(x)
            if self.causal_matrix[z,x] ==1 or z == x:
                #print("z: ", z)
                for w in range(y, self.n):#w is incfut(y)
                    if self.causal_matrix[y,w] ==1 or w == y:
                        #print("w: ", w)
                        if self.causal_matrix[z,w] ==1: 
                            #print("Not suitable pair, z, w: ", z,w)
                            # If there is a z in incpast(x) related to w incfut(y), then not suitable
                            return False
                
        return True

    def generate_causal_matrix(self):
        """
        Generates a random causal matrix of size n x n.
        Not uniform random(as this is very hard to do), but random in the sense that it is the transitive closure of a random upper triangular binary matrix.
        
        Returns:
        np.ndarray: A random causal matrix.
        """
        
        matrix = np.triu(np.random.randint(0, 2, size=(self.n, self.n)), 1)
        matrix = self.transitive_closure(matrix)
        return matrix
    
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
    """
    def transitive_reduction(self, a):
        n = len(a)
        m = np.copy(a)

        for j in range(n):
            for i in range(0,j):
                
                
                #if i and j are connected
                if m[i,j]:
                    
                    for k in range(n):
                        
                        #if j and k are connected
                        if m[i,k] and m[k,j]:
                            
                            # unconnect i and k
                            m[i,j] = 0
        return m
    """
    
    def transitive_reduction(self,a):
        
        # need to first close it... ugh
        m = self.transitive_closure(a)
        n = a.shape[0]
        for j in range(n):
            for i in range(n):
                if (m[i][j]):
                    for k in range(n):
                        if (m[j][k]):
                            m[i][k] = 0
        return m
    
    """

    def transitive_reduction(self,a):
        n = a.shape[0]
        reduced_matrix = a.copy()

        for k in range(n):
            for i in range(k):
                for j in range(k + 1, n):
                    # If there's a path i -> k -> j, remove the direct edge i -> j
                    if reduced_matrix[i, k] and reduced_matrix[k, j]:
                        reduced_matrix[i, j] = 0

        return reduced_matrix"""

    def transitive_closure(self,a):
        # copied from https://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
        n = len(a)
        m = np.copy(a)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    m[i][j] = m[i][j] or (m[i][k] and m[k][j])
        return m
    def num_relations(self, matrix):
        """
        Calculates the number of relations in a given causal matrix.
        
        Parameters:
        matrix (np.ndarray): The causal matrix to calculate the number of relations of.
        
        Returns:
        int: The number of relations in the causal matrix.
        """
        return np.sum(matrix)
    
    
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
            
        start_time = time.time()
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
                matrix_str = self.causal_matrix.tostring()
                unique_causal_matrices[matrix_str] += 1
        end_time = time.time()
        #print("Time taken: ", end_time - start_time)
        print("Time per step: ", (end_time - start_time)/steps)
        print("acceptance rate: ", acceptance/steps)
        return unique_causal_matrices
            



def test_transitive_functions(m_1 = None):
    
    if m_1 is None:
        sampler = Henson_sampling(3)
        m_1 = np.array([[0,0,1],[0,0,1],[0,0,0]])
    else:
        sampler = Henson_sampling(len(m_1))
        
        
    
    tc_m1 = sampler.transitive_closure(m_1)
    tr_m1 = sampler.transitive_reduction(m_1)
    
    tctr_m1 = sampler.transitive_closure(tc_m1)
    
    trtc_m1 = sampler.transitive_reduction(tc_m1)
    
    if not np.all(tc_m1 == tctr_m1):
        print("Error 1: tc != trtc ")
    if not np.all(tr_m1 == trtc_m1):
        print("Error 2: tr != trtc ")    
        print("m1: ", m_1)
        print("tr: ", tr_m1)
        print("tc: ", tc_m1)
        print("trtc: ", trtc_m1)


"""
for i in range(100):
    test_transitive_functions(np.triu(np.random.randint(0, 2, size=(9,9)), 1))"""





n = 4
num_samples = 100
sample_frequency = 2* n#n**3
T_therm = sample_frequency *2


sampler = Henson_sampling(n)

# Define the combinations of link_move and relation_move
combinations = [(True, False), (False, True), (True, True)]
colors = ['r', 'g', 'b', 'y']
labels = ['Link Only', 'Relation Only', 'Both']

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10, 6))

for (link_move, relation_move), color, label in zip(combinations, colors, labels):
    uniques = sampler.sample(num_samples = num_samples, sample_frequency=sample_frequency, T_therm = T_therm, link_move=link_move, relation_move=relation_move)
    matrix_labels = list(uniques.keys())
    
    heights = []
    num_relations = []
    for string in matrix_labels:
        matrix = np.frombuffer(string, dtype=np.int32).reshape(n, n)
        heights.append(sampler.height(matrix))
        num_relations.append(sampler.num_relations(matrix))
    
    # For num_relations
    counts_i = np.zeros((n*(n-1))//2+1)
    for num_relation in num_relations:
        counts_i[num_relation] += 1
    counts_i = np.array(counts_i)
    T = np.sum(counts_i)
    freq = counts_i/T
    error = np.array([np.sqrt((l * (1-l))/(T-1)) for l in freq])  #√ f (1 − f )/(T − 1
    ax2.errorbar(np.arange(0,len(counts_i),1), freq, yerr = error, color=color, linewidth = 0, elinewidth = 1,marker = "o",markersize = 3, label=label)
    
    # For  heights
    counts_i = np.zeros(n+1)
    for height in heights:
        counts_i[height] += 1
    counts_i = np.array(counts_i)
    T = np.sum(counts_i)
    freq = counts_i/T
    error = np.array([np.sqrt((l * (1-l))/(T-1)) for l in freq])  #√ f (1 − f )/(T − 1
    ax1.errorbar(np.arange(0,len(counts_i),1), freq, yerr = error, color=color, linewidth = 0, elinewidth = 1,marker = "o",markersize = 3, label=label)
    
    
    

exact_data_height = np.array([[1, 8.57e-9], [2, 0.051], [3, 0.61], [4, 0.40], 
                    [5, 0.05], [6, 0.0031], [7, 0.000090], [8, 0.0000013]])

if n ==9:
    ax1.plot(exact_data_height[:,0], exact_data_height[:,1], color = "k", linewidth = 0, marker = "o", label='Exact Data')


ax1.set_yscale('log')
ax1.set_xlabel('Height')
ax1.set_ylabel('Frequency (Normalized)')
ax1.set_title('Scatter Plot of Heights of Unique Causal Matrices')
#ax1.legend()


ax2.set_yscale('log')
ax2.set_xlabel('Number of relations')
ax2.set_ylabel('Frequency (Normalized)')
ax2.set_title('Scatter Plot of Heights of Unique Causal Matrices')

plt.show()


"""
# Create a violin plot with each integer having its own violin
plt.figure(figsize=(10, 6))
sns.violinplot(x =hamming_weights, y = matrix_counts)
plt.xlabel('Hamming Weight')
plt.ylabel('Density')
plt.title('Violin Plot of Hamming Weights')
plt.show()
"""