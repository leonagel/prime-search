import numpy as np
import math as m

def count_distinct_permutations(perm_matrix):
    """
    Takes a permutation matrix and returns the number of distinct permutations
    reached through matrix exponentiation.
    
    Args:
        perm_matrix: numpy 2D array representing a permutation matrix
    
    Returns:
        int: number of distinct permutations reached through powers of the matrix
    """
    n = len(perm_matrix)
    seen = set()
    current = perm_matrix.copy()
    
    # Convert matrices to tuples for hashability
    def matrix_to_tuple(mat):
        return tuple(map(tuple, mat))
    
    # Add initial permutation
    seen.add(matrix_to_tuple(current))
    
    while True:
        # Compute next power
        current = current @ perm_matrix
        
        # Convert to tuple for comparison
        current_tuple = matrix_to_tuple(current)
        
        # If we've seen this permutation before, we've found all distinct ones
        if current_tuple in seen:
            break
            
        seen.add(current_tuple)
    
    return len(seen)

def count_distinct_random_permutations(A1, A2):
    """
    Takes two permutation matrices and counts distinct permutations reached by
    randomly multiplying either A1 or A2 at each step with equal probability.
    
    Args:
        A1: First permutation matrix (numpy 2D array)
        A2: Second permutation matrix (numpy 2D array)
    
    Returns:
        int: Number of distinct permutations reached
    """
    n = len(A1)
    seen = set()
    current = np.eye(n)  # Start with identity matrix
    
    # Convert matrices to tuples for hashability
    def matrix_to_tuple(mat):
        return tuple(map(tuple, mat))
    
    # Add initial permutation
    seen.add(matrix_to_tuple(current))

    n = np.random.randint(0, 4294967295)
    n_original = n
    m = np.random.randint(0, 4294967295)
    
    while True:
        # Randomly choose A1 or A2
        next_matrix = A1 if np.random.random() < 0.5 else A2
        
        # Compute next permutation
        current = current @ next_matrix
        
        # Convert to tuple for comparison
        current_tuple = matrix_to_tuple(current)
        
        # If we've seen this permutation before, check if we should continue
        if current_tuple in seen:
            # Run for a few more iterations to ensure we've found all permutations
            # Break if no new permutations found in last 100 iterations
            found_new = False
            for _ in range(100):
                next_matrix = A1 if (n // (2 ** 16)) % 2 == 0 else A2

                n = n * n_original + m
                m = ((m << 1) | (m >> 31)) & 0xFFFFFFFF

                current = current @ next_matrix
                current_tuple = matrix_to_tuple(current)
                if current_tuple not in seen:
                    seen.add(current_tuple)
                    found_new = True
            if not found_new:
                break
        
        seen.add(current_tuple)

        if len(seen) % 1000 == 0:
            print(len(seen))
    
    return len(seen)

def count_distinct_alternating_permutations(A1, A2):
    """
    Takes two permutation matrices and counts distinct permutations reached by
    alternating between A1 and A2 multiplication.
    
    Args:
        A1: First permutation matrix (numpy 2D array)
        A2: Second permutation matrix (numpy 2D array)
    
    Returns:
        int: Number of distinct permutations reached
    """
    n = len(A1)
    seen = set()
    current = np.eye(n)  # Start with identity matrix
    
    def matrix_to_tuple(mat):
        return tuple(map(tuple, mat))
    
    seen.add(matrix_to_tuple(current))
    use_A1 = True
    
    while True:
        # Alternate between A1 and A2
        next_matrix = A1 if use_A1 else A2
        use_A1 = not use_A1  # Switch for next iteration
        
        current = current @ next_matrix
        current_tuple = matrix_to_tuple(current)
        
        if current_tuple in seen:
            # Run for two more complete cycles to ensure we've found everything
            found_new = False
            for _ in range(4):  # 4 steps = 2 complete A1-A2 cycles
                next_matrix = A1 if use_A1 else A2
                use_A1 = not use_A1
                current = current @ next_matrix
                current_tuple = matrix_to_tuple(current)
                if current_tuple not in seen:
                    seen.add(current_tuple)
                    found_new = True
            if not found_new:
                break
        
        seen.add(current_tuple)
        if len(seen) % 1000 == 0:
            print(len(seen))
    
    return len(seen)

def generate_permutation_matrix(n):
    """
    Generates a random n×n permutation matrix.
    
    Args:
        n: size of the matrix
    
    Returns:
        numpy.ndarray: An n×n permutation matrix
    """
    # Create identity matrix
    perm = np.eye(n)
    # Get random permutation of indices
    indices = np.random.permutation(n)
    # Rearrange rows according to permutation
    return perm[indices]

# Example usage:
if __name__ == "__main__":
    # Test the generator
    n = 32
    random_perm1 = generate_permutation_matrix(n)
    random_perm2 = generate_permutation_matrix(n)
    print("\nRandom permutation matrix:")
    print(random_perm1)
    print(random_perm2)
    
    # print("\nTesting with random multiplication:")
    # result_random = count_distinct_random_permutations(random_perm1, random_perm2)
    # print(f"Random distinct permutations: {result_random}")
    # print(f"Random fraction covered: {result_random / m.factorial(n)}")
    
    print("\nTesting with alternating multiplication:")
    result_alt = count_distinct_random_permutations(random_perm1, random_perm2)
    print(f"Alternating distinct permutations: {result_alt}")
    print(f"Alternating fraction covered: {result_alt / m.factorial(n)}")

