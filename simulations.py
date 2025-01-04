def get_first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for prime in primes:
            if prime * prime > num:
                break
            if num % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

# Get first n primes
number_of_primes = 100000
primes = get_first_n_primes(number_of_primes)

print(primes[-1])

exit()

# Check numbers in range
start = 10**14 - 10**8 // 10
end = 10**14
count = 0
from multiprocessing import Pool, cpu_count
from functools import partial

def check_number(num, primes):
    if num % 1000000 == 0:
        print("Reached: " + str(num))
    not_divisible = True
    for prime in primes:
        if prime * prime > num:  # Optimization - only need to check up to sqrt(num)
            break
        if num % prime == 0:
            not_divisible = False
            break
    return not_divisible

# Create a pool of worker processes
with Pool(cpu_count()) as pool:
    # Create partial function with fixed primes argument
    check_func = partial(check_number, primes=primes)
    
    # Map the function across the range in parallel
    results = pool.map(check_func, range(start, end + 1))
    
    # Sum up all True results to get final count
    count = sum(1 for x in results if x)

print(f"Numbers not divisible by first {number_of_primes} primes: {count}")


# Big conclusions from these tests:
# We ought to test against divisibility by the first 10,000 primes to take out ~96% of prime candidates
# or ~93% of odd prime candidates. 100,000 primes would likely yield diminishing returns for the search window.
# There is a power law thing here (i.e., every third number is divisible by 3, so they will filter out quickly)

# For search space of 1e8 numbers with ~3.1e6 primes
# Numbers indivisible by   1,000 primes (1e10): 6,278,040
# Numbers indivisible by   1,000 primes (1e12): 6,263,921
# Numbers indivisible by  10,000 primes (1e12): 4,810,151
# Numbers indivisible by  50,000 primes (1e12): 3,895,700
# Numbers indivisible by 100,000 primes (1e14): 3,894,570