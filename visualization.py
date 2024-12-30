from math import comb

# Calculate and print the expression for each x from 0 to 32
print("x\tValue")
for x in range(33):
    value = comb(32, x)
    print(f"{x}\t{value}")

