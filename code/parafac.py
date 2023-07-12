import numpy as np
import tensorly as tl
import tensorly.decomposition
from icecream import ic
from tabulate import tabulate

np.random.seed(0)

n = 2
# Generate fake data
data = np.random.rand(2, 2, n)
# print(tabulate(data, tablefmt="latex", floatfmt=".2f"))

print("Random Data is :\n", data)
print()
# decompose the tensor using PARAFAC
# An Extension of PCA: Every component has a score and loading for each mode that it goes with
for i in range(1, n + 1):
    print(f"Rank : {i}")
    print()
    print(list(tl.decomposition.parafac(data, rank=i)))
    print()

# Invited in 1970 : https://en.wikipedia.org/wiki/Parafac
