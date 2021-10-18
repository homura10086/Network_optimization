import pandas as pd
import numpy as np

a = [i for i in range(10)]
b = [i for i in range(10)]
c = np.array((a, b)).transpose()
print(c.shape)
