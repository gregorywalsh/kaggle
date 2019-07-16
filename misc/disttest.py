from distributions import loguniform
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
a, b, base = 0.0001, 0.5, 10
sample_count = 1000
feature_count = 20
# dist = binom(sample_count - 1, (np.log(sample_count**2) / np.log(2)) / sample_count, 1)
dist = binom(feature_count - 1, 1/(feature_count**0.5 + 1), 1)
# dist = loguniform(a=a, b=b, base=base)
x = dist.rvs(size=100000)
# x = np.log(x) / np.log(base)
plt.hist(x)
plt.show()