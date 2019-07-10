from joblib import Parallel, delayed
Parallel(n_jobs=2, backend='loky')(delayed(print)('hello') for _ in range(10))
