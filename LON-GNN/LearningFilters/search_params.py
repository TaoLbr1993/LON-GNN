import numpy as np

SEARCH_PARAMS = {
    'lr': [0.001, 0.005, 0.01, 0.05],
    'wd': [0.0, 1e-4, 5e-4, 1e-3],
    'lr_ab': [0.001, 0.005, 0.01, 0.05],
    'alpha': np.arange(0.5, 2.5, 0.5),
    # 'jacob_a': np.arange(-1.1, 0.05, 0.005),
    # 'jacob_b': np.arange(-0.2, 3.5, 0.05)
}