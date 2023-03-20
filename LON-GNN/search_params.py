import numpy as np

SEARCH_PARAMS={
    # wd_mlp -> wd
    'GCN': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'dropout': np.arange(0.,0.9,0.1).tolist(),
    },
    'ChebNet': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'dropout': np.arange(0.,0.9,0.1).tolist(),
    },
    'APPNP': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'dropout': np.arange(0.,0.9,0.1).tolist(),
        # deduce from GPRGNN's settings
        'alpha': np.arange(0.1,0.9,0.1).tolist()
    },
    'GAT': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'dropout': np.arange(0.,0.9,0.1).tolist(),
        # need check
        'heads': [2,4,8],
        'output_heads': [2,4,8]
    },
    'GPRGNN': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'alpha': np.arange(0.,1.0,0.1).tolist(),
        'dropout': np.arange(0.,0.9,0.1).tolist(),
        'dpr_con': np.arange(0.,0.9,0.1).tolist()
    },
    'BernNet': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'dropout': np.arange(0.,0.9,0.1).tolist(),
        'dpr_con': np.arange(0.,0.9,0.1).tolist()
    },
    'JacobiSGNN': {
        'jacob_a': np.arange(-1,2,0.25).tolist(),
        'jacob_b': np.arange(-1,2,0.25).tolist(),
    },
    'JacobiSGNNS': {
        'jacob_a': np.arange(-1,2,0.25).tolist(),
        'jacob_b': np.arange(-1,2,0.25).tolist(),
    },
    'StdJacobiSGNN': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        "lr_lap": [0.0005,0.001,0.005,0.01,0.05],
        'lr_comb': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'wd_lap': [0.,5e-5,1e-4,5e-4,1e-3],
        'wd_comb': [0.,5e-5,1e-4,5e-4,1e-3],
        'alpha': np.arange(0.5,2,0.25).tolist(),
        'jacob_a': np.arange(-1,2,0.25).tolist(),
        'jacob_b': np.arange(-1,2,0.25).tolist(),
        'dropout': np.arange(0.,0.9,0.1).tolist(),
        'dpr_con': np.arange(0.,0.9,0.1).tolist()
    },
    'StdJacobiSGNNS': {
        'lr_mlp': [0.0005,0.001,0.005,0.01,0.05],
        "lr_lap": [0.0005,0.001,0.005,0.01,0.05],
        'lr_comb': [0.0005,0.001,0.005,0.01,0.05],
        'lr_ab': [0.0005,0.001,0.005,0.01,0.05],
        'wd_mlp': [0.,5e-5,1e-4,5e-4,1e-3],
        'wd_lap': [0.,5e-5,1e-4,5e-4,1e-3],
        'wd_comb': [0.,5e-5,1e-4,5e-4,1e-3],
        # 'wd_ab': [0.,5e-5,1e-4,5e-4,1e-3],
        'alpha': np.arange(0.5,2,0.25).tolist(),
        'jacob_a': np.arange(-1,2,0.25).tolist(),
        'jacob_b': np.arange(-1,2,0.25).tolist(),
        'dropout': np.arange(0.,0.9,0.1).tolist(),
        'dpr_con': np.arange(0.,0.9,0.1).tolist()
    },
}
