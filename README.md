# LON-GNN: Spectral GNNs with Learnable Orthonormal Basis

Source code for "LON-GNN: Spectral GNNs with Learnable Orthonormal Basis".

## Requirements
+ Python 3.7
+ numpy 1.21
+ pytorch 1.12
+ pyg 1.7.1
+ optuna 3.0.5

## Experiments on Fitting Images
Codes on fitting images can be found in ```LON-GNN/LearningFilters/```.
To reproduce the results of LON-GNN, run the following commands:

```
cd LON-GNN/LearningFilters/
./test_LONGNN.sh
```

The detailed parameters can also be found in ```LON-GNN/LearningFilters/test_LONGNN.sh```.

## Experiments on Real-World Datasets

**Reproduce LON-GNN**

To reproduce the results of LON-GNN on real-world datasets, run the following commands:
```
cd LON-GNN/
./test_LONGNN.sh
```

The detailed parameters can also be found in ```LON-GNN/test_LONGNN.sh```.

Users can also utilize optuna to search the best parameters:
```
cd LON-GNN/
./param_search.sh ${dataset} 0
```
where ```${dataset}``` can be selected from cora, pubmed, citeseer, computers, photo, actor, chameleon, squirrel, cornell and texas.

**Ablation**

To reproduce Jacobi+Orthnorm in ablation analysis, run the following commands:
```
cd LON-GNN/
./test_Jacobi+Orthnorm.sh
```

To reproduce Jacobi+Learnable in ablation analysis, 
run the following commands:
```
cd Jacobi+Learnable
./test_Jacobi+Learnable.sh
```




