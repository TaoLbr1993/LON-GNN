dataset=$1
cuda=$2

fix_args="--storage '' --coef_upd pcd --emb mlp"

python train_model.py --model StdJacobiSGNNS --dataset $dataset --param_search --gpu_id $cuda $fix_args > search_StdJacobiSGNNS_on_${dataset}.out 2>search_StdJacobiSGNNS_on_${dataset}.err