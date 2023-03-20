set -e

dataset=$1

# ss0
wds=(0.00005 0.0001 0.0005 0.001)
lrs=(0.001 0.01)

# ss1
#wds=(0.00005 0.0005)
#lrs=(0.0005 0.001 0.01 0.05)

#for (( i=0; i<${#wds[@]}; i++ ))
#do
#  for (( j=0; j<${#lrs[@]}; j++ ))
#  do
#    k=$((2*$i+$j))
#    echo $k
#    CUDA_VISIBLE_DEVICES="${k}" python RealWorld.py --repeat 3 --optruns 400 --split dense --dataset $dataset --learnable_bases --wd4 ${wds[$i]} --lr4 ${lrs[$j]} --name $dataset >exp${k}_on_${dataset}.out 2>exp${k}_on_${dataset}.err &
#  done
#done

for (( i=0; i<${#wds[@]}; i++ ))
do
  for (( j=0; j<${#lrs[@]}; j++ ))
  do
    k=$((2*$i+$j))
    echo $k
    CUDA_VISIBLE_DEVICES="${k}" python RealWorld.py --test --repeat 10 --split dense --dataset $dataset --learnable_bases --wd4 ${wds[$i]} --lr4 ${lrs[$j]} --path "results" --name "exp${k}_on_${dataset}.out" >test${k}_on_${dataset}.out 2>test${k}_on_${dataset}.err &
  done
done
