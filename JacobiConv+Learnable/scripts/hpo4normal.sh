set -e

#datasets=("cora" "citeseer" "pubmed" "chameleon" "film" "squirrel" "cornell" "texas")
datasets=("cora" "citeseer" "pubmed" "computers" "photo" "chameleon" "film" "squirrel")

for (( i=0; i<${#datasets[@]}; i++ ))
do
  CUDA_VISIBLE_DEVICES="${i}" python RealWorld.py --repeat 3 --optruns 400 --split dense --dataset ${datasets[$i]} --normalized_bases --name ${datasets[$i]} >exp_on_${datasets[$i]}.out 2>exp_on_${datasets[$i]}.err &
done

#for (( i=0; i<${#datasets[@]}; i++ ))
#do
#  CUDA_VISIBLE_DEVICES="${i}" python RealWorld.py --test --repeat 10 --split dense --dataset ${datasets[$i]} --normalized_bases --path "results" --name "exp_on_${datasets[$i]}.out" >test_on_${datasets[$i]}.out 2>test_on_${datasets[$i]}.err &
#done
