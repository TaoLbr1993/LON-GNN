set -e

datasets=("low" "high" "band" "rejection" "comb")

for (( i=0; i<${#datasets[@]}; i++ ))
do
  CUDA_VISIBLE_DEVICES="$i" python ImgFilter.py --test --repeat 1 --dataset ${datasets[$i]} --fixalpha >on_${datasets[$i]}.out 2>on_${datasets[$i]}.err &
done
