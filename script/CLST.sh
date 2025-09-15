eval "$(conda shell.bash hook)"
conda activate omniclst
name=CLST_15k

timestamp=$(date +%Y%m%d-%H%M%S)
number_gpu=4
echo "timestamp: $timestamp"
echo "number_gpu: $number_gpu"

bash grpo.sh $name ../conf/$name.yaml $timestamp $number_gpu 32721


# infer
checkpoint_dir=""
latest_ckpt=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null \
  | sed 's/.*checkpoint-\([0-9]*\)$/\1 checkpoint-\1/' \
  | sort -n \
  | tail -n1 \
  | awk '{print $2}' \
  | xargs -I{} echo "${checkpoint_dir}/{}")
echo "Latest checkpoint: $latest_ckpt"
bash infer.sh $latest_ckpt