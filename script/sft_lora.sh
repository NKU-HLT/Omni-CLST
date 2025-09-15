conda activate omniclst
nproc_per_node=4
name=Omni-CLST
timestamp=$(date +%Y%m%d_%H%M%S)
NPROC_PER_NODE=$nproc_per_node \
VIDEO_MAX_PIXELS=50178 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
MASTER_PORT=29502 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset '../data/sft_guided_drop_thought/train.jsonl' \
    --val_dataset '../data/sft_guided_drop_thought/valid.jsonl' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ../$name \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero3 \

model_dir=$(ls -td ../$name/v* | head -n 1)
checkpoint=$(ls -td ${model_dir}/checkpoint-* | head -n 1)
echo "Exporting model from ${checkpoint}..."

swift export \
--adapters ${checkpoint} \
--merge_lora true