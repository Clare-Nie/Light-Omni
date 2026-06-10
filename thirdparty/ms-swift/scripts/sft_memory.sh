MAX_PIXELS=230400 \
NPROC_PER_NODE=8 \
swift sft \
    --model response_adapter_checkpoint_directory \
    --model_type qwen2_5_omni \
    --train_type full \
    --dataset memory_adapter.json \
    --dataset_shuffle true \
    --target_modules all-linear \
    --freeze_vit true \
    --learning_rate 2e-5 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_pixels 230400 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --save_total_limit 1 \
    --save_steps 200 \
    --logging_steps 5 \
    --output_dir 0120_sft_memory \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --max_length 16384 \
    --remove_unused_columns false \
    --training_stage stage2

# setting training_stage1 = False in transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/sft_memory.sh

