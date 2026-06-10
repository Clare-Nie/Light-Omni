# 50176 230400

MAX_PIXELS=230400 \
NPROC_PER_NODE=8 \
swift sft \
    --model /huggingface_cache/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00 \
    --model_type qwen2_5_omni \
    --train_type full \
    --dataset generation_adapter.json \
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
    --output_dir 0120_sft_response \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --max_length 32768 \
    --remove_unused_columns false \
    --training_stage stage1



# setting training_stage1 = True in transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/sft_response.sh




