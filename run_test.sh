export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# export NCCL_NET=IB
# export NCCL_IB_HCA=mlx5_0
# export NCCL_DEBUG=info

lr=0.0002
lora_rank=1
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.05
lora_nums=3
blc_alpha=0.0
blc_weight=0.0

# pretrained_model=/public/LoRAMoE/llama2-7b
pretrained_model=meta-llama/Llama-2-7b-hf
pretrained_model_base=meta-llama/Llama-2-7b-hf
tokenizer_path=meta-llama/Llama-2-7b-hf
dataset_dir=./data
validation_file=./data/val.json

per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
max_seq_length=4096
output_dir=./output
exp_name=0501_train_format_for_opensource


# deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file=ds_zero3_offload.json

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 --master_port 29502 \
    run_test.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --model_base ${pretrained_model_base} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train True \
    --do_eval True \
    --seed 41 \
    --bf16 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --peft_path /node_data/mok/module/LoRAMoE/output/0402_train_format_for_opensource/checkpoint-5000/sft_lora_model \
    --eval_steps 5000 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_nums} \
    --blc_alpha ${blc_alpha} \
    --blc_weight ${blc_weight} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --overwrite_output_dir \
    # &> ./output/log/${exp_name}.log

    # --tokenizer_name_or_path ${tokenizer_path} \
