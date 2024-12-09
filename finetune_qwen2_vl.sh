#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

SAVE_NAME=sft_test             # default values
max_length=768
micro_batch_size=16
save_interval=1000
use_ckpt=False
train_epochs=10
learning_rate=1e-5
gradient_accumulation_steps=1
nproc_per_node=1
img_min_tokens=256
img_max_tokens=512

data_path=test
qwen_ckpt=default_path
pretrain_ckpt=default_path
save_path=default_path
deepspeed_path=default_path
use_lora=false



# 使用getopts解析命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --save-name) SAVE_NAME="$2"; shift ;;
        --max-length) max_length="$2"; shift ;;
        --micro-batch-size) micro_batch_size="$2"; shift ;;
		    --save-interval) save_interval="$2"; shift ;;
		    --use-ckpt) use_ckpt="$2"; shift ;;
        --train-epochs) train_epochs="$2"; shift ;;
        --img-min-tokens) img_min_tokens="$2"; shift ;;
        --img-max-tokens) img_max_tokens="$2"; shift ;;
        --nproc-per-node) nproc_per_node="$2"; shift ;;
        --data-path) data_path="$2"; shift ;;
        --learning-rate) learning_rate="$2"; shift ;;
        --gradient-accumulation-steps) gradient_accumulation_steps="$2"; shift ;;
        --qwen-ckpt) qwen_ckpt="$2"; shift ;;
        --pretrain-ckpt) pretrain_ckpt="$2"; shift ;;
        --save-path) save_path="$2"; shift ;;
        --deepspeed-path) deepspeed_path="$2"; shift ;;
        --use-lora) use_lora=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

GPUS_PER_NODE=${nproc_per_node}
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL=${pretrain_ckpt}
QWEN_PATH=${qwen_ckpt}
DATA=${data_path}
SAVE_PATH="${save_path}/${SAVE_NAME}"


LORA_ARG=""
if [ "$use_lora" = true ]; then
    LORA_ARG="--use_lora"
fi


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune_qwen2_vl.py \
    --model_name_or_path $MODEL \
    --qwen_path $QWEN_PATH \
    --data_path $DATA \
    --bf16 True \
    --fix_vit False \
    --output_dir $SAVE_PATH \
    --num_train_epochs ${train_epochs} \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_interval} \
    --save_total_limit 10 \
    --learning_rate ${learning_rate} \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --img_min_tokens ${img_min_tokens} \
    --img_max_tokens ${img_max_tokens} \
    --report_to "tensorboard" \
    --model_max_length ${max_length} \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --deepspeed ${deepspeed_path} \
    ${LORA_ARG}

