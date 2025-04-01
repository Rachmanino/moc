# Mainly follows the settings in https://github.com/jiaweizzhao/GaLore
DEVICE=${DEVICE-"0"}
IFS=',' read -ra array <<< "$DEVICE"
NGPU="${#array[@]}"
PORT=$(($RANDOM + 10000))

RUN_NAME=${RUN_NAME:-"None"}
CONFIG_NAME=${CONFIG_NAME:-"llama_60m"}
LR=${LR:-"0.0025"}
WD=${WD:-"0"}
GC=${GC:-"0"}
BZ=${BZ:-"256"}
CONTINUE=${CONTINUE:-"none"}
if [ "${CONTINUE}" != "none" ]; then
    readonly continue_from_flag="--continue_from=$CONTINUE"
else
    readonly continue_from_flag=""
fi

RUN_NAME=moc_60m-LR-$LR
TAG=${TAG:-"none"}
if [ "${TAG}" != "none" ]; then
    RUN_NAME=$TAG-$RUN_NAME
fi
STEPS=${STEPS:-"10000"}
if [ "${STEPS}" != "10000" ]; then
    RUN_NAME=$RUN_NAME-STEPS-$STEPS
fi
WU=${WU:-"1000"}
if [ "${WU}" != "1000" ]; then
    RUN_NAME=$RUN_NAME-WU-$WU
fi

CUDA_VISIBLE_DEVICES=$DEVICE torchrun --standalone --nproc-per-node=$NGPU --master-port=$PORT main.py \
    --model_type moc \
    --model_config configs/llama/$CONFIG_NAME.json \
    --lr $LR \
    --optimizer adamw \
    --batch_size $BZ \
    --total_batch_size 512 \
    --num_training_steps $STEPS \
    --warmup_steps $WU \
    --weight_decay $WD \
    --dtype bfloat16 \
    --eval_every 1000 \
    --run_name $RUN_NAME \
    --wandb_project moc-pretrain-60m \
    > ./results/moc/$RUN_NAME.log 2>&1 &