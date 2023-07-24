if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/RUL" ]; then
    mkdir ./logs/RUL
fi

if [ ! -d "./logs/RUL/RMixMLP" ]; then
    mkdir ./logs/RUL/RMixMLP
fi

if [ ! -d "./logs/RUL/RMixMLP/Layers" ]; then
    mkdir ./logs/RUL/RMixMLP/Layers
fi

pred_len=40
model_name=RMixMLP

root_path_name=./dataset/

# FD001
for seq_len in 60
do
for r_times in 0
do
for layers in 1 2 3 4 5 6 7 8
do
  python -u run_RULExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --train_path 'train_FD001.txt' \
    --test_path 'test_FD001.txt' \
    --rul_path 'RUL_FD001.txt' \
    --model_id $seq_len'_'$layers \
    --model $model_name \
    --layers $layers \
    --r_times $r_times \
    --data 'FD001' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu 1 \
    --patience 10 \
    --itr 1 --batch_size 16 --learning_rate 0.001 >logs/RUL/RMixMLP/Layers/FD001'_'$seq_len'_'$layers'_'$r_times.log
done
done
done
