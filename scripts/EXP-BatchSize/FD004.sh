if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/RUL" ]; then
    mkdir ./logs/RUL
fi

if [ ! -d "./logs/RUL/RMixMLP" ]; then
    mkdir ./logs/RUL/RMixMLP
fi

if [ ! -d "./logs/RUL/RMixMLP/BatchSize" ]; then
    mkdir ./logs/RUL/RMixMLP/BatchSize
fi

seq_len=60
pred_len=40
model_name=RMixMLP

root_path_name=./dataset/

# FD004
for batch_size in 8 16 32 64 128 256
do
for layers in 1
do
for r_times in 0
do
  python -u run_RULExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --train_path 'train_FD004.txt' \
    --test_path 'test_FD004.txt' \
    --rul_path 'RUL_FD004.txt' \
    --model_id $seq_len'_'$layers \
    --model $model_name \
    --layers $layers \
    --r_times $r_times \
    --data 'FD004' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu 1 \
    --patience 10 \
    --itr 1 --batch_size $batch_size --learning_rate 0.001 >logs/RUL/RMixMLP/BatchSize/FD004'_'$seq_len'_'$layers'_'$batch_size'_'$r_times.log
done
done
done

