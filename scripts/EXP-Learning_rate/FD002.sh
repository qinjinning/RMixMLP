if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/RUL" ]; then
    mkdir ./logs/RUL
fi

if [ ! -d "./logs/RUL/RMixMLP" ]; then
    mkdir ./logs/RUL/RMixMLP
fi

if [ ! -d "./logs/RUL/RMixMLP/LearningRate" ]; then
    mkdir ./logs/RUL/RMixMLP/LearningRate
fi

seq_len=60
pred_len=40
model_name=RMixMLP

root_path_name=./dataset/

# FD002
for learning_rate in 0.0001 0.0005 0.001 0.005 0.01 0.05
do
for layers in 1
do
for r_times in 0
do
  python -u run_RULExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --train_path 'train_FD002.txt' \
    --test_path 'test_FD002.txt' \
    --rul_path 'RUL_FD002.txt' \
    --model_id $seq_len'_'$layers \
    --model $model_name \
    --layers $layers \
    --r_times $r_times \
    --data 'FD002' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu 1 \
    --patience 10 \
    --itr 1 --batch_size 16 --learning_rate $learning_rate >logs/RUL/RMixMLP/LearningRate/FD002'_'$seq_len'_'$layers'_'$learning_rate'_'$r_times.log
done
done
done

