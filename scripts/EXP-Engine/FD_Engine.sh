if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/RUL" ]; then
    mkdir ./logs/RUL
fi

if [ ! -d "./logs/RUL/RMixMLP" ]; then
    mkdir ./logs/RUL/RMixMLP
fi

if [ ! -d "./logs/RUL/RMixMLP/Engine" ]; then
    mkdir ./logs/RUL/RMixMLP/Engine
fi

seq_len=70
pred_len=1
model_name=RMixMLP

root_path_name=./dataset/

# FD001
for layers in 3
do
for r_times in 2
do
  python -u run_RULExp.py \
    --is_training 0 \
    --do_predict 1 \
    --engine_num 68 \
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
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/RUL/RMixMLP/Engine/FD001'_'68.log
done
done

# FD002
for layers in 1
do
for r_times in 0
do
  python -u run_RULExp.py \
    --is_training 0 \
    --do_predict 1 \
    --engine_num 252 \
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
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/RUL/RMixMLP/Engine/FD002'_'252.log
done
done

# FD003
for layers in 1
do
for r_times in 0
do
  python -u run_RULExp.py \
    --is_training 0 \
    --do_predict 1 \
    --engine_num 99 \
    --root_path $root_path_name \
    --train_path 'train_FD003.txt' \
    --test_path 'test_FD003.txt' \
    --rul_path 'RUL_FD003.txt' \
    --model_id $seq_len'_'$layers \
    --model $model_name \
    --layers $layers \
    --r_times $r_times \
    --data 'FD003' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu 1 \
    --patience 10 \
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/RUL/RMixMLP/Engine/FD003'_'99.log
done
done

# FD004
for layers in 2
do
for r_times in 0
do
  python -u run_RULExp.py \
    --is_training 0 \
    --do_predict 1 \
    --engine_num 12 \
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
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/RUL/RMixMLP/Engine/FD004'_'12.log
done
done

