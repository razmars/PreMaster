if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting

fi
seq_len=336
model_name=Linear
for pred_len in 96 
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --des 'Exp' \
  --clustering_labels 4\
  --clustering_groups 1 3 1 1 2 0 0 1\
  --itr 1 --batch_size 32 --learning_rate 0.0005 --individual >logs/LongForecasting/$model_name'_I_'exchange_$seq_len'_'$pred_len.log 

done



