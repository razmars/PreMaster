@echo off

IF NOT EXIST ".\logs" (
    mkdir .\logs
)

IF NOT EXIST ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

set seq_len=336
set model_name=Linear

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path dataset\exchange_rate\ ^
  --data_path exchange_rate.csv ^
  --model_id Exchange_%seq_len%_96 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --enc_in 8 ^
  --des 'Exp' ^
  --itr 1 --batch_size 8 --learning_rate 0.0005 > logs\LongForecasting\%model_name%_Exchange_%seq_len%_96.log

