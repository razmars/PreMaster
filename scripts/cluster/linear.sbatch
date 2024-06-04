#!/bin/bash

### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH

#SBATCH --partition main                ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 0-10:30:00               ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name linear            ### name of the job. replace my_job with your desired job name
#SBATCH --output my_job-id-%J.out       ### output log for running job - %J is the job number variable
#SBATCH --mail-user=razmar@post.bgu.ac.il   ### user’s email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL      ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1                        ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team’s permission
##SBATCH --tasks=1                      ### 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

### Print some data to output file ###
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
nvidia-smi -L

### Start your code below ####
module load anaconda      ### load anaconda module
source activate preMaster ### activate a conda environment, replace my_env with your conda environment

### Create necessary directories ###
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Linear
for pred_len in 96 192 336 720
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
  --itr 1 --batch_size 8 --learning_rate 0.005 --individual >logs/LongForecasting/$model_name'_I_'exchange_$seq_len'_'$pred_len.log  
done

