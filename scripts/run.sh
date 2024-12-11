#!/bin/bash

export HF_HOME=/raid/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICES=$1
model_name=$2
model_path='/raid/LLM/'$model_name
cache_dir='./cache'
tasks=none
num_fewshot=none
limit=none
eval_ppl=true
eval_ppl_seqlen=2048
use_cuda_graph=true
seed=0
# Quantization
bits_a=16
sym_a=false
groupsize_a=-1
bits_w=4
sym_w=false
groupsize_w=-1
# GPTQ
gptq=false
gptq_dataset=c4
gptq_nsample=128
gptq_seqlen=2048
gptq_true_sequential=false
gptq_percdamp=0.01
gptq_act_order=false
gptq_static_groups=false
# SpQR
spqr=false
spqr_dataset=c4
spqr_nsample=128
spqr_seqlen=2048
spqr_true_sequential=false
spqr_percdamp=0.01
spqr_perm_order='identity'
spqr_outlier_threshold=0.01
spqr_save_quantization='./cache/spqr_'$model_name
# Chatbot Simulation
chat=false
# Log
logfile='logs/out.txt'
# Analysis Tools
analyze_stats=true
stats_csv_path='cache/opt-125m-w4a16-rtn-stats.csv'
get_layerwise_distance=false
distance_csv_path='cache/opt-125m-w4a16-rtn-dist.csv'

for bits_a in 4
do
for bits_w in 16
do
for gptq in false
do
for spqr in false
do
CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
    --model_path $model_path \
    --cache_dir $cache_dir \
    --tasks $tasks \
    --num_fewshot $num_fewshot \
    --limit $limit \
    --eval_ppl $eval_ppl \
    --eval_ppl_seqlen $eval_ppl_seqlen \
    --use_cuda_graph $use_cuda_graph \
    --seed $seed \
    --bits_a $bits_a \
    --sym_a $sym_a \
    --groupsize_a $groupsize_a \
    --bits_w $bits_w \
    --sym_w $sym_w \
    --groupsize_w $groupsize_w \
    --gptq $gptq \
    --gptq_dataset $gptq_dataset \
    --gptq_nsample $gptq_nsample \
    --gptq_seqlen $gptq_seqlen \
    --gptq_true_sequential $gptq_true_sequential \
    --gptq_percdamp $gptq_percdamp \
    --gptq_act_order $gptq_act_order \
    --gptq_static_groups $gptq_static_groups \
    --spqr $spqr \
    --spqr_dataset $spqr_dataset \
    --spqr_nsample $spqr_nsample \
    --spqr_seqlen $spqr_seqlen \
    --spqr_true_sequential $spqr_true_sequential \
    --spqr_percdamp $spqr_percdamp \
    --spqr_perm_order $spqr_perm_order \
    --spqr_outlier_threshold $spqr_outlier_threshold \
    --spqr_save_quantization $spqr_save_quantization \
    --chat $chat \
    --logfile $logfile \
    --analyze_stats $analyze_stats \
    --stats_csv_path $stats_csv_path \
    --get_layerwise_distance $get_layerwise_distance \
    --distance_csv_path $distance_csv_path
done
done
done
done