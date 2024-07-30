#!/bin/bash
file_path = './checkpoint' # If you have your own trained file, modify this to the directory
!python run_funsd_formnlu.py \
  --dataset_name formnlu \
  --do_predict \
  --model_name_or_path $file_path \
  --output_dir $file_path \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --save_steps -1 \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --learning_rate 2e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --dataloader_num_workers 8 \
  --max_steps 100 \
  --use_auth_token False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
