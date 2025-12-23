#!/bin/bash

cd data_cache

mkdir -p mask_ablation error_ablation

# For mask ablation
for dir in main_1mv_*_mask; do
  if [ -d "$dir" ]; then
    mask="${dir#main_1mv_}"
    src="$dir/qa_pairs_answers_Qwen_Qwen3-32B_text_True_thinking_False_prefix_q_True.csv"
    dst="mask_ablation/qa_pairs_answers_Qwen_Qwen3-32B_text_True_thinking_False_prefix_q_True_${mask}.csv"
    if [ -f "$src" ]; then
      cp "$src" "$dst"
      echo "Copied $src to $dst"
    else
      echo "File not found: $src"
    fi
  fi
done

# For error ablation
for dir in main_1mv_error_*; do
  if [ -d "$dir" ]; then
    suffix="${dir#main_1mv_}"
    src="$dir/qa_pairs_answers_Qwen_Qwen3-32B_text_True_thinking_False_prefix_q_True.csv"
    dst="error_ablation/qa_pairs_answers_Qwen_Qwen3-32B_text_True_thinking_False_prefix_q_True_${suffix}.csv"
    if [ -f "$src" ]; then
      cp "$src" "$dst"
      echo "Copied $src to $dst"
    else
      echo "File not found: $src"
    fi
  fi
done
