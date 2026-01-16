#!/bin/bash

INPUT_DIR=./results/baselines

FILES=(
  gpt-5-mini.txt
  gpt-5-mini_list.txt
  gpt-5-mini_mmr.txt
  gpt-5-mini_rag.txt
  gpt-5-mini_shuffle.txt
  gpt-5-mini_vsampling.txt
)

for file in "${FILES[@]}"; do
  echo "=============================="
  echo "Evaluating $file"
  echo "=============================="
  python ./src/evaluate.py --input_file "$INPUT_DIR/$file"
done