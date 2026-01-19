#!/bin/bash

INPUT_DIR=./results/baselines

FILES=(
  gpt-5.1_all.txt
)
  #gpt-5.1_all.txt

for file in "${FILES[@]}"; do
  echo "=============================="
  echo "Evaluating $file"
  echo "=============================="
  python ./src/evaluate.py --input_file "$INPUT_DIR/$file"
done