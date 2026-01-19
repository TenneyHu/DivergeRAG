
#!/bin/bash

echo "Starting all..."

LLM_MODELS=(
  "gpt-5-nano"
  "gpt-5.1"
)

for MODEL in "${LLM_MODELS[@]}"; do
  echo "=============================="
  echo "Running baselines for $MODEL"
  echo "=============================="

  python ./src/baselines/closebook.py \
    --k 1 \
    --k_per_query 10 \
    --llm_model "$MODEL" \
    --filename "List"

  python ./src/baselines/vsampling.py \
    --llm_model "$MODEL"

  python ./src/baselines/closebook.py \
    --k 10 \
    --k_per_query 1 \
    --llm_model "$MODEL" \
    --filename "Direct"

  python ./src/baselines/multi_turn.py \
    --llm_model "$MODEL"


done
