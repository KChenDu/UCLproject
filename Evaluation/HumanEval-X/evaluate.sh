#!/bin/zsh
python generate.py --model "${1:-deepseek-ai/deepseek-coder-1.3b-base}" --language "${2:-python}" --num_samples_per_task "${3:-1}"
../../CodeGeeX/scripts/evaluate_humaneval_x.sh humaneval-x-python_samples.jsonl python
