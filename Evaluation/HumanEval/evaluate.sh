#!/bin/zsh
python generate.py --model "${1:-deepseek-ai/deepseek-coder-1.3b-base}" --num_samples_per_task "${2:-1}"
evaluate_functional_correctness humaneval_samples.jsonl
