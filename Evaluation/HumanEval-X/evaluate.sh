#!/bin/bash
python -m evaluate.py --model "${1:-deepseek-ai/deepseek-coder-1.3b-base}" --language "${2:-python}" --num_samples_per_task "${3:-1}"
