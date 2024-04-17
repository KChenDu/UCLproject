#!/bin/bash
python -m evaluate.py --model "${1:-deepseek-ai/deepseek-coder-1.3b-base}" --num_samples_per_task "${2:-1}"
