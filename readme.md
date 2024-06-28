# UCL project
## Installation

---
### Windows
1. Install `jupyter` via Anaconda
2. Install Pytorch
    ```zsh
    pip3 install torch --index-url https://download.pytorch.org/whl/cu121
    ```
3. Install pip packages
    ```zsh
    pip install transformers[torch] datasets cython scikit-learn evaluate trl peft tqdm loguru
    ```
4. Install `huggingface_hub[cli]`
   ```zsh
   pip install -U "huggingface_hub[cli]"
   ```
5. Update submodules
   - First time
       ```zsh
       git submodule update --init --recursive
       ```
    - After the first time
       ```zsh
       git submodule update --recursive --remote
       ```
6. Install `human-eval`
   ```zsh
   pip install -e human-eval
   ```
---
### Ubuntu
1. Install Pytorch
    ```zsh
    pip3 install torch --index-url https://download.pytorch.org/whl/cu121
    ```
2. Install conda packeges
   ```zsh
   pip install transformers[torch] datasets cython evaluate trl peft tqdm loguru
   ```
3. Install `huggingface_hub[cli]`
   ```zsh
   pip install -U "huggingface_hub[cli]"
   ```
4. Update submodules
   - First time
       ```zsh
       git submodule update --init --recursive
       ```
    - After the first time
       ```zsh
       git submodule update --recursive --remote
       ```

7. Install `human-eval`
   ```zsh
   pip install -e human-eval
   ```
8. Install Codon
   ```zsh
   /bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
   ```
---
## Generate
---
```zsh
python data/MBPP/generate [--language LANGUAGE] [--model MODEL] [--num_samples_per_task NUM_SAMPLES_PER_TASK] [--compiler COMPILER] [--demo]
python data/MBPP/generate [--language LANGUAGE] [--model MODEL] [--num_samples_per_task NUM_SAMPLES_PER_TASK] [--compiler COMPILER] [--demo]
```
---
## Evaluation
- [HumanEval](human-eval/README.md)
- [HumanEval-X](CodeGeeX/codegeex/benchmark/README_zh.md)
- MBPP
