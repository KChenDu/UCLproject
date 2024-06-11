# UCL project
## Installation

---
### Windows
1. Install `jupyter` via Anaconda
2. Install Pytorch
    ```zsh
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
3. Install `datasets`, `transformers` and `cython` via Anaconda
4. Install pip packages
    ```zsh
    pip install tqdm loguru
    ```
5. Install `huggingface_hub[cli]`
   ```zsh
   pip install -U "huggingface_hub[cli]"
   ```
6. Update submodules
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
---
### Ubuntu
1. Install Pytorch
    ```zsh
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
2. Install conda packeges
   ```zsh
   conda install datasets transformers cython
   ```
3. Install pip packages
    ```zsh
    pip install tqdm loguru
    ```
5. Install `huggingface_hub[cli]`
   ```zsh
   pip install -U "huggingface_hub[cli]"
   ```
6. Update submodules
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
---
## Generate
---
```zsh
python data/generate [--model MODEL] [--num_samples_per_task NUM_SAMPLES_PER_TASK]
```
---
## Evaluation
- [HumanEval](human-eval/README.md)
- [HumanEval-X](CodeGeeX/codegeex/benchmark/README_zh.md)
- MBPP
