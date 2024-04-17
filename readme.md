# UCL project
## Installation

---
### Windows
1. Install `jupyter` via Anaconda
2. Install Pytorch
    ```zsh
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
3. Install `datasets` and `transformers` via Anaconda
4. Install pip packages
    ```zsh
    pip install tqdm
    ```
---
### Ubuntu
1. Install Pytorch
    ```zsh
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
2. Install conda packeges
   ```zsh
   conda install datasets transformers
   ```
3. Install pip packages
    ```zsh
    pip install tqdm
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
---
## Generation
```zsh
python generate.py [--dataset <dataset>] [--num_samples_per_task <num_samples_per_task>]
```
---
## Evaluation
- [HumanEval](human-eval/README.md)
- [HumanEval-X](CodeGeeX/codegeex/benchmark/README_zh.md)
