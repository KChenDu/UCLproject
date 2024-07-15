from os import cpu_count
from datasets import load_dataset
from json import load
from datasets import Dataset


def get_prompt(task_id: int, task_id2data: dict, prompt_examples: Dataset, language: str) -> str:
    def format_train_example(q: str, language: str, tests: list[str] = None, code: str = None):
        if language == 'Python':
            prompt = ">>> Problem:\n{}\n".format(q.strip())
        elif language == 'C++':
            prompt = ">>> Problem:\n{}\n".format(q.strip().replace('python', 'C++'))
        else:
            raise ValueError
        if tests is not None:
            prompt += ">>> Test Cases:\n{}\n".format('\n'.join(tests))
        if code is not None:
            code = code.replace("\r", "").replace("\t", "    ")
            if language == 'Python':
                prompt += f">>> Code:\n```python\n{code}\n```"
            elif language == 'C++':
                prompt += f">>> Code:\n```cpp\n{code}\n```"
            else:
                raise ValueError
        return prompt

    examples_str = [None, None, None]
    if language == 'Python':
        for i in range(3):
            example_prompt = format_train_example(prompt_examples[i]['text'], 'Python', prompt_examples[i]['test_list'], prompt_examples[i]['code'])
            examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

        example = task_id2data[task_id]
        prompt = format_train_example(example['text'], language, example['test_list'])
        prompt_with_shots = '''Please refer the given examples and generate a Python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt) + ">>> Code:\n```python\n"
    elif language == 'C++':
        codes = ['''#include <vector>
#include <algorithm>

using namespace std;

int R = 3;
int C = 3;

int min_cost(vector<vector<int>>& cost, int m, int n) {
    vector<vector<int>> tc(R, vector<int>(C));
    tc[0][0] = cost[0][0];
    for (int i = 1; i <= n; ++i)
        tc[0][i] = tc[0][i - 1] + cost[0][i];
    for (int i = 1; i <= m; ++i)
        tc[i][0] = tc[i - 1][0] + cost[i][0];
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
            tc[i][j] = min({tc[i - 1][j - 1], tc[i - 1][j], tc[i][j - 1]}) + cost[i][j];
    return tc[m][n];
}''',
                 '''#include <vector>
#include <unordered_set>

using namespace std;

vector<int> similar_elements(vector<int>& test_tup1, vector<int>& test_tup2) {
    vector<int> res;
    unordered_set<int> set;
    for (int element : test_tup1)
        set.insert(element);
    for (int element : test_tup2)
        if (set.find(element) != set.end())
            res.push_back(element);
    return res;
}''',
                 '''#include <cmath>

using namespace std;

bool is_not_prime(int n) {
    int sqrt_n = sqrt(n);
    for (int i = 2; i <= sqrt_n; ++i)
        if (n % i == 0)
            return true;
    return false;
}''']
        for i in range(3):
            example_prompt = format_train_example(prompt_examples[i]['text'], 'C++', code=codes[i])
            examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

        example = task_id2data[task_id]
        prompt = format_train_example(example['text'], 'C++')
        prompt_with_shots = '''Please refer the given examples and generate a C++ function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt) + ">>> Code:\n```cpp\n"
    return prompt_with_shots








task_id2data = {}
prompt_examples = load_dataset("mbpp", "full", split="prompt", num_proc=cpu_count()) ######
train_examples = load_dataset("mbpp", "full", split="train", num_proc=cpu_count()) #####

for train_example in train_examples:
    task_id = train_example['task_id']
    train_example.pop('task_id')
    task_id2data[task_id] = train_example

dpo_dataset_dict = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

with open('temp.json', 'r') as file:
    data = load(file)

for task_id, pair in data.items():
    task_id = int(task_id)
    dpo_dataset_dict["prompt"].append(get_prompt(task_id, task_id2data, prompt_examples, 'Python'))
    chosen = pair[0][0]
    if chosen[-1] == '\n':
        chosen += "```"
    else:
        chosen += "\n```"
    dpo_dataset_dict["chosen"].append(chosen)
    rejected = pair[-1][0]
    if rejected[-1] == '\n':
        rejected += "```"
    else:
        rejected += "\n```"
    dpo_dataset_dict["rejected"].append(rejected)

from json import dump
with open('cc.json', 'w') as file:
    dump(dpo_dataset_dict, file)
exit()

train_dataset = Dataset.from_dict(dpo_dataset_dict)


from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, LoraModel
from transformers import AutoTokenizer, AutoModelForCausalLM


training_args = DPOConfig(beta=0.1,
                          output_dir='checkpoints',
                          max_length=1024,
                          max_prompt_length=2048,
                          remove_unused_columns=False,
                          per_device_train_batch_size=1,
                          gradient_accumulation_steps=4
)
config = LoraConfig(task_type="CAUSAL_LM", r=1)
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True).cuda()
lora_model = LoraModel(model, config, "default")
dpo_trainer = DPOTrainer(
    lora_model,
    model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,  # for visual language models, use tokenizer=processor instead
)

dpo_trainer.train()
