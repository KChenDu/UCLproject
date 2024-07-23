from datasets import Dataset
from argparse import ArgumentParser
from pathlib import Path
from os import cpu_count
from datasets import load_dataset
from json import loads, dump


def get_prompt(task_id: int, task_id2data: dict, prompt_examples: Dataset, language: str) -> str:
    def format_train_example(q: str, language: str, tests: list[str] = None, code: str = None):
        q = q.strip()
        if language == 'C++':
            q = q.replace('python', 'C++')
        prompt = f">>> Problem:\n{q}\n"
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    path = Path(args.path)
    assert path.is_file()

    task_id2data = {}

    n_cpu = cpu_count()
    prompt_examples = load_dataset("mbpp", "full", split="prompt", num_proc=n_cpu)  #
    train_examples = load_dataset("mbpp", "full", split="train", num_proc=n_cpu)  #

    for train_example in train_examples:
        task_id = train_example['task_id']
        train_example.pop('task_id')
        task_id2data[task_id] = train_example
    del train_examples

    task_id2samples = {}

    with open(path, 'r') as f:
        for line in f:
            data = loads(line)
            data.pop('attempt')
            sample = data['sample']
            data.pop('sample')
            task_id = data['task_id']
            data.pop('task_id')
            if task_id not in task_id2samples:
                task_id2samples[task_id] = [[data]]
            elif sample >= len(task_id2samples[task_id]):
                task_id2samples[task_id].append([data])
            else:
                task_id2samples[task_id][sample].append(data)

    dpo_dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for task_id, samples in task_id2samples.items():
        prompt = get_prompt(task_id, task_id2data, prompt_examples, 'Python') + ">>> Code:\n```python\n"
        for sample in samples:
            if sample[-1]['compilable']:
                for attempt in sample[:-1]:
                    dpo_dataset_dict["prompt"].append(prompt)
                    dpo_dataset_dict["chosen"].append(sample[-1]['generation'] + "```")
                    dpo_dataset_dict["rejected"].append(attempt['generation'])
                if not sample[-1]['pass']:
                    dpo_dataset_dict["prompt"].append(prompt)
                    dpo_dataset_dict["chosen"].append(task_id2data[task_id]['code'] + "\n```")
                    dpo_dataset_dict["rejected"].append(sample[-1]['generation'])
            else:
                dpo_dataset_dict["prompt"].append(prompt)
                dpo_dataset_dict["chosen"].append(task_id2data[task_id]['code'] + "\n```")
                dpo_dataset_dict["rejected"].append(sample[-1]['generation'])
    with open('dpo_dataset_dict.json', 'w') as f:
        dump(dpo_dataset_dict, f)
