import argparse

from torch.utils.data import Dataset
from re import search, DOTALL
from loguru import logger
from torch import manual_seed
from os import cpu_count, remove
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from subprocess import run
from human_eval.data import write_jsonl


def read_train_examples(train_examples: Dataset, prompt_examples: Dataset, language: str) -> dict:
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
                prompt += f"\n>>> Code:\n```python\n{code}\n```"
            elif language == 'C++':
                prompt += f"\n>>> Code:\n```cpp\n{code}\n```"
            else:
                raise ValueError
        return prompt

    examples_str = [None, None, None]
    if language == 'Python':
        for i in range(3):
            example_prompt = format_train_example(prompt_examples[i]['text'], 'Python', prompt_examples[i]['test_list'], prompt_examples[i]['code'])
            examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

        for example in train_examples:
            prompt = format_train_example(example['text'], language, example['test_list'])
            prompt_with_shots = '''Please refer the given examples and generate a Python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt)
            yield {'task_id': example['task_id'], 'text': example['text'], 'prompt': prompt_with_shots, 'code': example['code']}
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

        for example in train_examples:
            prompt = format_train_example(example['text'], 'C++')
            prompt_with_shots = '''Please refer the given examples and generate a C++ function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt)
            yield {'task_id': example['task_id'], 'text': example['text'].replace('python', 'C++'), 'prompt': prompt_with_shots, 'code': example['code']}
    else:
        raise ValueError


def convert_for_evaluation(generation: str, language: str) -> str:
    try:
        if language == 'C++':
            generation = search('```cpp\n.*?\n```', generation, DOTALL).group()[7:-3]
        if language == 'Python':
            generation = search('```python\n.*?\n```', generation, DOTALL).group()[10:-3]
    except Exception:
        logger.warning(f"Failed to extract codeblock:\n{generation}")
    return generation.lstrip()


def generate_one(prompt: str, new_prompt: str, tokenizer, model, language: str) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    new_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": new_prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(new_inputs, temperature=.9, max_new_tokens=1024, do_sample=True, top_k=0, top_p=.92, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).replace(" [/INST]", "")
    return convert_for_evaluation(output, language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=('C++', 'Python'), default='Python')
    parser.add_argument('--model', choices=('deepseek-ai/deepseek-coder-1.3b-base', 'deepseek-ai/deepseek-coder-1.3b-instruct'), default='deepseek-ai/deepseek-coder-1.3b-base', type=str)
    parser.add_argument('--num_samples_per_task', default=20, type=int)
    parser.add_argument('--num_attempts', default=5, type=int)
    parser.add_argument('--compiler', choices=('Clang', 'Cython', 'Codon'), default='Codon', type=str)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    compiler = args.compiler
    language = args.language
    if language == 'C++':
        assert compiler == 'Clang'
        file = 'generation.cpp'
        command = ("clang", "-S", "-Os", "-fsave-optimization-record=yaml", "generation.cpp")
    elif language == 'Python':
        if compiler == 'Cython':
            command = ("cython", "generation.py", "-+", "--3")
        elif compiler == 'Codon':
            command = ("codon", "build", "-release", "-llvm", "generation.py")
        else:
            raise ValueError
        file = 'generation.py'
    else:
        raise ValueError

    manual_seed(42)
    num_proc = cpu_count()
    prompt_examples = load_dataset("mbpp", split="prompt", num_proc=num_proc)
    if args.demo:
        train_examples = load_dataset("mbpp", split="train[:10]", num_proc=num_proc)
    else:
        train_examples = load_dataset("mbpp", split="train", num_proc=num_proc)

    num_attempts = args.num_attempts
    num_samples_per_task = args.num_samples_per_task
    num_tasks = train_examples.num_rows

    model_name_or_path = args.model
    logger.info("model " + model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    logger.info(f"load tokenizer {tokenizer.__class__} from {model_name_or_path} over.")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()

    for i in range(num_samples_per_task):
        examples = read_train_examples(train_examples, prompt_examples, language)
        for j, example in enumerate(tqdm(examples, f"sample {i}", num_tasks, leave=False, unit="example")):
            prompt = example['prompt']
            if language == 'Python':
                new_prompt = prompt + "\n>>> Code:\n```python\n"
            elif language == 'C++':
                new_prompt = prompt + "\n>>> Code:\n```cpp\n"
            compilable = False
            attempt = 0
            while attempt < num_attempts and not compilable:
                generation = generate_one(prompt, new_prompt, tokenizer, model, language)
                with (open(file, 'w') as generation_file):
                    print(generation, file=generation_file)
                output = run(command, capture_output=True)
                compilable = output.returncode == 0
                if compilable:
                    generated_example = dict(task_id=example['task_id'], sample=i, attempt=attempt, content=example['text'], generation=generation, compilable=True)
                    if language == 'C++' and compiler == 'Clang':
                        optimization = run(("llvm-opt-report", "generation.opt.yaml"), capture_output=True).stdout.decode()
                        generated_example['optimization'] = optimization[optimization.rfind("< generation.cpp\n") + 17:]
                else:
                    output = output.stderr.decode()
                    generated_example = dict(task_id=example['task_id'], sample=i, attempt=attempt, content=example['text'], generation=generation, compilable=False, output=output)
                    if language == 'Python':
                        output = output[18:]
                        try:
                            new_prompt = prompt + "\n>>> Code:\n```python\n" + '\n'.join(generation.splitlines()[:int(output[:output.find(':')]) - 1]) + '\n'
                        except ValueError:
                            new_prompt = prompt + "\n>>> Code:\n```python\n"
                    elif language == 'C++':
                        output = output[15:]
                        new_prompt = prompt + "\n>>> Code:\n```cpp\n" + '\n'.join(generation.splitlines()[:int(output[:output.find(':')]) - 1]) + '\n'
                    else:
                        raise ValueError
                if language == 'Python':
                    generated_example['code'] = example['code']
                write_jsonl("mbpp_compiler_feedback.jsonl", [generated_example], True)
                attempt += 1
    logger.info("Generate all over!!!")
    logger.info(f"Save {num_tasks * num_samples_per_task} processed examples into mbpp_compiler_feedbacks.jsonl over!")

    remove(file)
    if compiler == 'Clang':
        remove("generation.s")
        remove("generation.opt.yaml")
    elif compiler == 'Codon':
        remove("generation.ll")
    elif compiler == "Cython":
        remove("generation.cpp")
    else:
        raise ValueError
