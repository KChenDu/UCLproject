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
        prompt = ">>> Problem:\n{}\n".format(q.strip().replace('python', 'C++'))
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


    if language == 'Python':
        examples_str = [None, None, None]
        for i in range(3):
            example_prompt = format_train_example(prompt_examples[i]['text'], 'Python', prompt_examples[i]['test_list'], prompt_examples[i]['code'])
            examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

        for example in train_examples:
            prompt = format_train_example(example['text'], example['test_list'])
            prompt_with_shots = '''Please refer the given examples and generate a Python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt)
            yield {'task_id': example['task_id'], 'text': example['text'], 'prompt': prompt_with_shots, 'code': example['code']}
    elif language == 'C++':

        examples_str = [None, None, None]
        for i in range(3):
            example_prompt = format_train_example(prompt_examples[i]['text'], 'C++', code='')
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
            generation = search(f'```cpp\n.*?\n```', generation, DOTALL).group()[7:-3]
        if language == 'Python':
            generation = search(f'```python\n.*?\n```', generation, DOTALL).group()[10:-3]
    except Exception:
        logger.warning(f"Failed to extract codeblock:\n{generation}")
    return generation


def generate_one(prompt: str, tokenizer, model) -> str:
    print(prompt)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=0, top_p=.92, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return convert_for_evaluation(output, language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=('C++', 'Python'), default='C++')
    parser.add_argument('--model', choices=('deepseek-ai/deepseek-coder-1.3b-base', 'deepseek-ai/deepseek-coder-1.3b-instruct'), default='deepseek-ai/deepseek-coder-1.3b-base', type=str)
    parser.add_argument('--num_samples_per_task', default=10, type=int)
    parser.add_argument('--compiler', choices=['Clang', 'Cython', 'Codon'], default='Clang', type=str)
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

    num_samples_per_task = args.num_samples_per_task
    num_tasks = train_examples.num_rows
    generated_examples = [None] * num_tasks * num_samples_per_task

    model_name_or_path = args.model
    logger.info("model " + model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    logger.info(f"load tokenizer {tokenizer.__class__} from {model_name_or_path} over.")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()

    for i in range(num_samples_per_task):
        examples = read_train_examples(train_examples, prompt_examples, language)
        for j, example in enumerate(tqdm(examples, f"sample {i}", num_tasks, leave=False, unit="example")):
            generation = generate_one(example['prompt'], tokenizer, model)
            print(generation)
            with (open(file, 'w') as generation_file):
                print(generation, file=generation_file)
            output = run(command, capture_output=True)
            compilable = output.returncode == 0
            if compilable and language == 'C++' and compiler == 'Clang':
                optimization = run(("llvm-opt-report", "generation.opt.yaml"), capture_output=True).stdout.decode()
                print(optimization[optimization.rfind("< generation.cpp\n") + 17:])
                generated_examples[i * num_tasks + j] = dict(task_id=example['task_id'], sample=i, content=example['text'], generation=generation, compilable=compilable, output=output.stderr.decode(), optimization=optimization[optimization.rfind("< generation.cpp\n") + 17:])
            else:
                generated_examples[i * num_tasks + j] = dict(task_id=example['task_id'], sample=i, content=example['text'], code=example['code'], generation=generation, compilable=compilable, output=output.stderr.decode())

    logger.info("Generate all over!!!")
    write_jsonl("mbpp_compiler_feedback.jsonl", generated_examples)
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
# Fine-tune DPO and SFT sources
# check if deepseek support LORA
