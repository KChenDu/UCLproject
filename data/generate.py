import argparse

from torch.utils.data import Dataset
from re import search, DOTALL
from loguru import logger
from os import cpu_count, remove
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from subprocess import run
from human_eval.data import write_jsonl


def read_train_examples(train_examples: Dataset, prompt_examples: Dataset) -> dict:
    def format_test_example(q: str, tests: list[str], code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), '\n'.join(tests))
        if code is not None:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += f"\n>>> Code:\n```python\n{code}\n```"
        return prompt

    examples_str = [None, None, None]
    for i in range(3):
        example_prompt = format_test_example(prompt_examples[i]['text'], prompt_examples[i]['test_list'], prompt_examples[i]['code'])
        examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

    for example in train_examples:
        prompt = format_test_example(example['text'], example['test_list'], code=None)
        prompt_with_shots = '''Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt)
        yield {'task_id': example['task_id'], 'text': example['text'], 'prompt': prompt_with_shots, 'code': example['code']}


def convert_for_evaluation(generation: str) -> str:
    try:
        generation = search(f'```python\n.*?\n```', generation, DOTALL).group()[10:-3]
    except Exception:
        logger.warning(f"Failed to extract codeblock:\n{generation}")
    return generation


def generate_one(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return convert_for_evaluation(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["deepseek-ai/deepseek-coder-1.3b-base", "deepseek-ai/deepseek-coder-1.3b-instruct"], default="deepseek-ai/deepseek-coder-1.3b-base", type=str)
    parser.add_argument('--num_samples_per_task', default=1, type=int)
    parser.add_argument('--compiler',  choices=["Cython", "Codon"], default="Cython", type=str)
    args = parser.parse_args()

    compiler = args.compiler
    if compiler == "Cython":
        command = ["cython", "generation.py", "-+", "--3"]
    elif compiler == "Codon":
        command = ["codon",  "build", "-release", "-llvm", "generation.py"]
    else:
        raise ValueError

    num_samples_per_task = args.num_samples_per_task
    generated_examples = [None] * num_samples_per_task * 374
    num_proc = cpu_count()
    prompt_examples = load_dataset("mbpp", split="prompt", num_proc=num_proc)
    train_examples = load_dataset("mbpp", split="train", num_proc=num_proc)
    examples = read_train_examples(train_examples, prompt_examples)

    model_name_or_path = args.model
    logger.info("model " + model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    logger.info(f"load tokenizer {tokenizer.__class__} from {model_name_or_path} over.")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()

    for i in range(num_samples_per_task):
        for j, example in enumerate(tqdm(examples, "MBPP", 374, leave=False, unit="example")):
            generation = generate_one(example['prompt'], tokenizer, model)
            with (open('generation.py', 'w') as generation_file):
                print(generation, file=generation_file)
            output = run(command, capture_output=True)
            compilable = output.returncode == 0
            generated_examples[i * 374 + j] = dict(task_id=example['task_id'], sample=i, prompt=example['text'], code=example['code'], generation=generation, compilable=compilable, output=output.stderr.decode())

    logger.info("Generate all over!!!")
    if compiler == "Cython":
        remove("generation.cpp")
    elif compiler == "Codon":
        remove("generation.ll")
    remove("generation.py")
    write_jsonl("mbpp_compiler_feedback.jsonl", generated_examples)
    logger.info(f"Save {num_samples_per_task * 374} processed examples into mbpp_compiler_feedbacks.jsonl over!")
