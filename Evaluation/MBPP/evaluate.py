import argparse

from torch.utils.data import Dataset
from re import search, DOTALL, IGNORECASE
from loguru import logger
from os import cpu_count
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from human_eval.data import write_jsonl


def read_test_examples(test_examples: Dataset, prompt_examples: Dataset) -> dict:
    def format_test_example(q: str, tests: list[str], code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), '\n'.join(tests))
        if code is not None:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += f"\n>>> Code:\n```python\n{code}\n```"
        return prompt

    # test_cases
    length = len(prompt_examples)
    examples_str = [None] * length
    for i, example in enumerate(prompt_examples):
        example_prompt = format_test_example(example['text'], example['test_list'], example['code'])
        examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

    for example in test_examples:
        prompt = format_test_example(example['text'], example['test_list'], code=None)
        prompt_with_shots = '''Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}'''.format('\n\n'.join(examples_str), prompt)
        yield {'task_id': example['task_id'], 'prompt': prompt_with_shots}


def convert_for_evaluation(generation: str) -> str:
    try:
        generation = search(f'```python\n.*?\n```', generation, DOTALL).group()
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
    args = parser.parse_args()

    model = args.model
    logger.info("model " + model)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    logger.info(f"load tokenizer {tokenizer.__class__} from {model} over.")
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).cuda()

    generated_examples = [None] * 500
    num_proc = cpu_count()
    prompt_examples = load_dataset("mbpp", split="prompt", num_proc=num_proc)
    test_examples = load_dataset("mbpp", split="test", num_proc=num_proc)
    examples = read_test_examples(test_examples, prompt_examples)

    for i, example in enumerate(tqdm(examples, "MBPP", 500, leave=False, unit="example")):
        generated_examples[i] = generate_one(example['prompt'], tokenizer, model)

    logger.info("Generate all over!!!")
    write_jsonl("mbpp_samples.jsonl", generated_examples)
    logger.info(f"Save 500 processed examples into mbpp_samples.jsonl over!")
