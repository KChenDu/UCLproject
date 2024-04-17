import argparse

from re import search, DOTALL
from loguru import logger
from os import cpu_count
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from human_eval.data import write_jsonl


def generate_one(prompt: str, tokenizer, model) -> str:
    prompt = f"Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:\n```python\n{prompt.strip()}\n```"
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def extract_completion(problem: dict, generation: str) -> str:
    try:
        code_block = search(f'```python\n(.*?)\n```', generation, DOTALL).group()[10:-3]
        completion = code_block[len(problem['prompt']):]
    except Exception as exception:
        logger.warning(f"Failed to extract code block with error `{exception}`:\n>>> Task: {problem['task_id']}\n>>> Output:\n{generation}")
        completion = generation
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["deepseek-ai/deepseek-coder-1.3b-base", "deepseek-ai/deepseek-coder-1.3b-instruct"], default="deepseek-ai/deepseek-coder-1.3b-base", type=str)
    parser.add_argument('--num_samples_per_task', default=1, type=int)
    args = parser.parse_args()

    num_samples_per_task = args.num_samples_per_task
    problems = load_dataset("openai_humaneval", split="test", num_proc=cpu_count())
    length = len(problems)
    samples = [None] * length * num_samples_per_task

    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).cuda()

    for i in range(num_samples_per_task):
        for j, problem in enumerate(tqdm(problems, f"sample {i + 1}", leave=False, unit="problem")):
            prompt = problem['prompt']
            generation = generate_one(prompt, tokenizer, model)
            samples[i * length + j] = dict(task_id=problem['task_id'], completion=extract_completion(problem, generation))

    write_jsonl("samples_humaneval.jsonl", samples)
