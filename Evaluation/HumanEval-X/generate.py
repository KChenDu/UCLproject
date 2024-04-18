import argparse

from re import search, DOTALL, IGNORECASE
from loguru import logger
from os import cpu_count
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from human_eval.data import write_jsonl


def generate_one(prompt: str, lang: str, tokenizer, model) -> str:
    prompt = f"Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:\n```{lang.lower()}\n{prompt.strip()}\n```"
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def extract_completion(problem: dict, generation: str, lang_code: str) -> str:
    try:
        code_block = search(f'```{lang_code}\n.*?\n```', generation, DOTALL | IGNORECASE).group()[4 + len(lang_code):-3]
        # currently only Python
        completion = code_block[search('def .*?\(.*?\).*?:\n( {4}""".*?"""\n)?', code_block, DOTALL).end():]
    except Exception as exception:
        logger.warning(f"Failed to extract code block with error `{exception}`:\n>>> Task: {problem['task_id']}\n>>> Output:\n{generation}")
        completion = generation
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["deepseek-ai/deepseek-coder-1.3b-base", "deepseek-ai/deepseek-coder-1.3b-instruct"], default="deepseek-ai/deepseek-coder-1.3b-base", type=str)
    parser.add_argument('--language', choices=["python", "Python", "cpp"], default="python", type=str)
    parser.add_argument('--num_samples_per_task', default=1, type=int)
    args = parser.parse_args()

    language = args.language.lower()
    problems = load_dataset("THUDM/humaneval-x", language, split="test", num_proc=cpu_count())

    num_samples_per_task = args.num_samples_per_task
    length = len(problems)
    samples = [None] * length * num_samples_per_task
    model = args.model
    logger.info("model " + model)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    logger.info(f"load tokenizer {tokenizer.__class__} from {model} over.")
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).cuda()

    for i in range(num_samples_per_task):
        for j, problem in enumerate(tqdm(problems, f"sample {i + 1}", leave=False, unit="problem")):
            prompt = problem['prompt']
            generation = generate_one(prompt, language, tokenizer, model)
            samples[i * length + j] = dict(task_id=problem['task_id'], prompt=prompt, generation=extract_completion(problem, generation, language))

    logger.info("Generate all over!!!")
    saved_path = "humaneval-x-" + language + "_samples.jsonl"
    write_jsonl(saved_path, samples)
    logger.info(f"Save {length} processed examples into {saved_path} over!")
