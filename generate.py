import argparse

from human_eval.data import read_problems, write_jsonl
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def generate_one_completion(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256, pad_token_id=tokenizer.eos_token_id)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_samples_per_task', type=int)
    args = parser.parse_args()

    num_samples_per_task = args.num_samples_per_task

    problems = read_problems()
    length = len(problems)
    samples = [None] * length * num_samples_per_task
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()

    for i in range(num_samples_per_task):
        for j, task_id in enumerate(tqdm(problems, f"sample {i + 1}", leave=False, unit="problem")):
            samples[i * length + j] = dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))

    write_jsonl("samples.jsonl", samples)
