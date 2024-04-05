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
    parser.add_argument('--dataset', choices=['humaneval', 'humaneval-x'], default='humaneval', type=str)
    parser.add_argument('--num_samples_per_task', default=1, type=int)
    args = parser.parse_args()

    dataset = args.dataset

    if dataset == 'humaneval':
        key = "completion"
        problems = read_problems()
    elif dataset == 'humaneval-x':
        key = "generation"
        problems = read_problems("data/humaneval_python.jsonl")
    else:
        raise ValueError

    num_samples_per_task = args.num_samples_per_task
    length = len(problems)
    samples = [None] * length * num_samples_per_task

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()

    if dataset == 'humaneval':
        for i in range(num_samples_per_task):
            for j, task_id in enumerate(tqdm(problems, f"sample {i + 1}", leave=False, unit="problem")):
                samples[i * length + j] = dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
                # if j > 4:
                #     break
    elif dataset == 'humaneval-x':
        for i in range(num_samples_per_task):
            for j, task_id in enumerate(tqdm(problems, f"sample {i + 1}", leave=False, unit="problem")):
                prompt = problems[task_id]["prompt"]
                samples[i * length + j] = dict(task_id=task_id, prompt=prompt, generation=generate_one_completion(prompt))
                # if j > 4:
                #     break
    else:
        raise ValueError

    write_jsonl("samples_" + dataset + ".jsonl", samples)
git 