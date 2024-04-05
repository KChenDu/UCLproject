import argparse

from pathlib import Path
from loguru import logger
from json import loads, dumps
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    path = Path(args.file)

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()

    with open(path, 'r') as f, open(path.stem + '_generation.jsonl', 'w') as g:
        for i, line in enumerate(f):
            logger.info(f"Processing line {i + 1}...")
            datum = loads(line)
            task_id = datum["task_id"]
            input_text = loads(line)["prompt"]
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=512)
            g.write(dumps({"task_id": task_id, "generation": tokenizer.decode(outputs[0], skip_special_tokens=True)}) + '\n')
            logger.info(f"Line {i + 1} processed.")
