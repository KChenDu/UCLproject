from json import loads


dpo_dataset_dict = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

with open('MBPP(python)_nucleus92_demo.jsonl', 'r') as file:
    for line in file:
        datum = loads(line)
        dpo_dataset_dict["prompt"].append(datum["content"])
        dpo_dataset_dict["chosen"].append(datum["code"])
        dpo_dataset_dict["rejected"].append(datum["generation"])

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


base_model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True).cuda()

lora_config = LoraConfig(r=1)
peft_model = get_peft_model(base_model, lora_config)

for e in base_model.modules():
    print(e)

from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer
from datasets import Dataset


training_args = DPOConfig(beta=0.1, output_dir='temp')
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True)

train_dataset = Dataset.from_dict(dpo_dataset_dict)

dpo_trainer = DPOTrainer(
    peft_model,
    None,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,  # for visual language models, use tokenizer=processor instead
)

dpo_trainer.train()
