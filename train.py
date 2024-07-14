from json import loads
from datasets import Dataset


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

train_dataset = Dataset.from_dict(dpo_dataset_dict)


from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, LoraModel
from transformers import AutoTokenizer, AutoModelForCausalLM


training_args = DPOConfig(beta=0.1,
                          output_dir='checkpoints',
                          max_length=1024,
                          max_prompt_length=2048,
                          remove_unused_columns=False,
                          per_device_train_batch_size=1,
                          gradient_accumulation_steps=4
)
config = LoraConfig(task_type="CAUSAL_LM", r=1)
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True).cuda()
lora_model = LoraModel(model, config, "default")
dpo_trainer = DPOTrainer(
    lora_model,
    model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,  # for visual language models, use tokenizer=processor instead
)

dpo_trainer.train()
