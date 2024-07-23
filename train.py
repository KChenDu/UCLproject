from json import load
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, LoraModel
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    training_args = DPOConfig(beta=0.1,
                              output_dir='checkpoints',
                              max_length=1024,
                              max_prompt_length=2048,
                              remove_unused_columns=False,
                              per_device_train_batch_size=1,
    )

    # config = LoraConfig(task_type="CAUSAL_LM", r=1)

    with open('dpo_dataset_dict.json', 'r') as file:
        dict = load(file)
    for key, value in dict.items():
        dict[key] = value[:10]

    train_dataset = Dataset.from_dict(dict)

    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True).cuda()
    # lora_model = LoraModel(model, config, "default")
    dpo_trainer = DPOTrainer(
        # lora_model,
        model,
        model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,  # for visual language models, use tokenizer=processor instead
    )

    dpo_trainer.train()
    dpo_trainer.save_model()
