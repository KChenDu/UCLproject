import argparse

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from json import load
from random import sample
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger


def count_parameters(model, trainable=False) -> int:
    n = 0
    if trainable:
        for parameter in model.parameters():
            if parameter.requires_grad:
                n += parameter.numel()
    else:
        for parameter in model.parameters():
            n += parameter.numel()
    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('compiler', 'test', 'mix'), default="compiler", type=str)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--output_dir', default="my_model", type=str)
    args = parser.parse_args()

    training_args = DPOConfig(
        'checkpoints',
        True,
        beta=0.1,
        per_device_train_batch_size=1,
        save_strategy='no',
        remove_unused_columns=False,
        max_length=1024,
        max_prompt_length=2048
    )
    config = LoraConfig(r=4, task_type="CAUSAL_LM") # 看加了多少参数

    dataset = args.dataset

    if dataset == 'compiler':
        with open('compiler_dpo_dataset_dict.json', 'r') as file:
            dpo_dataset_dict = load(file)
    elif dataset == 'test':
        with open('test_dpo_dataset_dict.json', 'r') as file:
            dpo_dataset_dict = load(file)
    elif dataset == 'mix':
        with open('compiler_dpo_dataset_dict.json', 'r') as file1, open('test_dpo_dataset_dict.json', 'r') as file2:
            dict1 = load(file1)
            dict2 = load(file2)
            dpo_dataset_dict = {"prompt": dict1["prompt"] + dict2["prompt"],
                                "chosen": dict1["chosen"] + dict2["chosen"],
                                "rejected": dict1["rejected"] + dict2["rejected"]}
            del dict1
            del dict2
    else:
        raise ValueError

    if args.do_sample:
        indexes = sample(list(range(len(dpo_dataset_dict["prompt"]))), 128)
        for key, value in dpo_dataset_dict.items():
            dpo_dataset_dict[key] = [value[index] for index in indexes] # 超参数 / DPO + LoRA / 数据太少 / Gradient_Checkp or Gradient Acc

    train_dataset = Dataset.from_dict(dpo_dataset_dict)
    del dpo_dataset_dict

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    ref_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base").eval().cuda()
    logger.info(f"model: deepseek-ai/deepseek-coder-1.3b-base | parameters: {count_parameters(ref_model)}")
    
    lora_model = get_peft_model(AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base").cuda(), config).cuda()
    logger.info(f"LoRA parameters: {count_parameters(lora_model, True)}")

    dpo_trainer = DPOTrainer(
        lora_model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(args.output_dir)
    # dpo_trainer.model.merge_and_unload().save_pretrained(args.output_dir)
