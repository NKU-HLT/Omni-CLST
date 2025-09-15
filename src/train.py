import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig
import argparse
import yaml
from trainer.grpo_trainer import GRPOTrainer
from utils.rewards import accuracy_reward, format_reward, accuracy_reward_easy
from dataset.dataset import AudioDataset
import os
from peft import LoraConfig, PeftModel
import json
def parse_args_from_yaml():
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--yaml_config", type=str, required=True, help="Path to YAML config file")
    base_args, remaining_args = base_parser.parse_known_args()
    with open(base_args.yaml_config, "r") as f:
        config_dict = yaml.safe_load(f)
    full_parser = argparse.ArgumentParser()
    full_parser.add_argument("--yaml_config", type=str, required=True)

    full_parser.add_argument("--time", type=str, default=None, help="Optional run time")
    full_parser.add_argument("--device_number", type=int, default=1, help="Number of devices to use for training")

    for key, value in config_dict.items():
        if key == "time" or key == "device_number":
            continue  
        if isinstance(value, bool):
            full_parser.add_argument(f"--{key}", type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=value)
        else:
            full_parser.add_argument(f"--{key}", type=type(value), default=value)

    final_args = full_parser.parse_args()
    return final_args


def main():
    args = parse_args_from_yaml()
    print("args", args)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    run_name = f"{args.name}-epoch{args.epochs}-bs{args.train_batch_size}-lr{args.lr}"
    run_name = f"grpo-{run_name}-{args.time}"
    output_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.think == True:
        reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward}
        reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["format"]]
        reward_weights = [2.0, 1.0]
    else:
        reward_funcs_registry = {"accuracy": accuracy_reward_easy}
        reward_funcs = [reward_funcs_registry["accuracy"]]
        reward_weights = [1.0]

    print(f"Using reward functions: {[func.__name__ for func in reward_funcs]}")
    train_dataset = AudioDataset(args.data_file, is_think=args.think, think_max_len=args.think_max_len, is_think_train=args.is_think_train, is_drop=args.is_drop)
    train_length = len(train_dataset)
    print(f"Train dataset length: {train_length}")

    sample = train_dataset[0]
    print("[DEBUG] Example sample:")
    print(sample)   

    if train_length > 0:
        total_steps_per_epoch = train_length // (args.train_batch_size * args.gradient_accumulation_steps * args.device_number)
        max_steps = args.epochs * total_steps_per_epoch
        save_steps = max(1, total_steps_per_epoch // 10) # Save roughly 10 times per epoch
        logging_steps = 5
        print(f"Estimated training steps: {max_steps} ({args.epochs} epochs)")
        print(f"Saving checkpoints every {save_steps} steps")
    
    learning_rate = float(args.lr)
    training_args = GRPOConfig(
        seed=42,
        data_seed=42,
        output_dir=output_dir, 
        deepspeed=args.ds_config_path, 
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=logging_steps,
        bf16=True,
        report_to=['tensorboard'],
        gradient_checkpointing=False, 
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        run_name="AQA-GRPO", 
        save_steps=save_steps,
        save_only_model=True, 
        temperature=1.2,
        beta=args.beta,
        learning_rate=learning_rate,
        reward_weights=reward_weights,
        num_generations=args.num_generations)
    
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        think=args.think,
        train_dataset=train_dataset,
        eval_dataset=None)

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
