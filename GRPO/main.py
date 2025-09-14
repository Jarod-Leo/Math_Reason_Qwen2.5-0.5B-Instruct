# main.py

import os
import wandb
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import config   
from utils import get_gsm8k_questions, correctness_reward_func, int_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func

def main():
    # Initialize Weights & Biases
    os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
    os.environ['WANDB_INIT_TIMEOUT'] = '600'
    
    model_name = config.MODEL_NAME
    training_args = config.TRAINING_ARGS
    output_dir = config.OUTPUT_DIR
    run_name = config.RUN_NAME

    wandb.login(key="4feae54effa30a16590278a5ae843b3a3ca69419")  # Replace with your actual WandB API key
    wandb.init(project=run_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = get_gsm8k_questions()

    # Initialize the trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()