# Configuration settings for the GRPO training process
from trl import GRPOConfig

# Model parameters
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/Qwen2.5-0.5B-reasoning-GRPO"
RUN_NAME = "Qwen2.5-0.5B-GRPO-gsm8k"

# Training arguments
TRAINING_ARGS = GRPOConfig(
    learning_rate =  5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 40,
    gradient_accumulation_steps = 2,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 1,
    save_steps = 100,
    max_grad_norm = 0.1,
    log_on_each_node = False,
    report_to = "wandb"
)