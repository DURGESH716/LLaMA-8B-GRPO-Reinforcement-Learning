import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
# Ensure efficiency_reward_func is imported here
from rewards import code_reward_func, format_reward_func, efficiency_reward_func

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 4-bit Quantization to fit 16 generations in VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# LoRA Configuration
peft_config = LoraConfig(
    r=64, 
    lora_alpha=128,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

def format_mbpp(example):
    """Formats MBPP dataset into a conversation with a strict system prompt."""
    return {
        "prompt": [
            {
                "role": "system", 
                "content": "Provide a concise solution. Reason in <think> tags and provide the Python function in <answer> tags. Do not exceed the token limit."
            },
            {"role": "user", "content": example['prompt']}
        ],
        "answer": example['test_list'][0] # The ground truth assertion
    }

# Load and process dataset
dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
dataset = dataset.map(format_mbpp)

training_args = GRPOConfig(
    output_dir="outputs/llama-8b-python-grpo",
    learning_rate=2e-6, 
    per_device_train_batch_size=1, 
    num_generations=16, # Compare 16 attempts per prompt
    generation_batch_size=16, # Matches generations to keep memory stable
    
    # Increased to allow the model to finish its code
    max_completion_length=2048, 
    
    # Lowered to keep reasoning more focused/less random
    temperature=1.4, 
    
    bf16=True,
    logging_steps=1,
    sync_ref_model=False, 
    beta=0.01 # High beta helps stay close to base model's coding ability
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # Order: Check format and efficiency first, then run code
    reward_funcs=[
        format_reward_func, 
        efficiency_reward_func, 
        code_reward_func
    ],
)

# Start Training
trainer.train()