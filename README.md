Table of Contents

Introduction

Project Features

Requirements

Setup & Installation

Model Preparation

Dataset Preparation (GSM8K)

Reward Functions

GRPO Training

Inference Before & After Training

Exporting the Model

Project Structure

Results

License

🧩 <a name="introduction"></a> 1. Introduction

Reasoning models outperform standard LLMs by showing explicit intermediate steps (“chain-of-thought”). This project trains such a model using GRPO, a reinforcement learning algorithm optimized for:

Multi-sample rollouts

Stability without a value function

Reward shaping

Efficient training on a single GPU

Unsloth provides:

70% faster training

Low-memory LoRA optimization

vLLM-style fast inference

Simple GRPOTrainer API

This project walks through the entire pipeline from model loading → GRPO training → evaluation → export.

⭐ <a name="project-features"></a> 2. Project Features
✔ Converts Llama 3.1 8B into a Reasoning Model
✔ Uses GRPO (Group Relative Policy Optimization)
✔ Produces XML-formatted reasoning + answer blocks
✔ Trains on GSM8K math reasoning dataset
✔ Multi-reward shaping for stability
✔ LoRA-based efficient fine-tuning
✔ Merged FP16 and 4-bit model export
✔ HuggingFace Hub upload support
✔ vLLM-compatible inference
🖥️ <a name="requirements"></a> 3. Hardware Requirements

Minimum:

16GB GPU (for 4-bit training)

32GB GPU recommended

Best:

AMD MI300X

A100 / H100

RTX 4090 (for 4-bit LoRA training)

⚙️ <a name="installation"></a> 4. Setup & Installation
Clone and install Unsloth:
git clone https://github.com/unslothai/unsloth.git
cd unsloth
pip install -e .
Install dependencies:
pip install unsloth unsloth_zoo bitsandbytes accelerate trl datasets vllm
🧱 <a name="model-preparation"></a> 5. Model Preparation

Load the base Llama 3.1 8B model with LoRA adapters:

from unsloth import FastLanguageModel
max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

This initializes the model for LoRA-based RL training.

📚 <a name="dataset-preparation"></a> 6. Dataset Preparation (GSM8K)

We train on the GSM8K math reasoning dataset in XML-formatted CoT style.

System prompt format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
Dataset loader:
from datasets import load_dataset

def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]}
        ],
        "answer": extract_hash_answer(x["answer"])
    })
    return data
🏆 <a name="reward-functions"></a> 7. Reward Functions

To make the model produce structured reasoning, we design five independent reward signals:

Reward Function	Purpose
xmlcount_reward_func	Partial credit for correct XML tag usage
soft_format_reward_func	Loose check for <reasoning> + <answer>
strict_format_reward_func	Exact match for strict XML formatting
int_reward_func	Rewards if the final answer is numeric
correctness_reward_func	Reward = 2.0 if final answer is correct

These rewards together stabilize training and enforce format discipline.

🧠 <a name="grpo-training"></a> 8. GRPO Training
Configure GRPO:
from trl import GRPOConfig

training_args = GRPOConfig(
    learning_rate=5e-6,
    num_generations=6,
    max_prompt_length=256,
    max_completion_length=768,
    max_steps=250,
    per_device_train_batch_size=1,
    optim="paged_adamw_8bit",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    output_dir="outputs"
)
Launch training:
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
What happens during GRPO?

For each prompt:

Model generates 6 reasoning samples

Rewards are computed for each sample

GRPO computes relative advantages

LoRA weights are updated to maximize high-reward reasoning

Model gradually learns:
✔ correct answers
✔ clean XML reasoning
✔ numerical discipline
✔ structured chain-of-thought

🔎 <a name="inference"></a> 9. Inference Before & After LoRA
Before applying LoRA (base model):
output = model.fast_generate(
    [text], sampling_params=sampling_params, lora_request=None
)[0].outputs[0].text
Save LoRA:
model.save_lora("grpo_saved_lora")
After applying LoRA:
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

Expected behavior:

The model now outputs explicit CoT reasoning.

Uses <reasoning></reasoning> and <answer></answer> format.

More accurate final answers.

📦 <a name="exporting"></a> 10. Exporting the Model

Unsloth allows saving in FP16, 4-bit, or LoRA-only formats:

Save as FP16 merged model:
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
Save as 4-bit merged model:
model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
Save only LoRA adapters:
model.save_pretrained_merged("model", tokenizer, save_method="lora")
Push to HuggingFace Hub:
model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")
📁 <a name="project-structure"></a> 11. Project Structure
├── notebook.ipynb
├── README.md
├── outputs/
│   ├── trainer_state.json
│   ├── pytorch_lora_weights.bin
├── grpo_saved_lora/
├── model/ (optional FP16 or 4bit merged export)
📊 <a name="results"></a> 12. Results

After ~250 GRPO steps:

✔ Model reliably outputs structured XML reasoning
✔ Improved final answer accuracy on GSM8K samples
✔ Long chain-of-thought (~500-700 tokens)
✔ Much more interpretable intermediate steps
✔ Easily deployable via VLLM

A full 1-epoch training run yields even stronger improvements.

📄 <a name="license"></a> 13. License

Base model license: Meta Llama 3.1

Code license: Apache 2.0

Dataset license: GSM8K (MIT License)

Please check individual components for compliance.
