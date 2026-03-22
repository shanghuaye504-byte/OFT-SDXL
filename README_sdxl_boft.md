OFT-SDXL (BOFT Fine-tuning for SDXL) 
Overview
This project fine-tunes SDXL using BOFT/OFT (PEFT) for DreamBooth-style subject learning (e.g., character/toy identity).

It includes:

train_dreambooth_boft_sdxl.py: BOFT training for UNet (+ optional text encoders)
annotator.py: auto-captioning/tagging with Qwen2-VL to build metadata.jsonl
test.py: BOFT loading + merge for faster inference across checkpoints
References
This project is adapted from:

diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
peft BOFT DreamBooth examples (examples/boft_dreambooth)
https://github.com/huggingface/peft/tree/main/examples
Project Structure
text
OFT-SDXL/
├── data/training_set/                # training images + metadata.jsonl
├── results_boft/                     # inference outputs
├── oft-trained-xs/                   # training outputs/checkpoints
├── annotator.py
├── test_prompts.json
├── test.py
├── train_dreambooth_boft_sdxl.py
└── requirements.txt

Environment Setup
bash
conda create -n oft-sdxl python=3.10 -y
conda activate oft-sdxl

# Install PyTorch (pick command matching your CUDA/MPS)
# Example (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
accelerate config

Usage
1) Auto-annotate training images

bash
python annotator.py ./data/training_set

Generates/appends metadata.jsonl in the same folder.

2) Train BOFT adapters (SDXL)

bash
accelerate launch train_dreambooth_boft_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --instance_data_dir ./data/training_set \
  --instance_prompt "a photo of skullpanda" \
  --output_dir ./oft-trained-xs \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --text_encoder_lr 5e-6 \
  --train_text_encoder \
  --boft_block_size 4 \
  --boft_block_num 0 \
  --boft_n_butterfly_factor 1 \
  --boft_dropout 0.0 \
  --boft_bias none \
  --mixed_precision fp16 \
  --max_train_steps 1000 \
  --checkpointing_steps 200 \
  --validation_prompt "a photo of skullpanda, highly detailed, studio lighting" \
  --num_validation_images 4 \
  --validation_epochs 50 \
  --seed 42

3) Resume training

bash
accelerate launch train_dreambooth_boft_sdxl.py ... --resume_from_checkpoint latest

4) Inference test

Edit paths in test.py (base_model_path, checkpoint_dirs, prompt_file, output_base_dir), then run:

bash
python test.py

Key Training CLI Arguments (Important)
Required

--pretrained_model_name_or_path
--instance_prompt
one of: --instance_data_dir or --dataset_name
BOFT core

--boft_block_size
--boft_block_num
--boft_n_butterfly_factor
--boft_dropout
--boft_bias
Rule: at least one of boft_block_size / boft_block_num must be non-zero.
Text encoder tuning

--train_text_encoder
--text_encoder_lr
Training control

--train_batch_size
--gradient_accumulation_steps
--learning_rate
--max_train_steps (overrides epochs)
--checkpointing_steps
--resume_from_checkpoint
Precision/performance

--mixed_precision {no,fp16,bf16}
--enable_xformers_memory_efficient_attention
--gradient_checkpointing
--allow_tf32
Validation/logging

--validation_prompt
--num_validation_images
--validation_epochs
--report_to {tensorboard,wandb,...}
