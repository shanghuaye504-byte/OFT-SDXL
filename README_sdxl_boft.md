# OFT-SDXL: Butterfly Orthogonal Fine-Tuning for SDXL

This repository provides a complete pipeline for fine-tuning **Stable Diffusion XL (SDXL)** using the **BOFT (Butterfly Orthogonal Fine-Tuning)** algorithm via PEFT.

## 1. Overview
This project is adapted and modified from the following open-source implementations:
* [diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py)
* [peft BOFT DreamBooth examples](https://github.com/huggingface/peft/tree/main/examples/boft_dreambooth)

The pipeline supports automated dataset annotation using **Qwen2-VL**, core BOFT training (UNet + Text Encoders), and optimized inference scripts with weight merging.

---

## 2. Project Structure
'''text
OFT-SDXL/
├── data/training_set/      # Training images and auto-generated metadata.jsonl
├── results_boft/           # Directory for inference test outputs
├── oft-trained-xs/         # Training logs and checkpoints
├── annotator.py            # Automated tagging script using Qwen2-VL
├── test_prompts.json       # List of prompts for evaluation
├── test.py                 # Weight merging and batch inference script
├── train_dreambooth_boft_sdxl.py  # Core BOFT training script
└── .gitignore              # Git ignore configuration


# 3. Environment Setup
Installation

It is recommended to use Python 3.10. Install the dependencies as follows:

Bash
conda create -n oft-sdxl python=3.10 -y
conda activate oft-sdxl

# Install PyTorch (Example for CUDA 12.1)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core project dependencies
pip install diffusers transformers peft accelerate xformers
Exporting Requirements

To export your current environment dependencies, run:

Bash
pip freeze > requirements.txt

# 4. Usage Guide
Step 1: Automated Data Annotation

Place your raw images in ./data/training_set/ and run the annotator to generate metadata.jsonl:

Bash
python annotator.py ./data/training_set
Step 2: BOFT Fine-tuning

Launch the training using accelerate. You can adjust the train_batch_size based on your VRAM:

Bash
accelerate launch train_dreambooth_boft_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="./data/training_set" \
  --output_dir="./oft-trained-xs" \
  --instance_prompt="a photo of skullpanda" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --train_text_encoder \
  --boft_block_size=4 \
  --boft_n_butterfly_factor=1 \
  --mixed_precision="fp16" \
  --max_train_steps=1000 \
  --checkpointing_steps=200 \
  --seed=42
Step 3: Inference & Validation

Update the paths in test.py (base model and checkpoint directories) and run:

Bash
python test.py
