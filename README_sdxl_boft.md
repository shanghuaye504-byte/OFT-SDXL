# OFT-SDXL: BOFT Fine-tuning for SDXL

## Overview
This project fine-tunes **SDXL** using **BOFT/OFT (PEFT)** for DreamBooth-style subject learning (e.g., character/toy identity).

It includes:
- `train_dreambooth_boft_sdxl.py`: BOFT training for UNet (+ optional text encoders).
- `annotator.py`: Auto-captioning with Qwen2-VL to build `metadata.jsonl`.
- `test.py`: BOFT loading and merging for faster inference across checkpoints.

## References
This project is adapted from:
- [diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py)
- [peft BOFT DreamBooth examples](https://github.com/huggingface/peft/tree/main/examples/boft_dreambooth)

## Project Structure
```text
OFT-SDXL/
├── data/training_set/      # training images + metadata.jsonl
├── results_boft/           # inference outputs
├── oft-trained-xs/         # training outputs/checkpoints
├── annotator.py
├── test_prompts.json
├── test.py
├── train_dreambooth_boft_sdxl.py
└── requirements.txt
Environment Setup
Bash
conda create -n oft-sdxl python=3.10 -y
conda activate oft-sdxl

# Install PyTorch (Example for CUDA 12.1)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

pip install -r requirements.txt
accelerate config
Usage
1. Auto-annotate training images

Bash
python annotator.py ./data/training_set
Generates/appends metadata.jsonl in the same folder.

2. Train BOFT adapters (SDXL)

Bash
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
  --boft_n_butterfly_factor 1 \
  --mixed_precision fp16 \
  --max_train_steps 1000 \
  --checkpointing_steps 200 \
  --validation_prompt "a photo of skullpanda, highly detailed, studio lighting" \
  --num_validation_images 4 \
  --seed 42
3. Inference test

Edit paths in test.py, then run:

Bash
python test.py

```bash
git pull origin main
