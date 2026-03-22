"""
inference_boft_fast.py
======================
Optimized version: Loads BOFT checkpoints and merges weights for high-speed inference.
"""

import os
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from peft import PeftModel

def _load_prompts(prompt_file: str) -> list:
    path = Path(prompt_file)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(p) for p in data]
        if isinstance(data, dict) and "prompts" in data:
            return [str(p) for p in data["prompts"]]
        raise ValueError(f"Unsupported JSON format: {prompt_file}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

def clear_memory():
    """Thoroughly clear VRAM/RAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.empty_cache()

def _load_boft_pipeline(
    base_model_path: str,
    ckpt_dir: str,
    device: str,
    train_text_encoder: bool = True,
    vae: AutoencoderKL = None,
) -> StableDiffusionXLPipeline:
    
    print(f"  > Loading base components from {base_model_path}...")

    # ---- 1) UNet + BOFT Merge ----
    unet = UNet2DConditionModel.from_pretrained(
        base_model_path, subfolder="unet", torch_dtype=torch.float16
    )
    unet_boft_path = os.path.join(ckpt_dir, "unet_boft")
    if os.path.exists(unet_boft_path):
        print(f"  > Merging UNet BOFT: {unet_boft_path}")
        unet = PeftModel.from_pretrained(unet, unet_boft_path)
        # Merge weights: This is the key to acceleration
        unet = unet.merge_and_unload()
    unet.eval()

    # ---- 2) Text Encoders + BOFT Merge ----
    extra_kwargs = {}
    if train_text_encoder:
        # Text Encoder 1
        te_one_path = os.path.join(ckpt_dir, "text_encoder_one_boft")
        if os.path.exists(te_one_path):
            print(f"  > Merging Text Encoder One BOFT: {te_one_path}")
            te_one = CLIPTextModel.from_pretrained(
                base_model_path, subfolder="text_encoder", torch_dtype=torch.float16
            )
            te_one = PeftModel.from_pretrained(te_one, te_one_path)
            te_one = te_one.merge_and_unload() # Merge
            te_one.eval()
            extra_kwargs["text_encoder"] = te_one

        # Text Encoder 2
        te_two_path = os.path.join(ckpt_dir, "text_encoder_two_boft")
        if os.path.exists(te_two_path):
            print(f"  > Merging Text Encoder Two BOFT: {te_two_path}")
            te_two = CLIPTextModelWithProjection.from_pretrained(
                base_model_path, subfolder="text_encoder_2", torch_dtype=torch.float16
            )
            te_two = PeftModel.from_pretrained(te_two, te_two_path)
            te_two = te_two.merge_and_unload() # Merge
            te_two.eval()
            extra_kwargs["text_encoder_2"] = te_two

    # ---- 3) VAE ----
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=torch.float16
        )

    # ---- 4) Assemble Pipeline ----
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        **extra_kwargs,
    )
    
    # Force switch scheduler (SDXL recommends DPM++ series)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True
    )
    
    pipeline.to(device)
    return pipeline

def run_inference_boft(
    base_model_path: str,
    checkpoint_dirs: list,
    prompt_file: str,
    output_base_dir: str,
    train_text_encoder: bool = True,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    resolution: int = 1024,
):
    prompts = _load_prompts(prompt_file)
    os.makedirs(output_base_dir, exist_ok=True)

    # Use fixed VAE to prevent black images in SDXL fp16
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"!!! Device detected: {device} !!!")

    total_checkpoints = len(checkpoint_dirs)

    for ckpt_idx, ckpt_dir in enumerate(checkpoint_dirs, start=1):
        ckpt_name = os.path.basename(ckpt_dir)
        output_dir = os.path.join(output_base_dir, ckpt_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing Checkpoint ({ckpt_idx}/{total_checkpoints}): {ckpt_dir}")
        print(f"{'='*70}")

        # Load and merge
        pipeline = _load_boft_pipeline(
            base_model_path, ckpt_dir, device, train_text_encoder, vae=vae
        )
        pipeline.set_progress_bar_config(disable=True)

        for prompt_idx, prompt in enumerate(prompts, start=1):
            generator = torch.Generator(device=device).manual_seed(seed + prompt_idx - 1)

            pbar = tqdm(
                total=num_inference_steps,
                desc=f"Generating [P{prompt_idx}/{len(prompts)}]",
                leave=False,
            )

            # Callback for tqdm progress bar updates
            def callback(step, timestep, latents):
                pbar.update(1)

            image = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=resolution,
                width=resolution,
                callback=callback,
                callback_steps=1,
            ).images[0]

            pbar.close()
            save_path = os.path.join(output_dir, f"{prompt_idx-1:04d}.png")
            image.save(save_path)
            tqdm.write(f"  [✓] Saved: {save_path}")

        # Completely release VRAM before loading the next checkpoint
        del pipeline
        clear_memory()

    print(f"\n[Done] All results saved to: {output_base_dir}")

if __name__ == "__main__":
    # Configure your paths here
    run_inference_boft(
        base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
        checkpoint_dirs=[
            "boft-trained-xl/checkpoint-200",
            "boft-trained-xl/checkpoint-400",
            "boft-trained-xl/checkpoint-600",
            "boft-trained-xl/checkpoint-800",
            "boft-trained-xl/checkpoint-1000",
        ],
        prompt_file="./test_prompts.json",
        output_base_dir="results_boft_fast/",
        train_text_encoder=True,
        num_inference_steps=30,  # 30 steps is safe after Merging
        guidance_scale=7.5,
        seed=42,
        resolution=1024,
    )
