import os
import sys
import json
import argparse
import torch
import re
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ================= Configuration Area =================
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

PROMPT_TEXT = """Act as an expert visual data annotator for a text-to-image AI model (SDXL). I am preparing a training dataset for a specific blind box art toy figure. The core trigger word for this figure is "skullpanda".

Your task is to scan the provided image and generate a highly detailed, comma-separated list of descriptive tags.

CRITICAL FORMATTING RULES (STRICT ANTI-LOOP):
1. Output strictly in English.
2. Output ONLY a single paragraph of comma-separated descriptive phrases. 
3. Do NOT use full sentences, conjunctions (like "and", "with"), or conversational filler.
4. You MUST start your exact response with: "a photo of skullpanda, "
5. HARD STOP REQUIREMENT: You MUST end your entire response with a single period (.). Once you type the period, STOP GENERATING IMMEDIATELY. Do not add anything after the period.
6. ABSOLUTE NO REPETITION: You are strictly forbidden from repeating any noun, adjective, or concept. Check your previous tags before writing a new one.
7. EXACT LENGTH: Generate between 15 and 25 unique tags. Once you have described the character, outfit, and environment, place your period (.) and finish the task.

CRITICAL CONTENT GUIDELINES (For SDXL Fine-tuning):
1. Emphasize Toy Materials: Always include the tag "art toy". Explicitly describe surface textures to distinguish it from a real human (e.g., glossy plastic face, matte body, metallic accents, translucent resin, smooth texture).
2. Disentangle Core Features from Outfits: To help the AI learn the core face, you must describe the variable elements in extreme detail. Describe the specific theme (e.g., gothic, sci-fi, fantasy), the exact headpiece, clothing, accessories, and props.
3. Facial Details (CRITICAL DISENTANGLEMENT RULE): 
   - DO NOT describe the permanent physical shape of the face (NEVER use tags like "chubby cheeks", "round face", or "facial proportions"). We want the AI to bind the core face shape entirely to the word "skullpanda".
   - DO describe the variable surface details: skin color (e.g., white skin, pale skin), skin texture (e.g., pearlescent skin, glossy skin), eye state/color (e.g., closed eyes, blue eyes), makeup (e.g., heavy eye makeup, glossy dark lips), and expression (e.g., melancholy look).
4. Environment: Describe the background, lighting, and color palette (e.g., studio lighting, dark background, cinematic lighting).

EXAMPLE INPUT: [An image of a toy figure holding a pumpkin surrounded by ghosts]
EXAMPLE OUTPUT: a photo of skullpanda, art toy, glossy pale plastic skin, closed eyes, red eye makeup, subtle blush, red hair, intricate white lace veil, holding a glowing carved pumpkin, white flowing dress, floating translucent white ghosts, dark gothic castle background, cobblestone ground, lit white candles, moonlight, spooky atmosphere, smooth texture, cinematic lighting, highly detailed.

Now, please describe the attached image following the exact format of the EXAMPLE OUTPUT. Remember to END WITH A PERIOD (.) and DO NOT REPEAT tags."""
# ====================================================

def process_vision_info(messages):
    """Helper function recommended by Qwen2-VL for processing vision info."""
    image_inputs = []
    video_inputs = []
    for message in messages:
        for content in message.get("content", []):
            if content["type"] == "image":
                image_inputs.append(content["image"])
            elif content["type"] == "video":
                video_inputs.append(content["video"])
    return image_inputs if image_inputs else None, video_inputs if video_inputs else None

def main():
    parser = argparse.ArgumentParser(description="Generate tags for images using Qwen-VL and save to metadata.jsonl")
    parser.add_argument("folder", type=str, help="Path to the folder containing training images")
    args = parser.parse_args()
    
    image_folder = args.folder
    
    if not os.path.isdir(image_folder):
        print(f"Error: Directory '{image_folder}' not found.")
        sys.exit(1)

    jsonl_path = os.path.join(image_folder, "metadata.jsonl")
    
    # ================= Breakpoint Continuation & File Repair =================
    processed_files = set()
    valid_lines = []
    file_needs_repair = False

    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    processed_files.add(data.get("file_name"))
                    valid_lines.append(line) # Keep track of valid lines
                except json.JSONDecodeError:
                    # Found a corrupted line (e.g., due to sudden power loss)
                    file_needs_repair = True
                    print("⚠️ Found corrupted JSONL records (likely due to unexpected interruption). Auto-cleaning...")
                    
        # Rewrite the file with only valid lines if corruption was detected
        if file_needs_repair:
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for valid_line in valid_lines:
                    f.write(valid_line + "\n")
            print("✅ Corrupted records have been successfully cleaned.")

        print(f"Found existing metadata.jsonl with {len(processed_files)} complete records.")
    # =========================================================================

    # Hardware acceleration detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🚀 Apple Silicon (MPS) GPU acceleration enabled!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🚀 NVIDIA CUDA GPU acceleration enabled!")
    else:
        device = torch.device("cpu")
        print("\n⚠️ Warning: No GPU detected. Running on CPU will be extremely slow!")

    print("Loading model and processor, please wait...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!\n")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        if filename in processed_files:
            print(f"Skipping already processed image: {filename}")
            continue
            
        image_path = os.path.join(image_folder, filename)
        print(f"Processing: {filename} ...")
        
        try:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = inputs.to(device)

            # Increased max_new_tokens to 512 to prevent output truncation
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            caption = output_text.strip()
            
            # ================= Text Cleaning Mechanism =================
            # 1. Replace newlines, carriage returns, and colons with commas
            caption = caption.replace('\n', ', ').replace('\r', ', ').replace(':', ', ')
            
            # 2. Use Regex to clean up redundant spaces and multiple commas
            caption = re.sub(r'\s+', ' ', caption)       # Replace multiple spaces with a single space
            caption = re.sub(r'[,]+', ',', caption)      # Replace multiple commas with a single comma
            caption = caption.replace(' ,', ',')         # Fix spaces before commas
            caption = caption.strip(', ')                # Remove leading/trailing commas and spaces
            
            # 3. Ensure it starts exactly with the trigger phrase
            if not caption.startswith("a photo of skullpanda,"):
                # Remove the phrase if it exists elsewhere, then prepend it cleanly
                caption = caption.replace("a photo of skullpanda,", "").strip(', ')
                caption = "a photo of skullpanda, " + caption
            # ===========================================================
            
            record = {
                "file_name": filename,
                "text": caption
            }
            
            # Force flush to disk to prevent data loss on sudden power failure
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                
            print(f"  - Successfully generated tags: {caption}\n")
            
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

if __name__ == "__main__":
    main()
