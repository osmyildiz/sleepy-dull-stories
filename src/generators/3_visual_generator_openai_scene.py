import os
import json
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Config paths
PROMPTS_PATH = "../output/1/visual_generation_prompts.json"
IMAGE_OUTPUT_DIR = "../output/1/images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def optimize_prompt(prompt: str, scene_data: dict) -> str:
    scene_number = scene_data.get('scene_number', 0)
    characters = scene_data.get('characters_in_scene', [])

    prefix = "79 CE Pompeii: "
    char_text = f"Roman citizen {characters[0]['name']} in period clothing. " if characters else ""

    ending = (
        ". Roman architecture, historically accurate."
        if scene_number != 99
        else ". YouTube thumbnail, dramatic background, Vesuvius eruption."
    )

    final = prefix + char_text + prompt + ending
    return final[:297] + "..." if len(final) > 300 else final


def generate_image(prompt: str, scene_number: int):
    print(f"🎨 Generating image for Scene {scene_number}...")

    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1536x1024",
            quality= "high",

        )

        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        filename = "thumbnail.png" if scene_number == 99 else f"scene_{scene_number:02d}.png"
        filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        print(f"✅ Saved: {filename}")

    except Exception as e:
        print(f"❌ Scene {scene_number} failed: {e}")


def main():
    print("🎬 Starting GPT-4o Scene Generation + Image Creation")

    all_scenes = load_prompts(PROMPTS_PATH)

    # Normal scenes
    scenes = [s for s in all_scenes if s.get("scene_number") != 99]
    thumbnail = next((s for s in all_scenes if s.get("scene_number") == 99), None)

    for scene_data in scenes:
        prompt_raw = scene_data.get("prompt", "")
        scene_number = scene_data.get("scene_number", 0)
        if not prompt_raw:
            continue

        final_prompt = optimize_prompt(prompt_raw, scene_data)
        generate_image(final_prompt, scene_number)
        time.sleep(2)  # Respect rate limits

    # Thumbnail
    if thumbnail:
        print("\n🖼️ Generating thumbnail...")
        prompt_raw = thumbnail.get("prompt", "")
        if prompt_raw:
            final_prompt = optimize_prompt(prompt_raw, thumbnail)
            generate_image(final_prompt, 99)


if __name__ == "__main__":
    main()
