import asyncio
import base64
import json
import os
from pathlib import Path
from runway_generator import OfficialRunwayGenerator, SDK_AVAILABLE, RUNWAY_API_SECRET

# === Settings ===
STORY_ID = 1
BASE_DIR = Path("../../output") / str(STORY_ID)
PROMPT_PATH = BASE_DIR / "prompt.json"
REFERENCE_ARCHITECTURE_PATH = Path("../../data/media/reference_architecture.jpg")
REFERENCE_PORTRAIT_PATH = Path("../../data/media/reference_classical_portrait.jpg")
OUTPUT_DIR = BASE_DIR / "runway_visuals"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "teaser_video.mp4"

def load_image_base64(path: Path) -> str:
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

async def main():
    if not SDK_AVAILABLE:
        print("❌ RunwayML SDK not installed.")
        return
    if not RUNWAY_API_SECRET:
        print("❌ RUNWAYML_API_SECRET not defined in .env file.")
        return
    if not PROMPT_PATH.exists():
        print(f"❌ prompt.json not found: {PROMPT_PATH}")
        return

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    teaser_video_prompt = prompt_data.get("teaser_video", "")
    if not teaser_video_prompt:
        print("❌ 'teaser_video' field not found in prompt.json.")
        return

    # Prepare primary reference image
    primary_image = None
    if REFERENCE_ARCHITECTURE_PATH.exists():
        primary_image = load_image_base64(REFERENCE_ARCHITECTURE_PATH)
    elif REFERENCE_PORTRAIT_PATH.exists():
        primary_image = load_image_base64(REFERENCE_PORTRAIT_PATH)

    if not primary_image:
        print("❌ No valid reference images found.")
        return

    data_uri = f"data:image/jpeg;base64,{primary_image}"
    generator = OfficialRunwayGenerator()

    print("🎬 Starting video generation...")
    print(f"📝 Prompt: {teaser_video_prompt}")

    try:
        # Using supported ratio (1280:720 for 16:9 aspect ratio)
        task = await generator.async_client.image_to_video.create(
            model='gen4_turbo',
            prompt_image=data_uri,
            prompt_text=teaser_video_prompt,
            ratio='1280:720',  # Desteklenen oranlardan biri
            duration=10,
            seed=42
        )
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return

    print(f"⏳ Task ID: {task.id}")
    print("⏳ Waiting for video generation to complete...")

    video_url, error = await generator.wait_for_completion(task.id, "video", max_wait=600)
    if not video_url:
        print(f"❌ Video generation failed: {error}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success, download_error, file_size = await generator.download_real_file(video_url, str(OUTPUT_VIDEO_PATH))
    if success:
        print(f"\n✅ Video successfully downloaded: {OUTPUT_VIDEO_PATH} ({file_size:.2f} MB)")
    else:
        print(f"❌ Download error: {download_error}")

if __name__ == "__main__":
    asyncio.run(main())