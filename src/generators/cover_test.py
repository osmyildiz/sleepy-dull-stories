import asyncio
import base64
import json
import os
from pathlib import Path
from runway_generator import OfficialRunwayGenerator, SDK_AVAILABLE, RUNWAY_API_SECRET

# === Ayarlar ===
STORY_ID = 1
BASE_DIR = Path("../../output") / str(STORY_ID)
PROMPT_PATH = BASE_DIR / "prompt.json"
REFERENCE_IMAGE_PATH = Path("../../data/media/reference_cover_style.jpg")
OUTPUT_DIR = BASE_DIR / "runway_visuals"
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "cover_test_right.jpg"


def build_right_side_prompt(title: str) -> str:
    return (
        "Right side of YouTube thumbnail. Classical oil painting of a Roman citizen in emotional awe. "
        "Mount Vesuvius erupting in the background. Ruins, smoke, volcanic fire. "
        "Dramatic chiaroscuro lighting, cinematic composition, 16:9 framing. No text."
    )


def load_image_base64(path: Path) -> str:
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


async def main():
    if not SDK_AVAILABLE:
        print("❌ RunwayML SDK yüklü değil.")
        return
    if not RUNWAY_API_SECRET:
        print("❌ .env dosyasında RUNWAYML_API_SECRET tanımlı değil.")
        return
    if not PROMPT_PATH.exists():
        print(f"❌ prompt.json bulunamadı: {PROMPT_PATH}")
        return
    if not REFERENCE_IMAGE_PATH.exists():
        print(f"❌ Referans görsel bulunamadı: {REFERENCE_IMAGE_PATH}")
        return

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    clickbait_title = prompt_data.get("clickbait_title", "Untitled")

    visual_prompt = build_right_side_prompt(clickbait_title)
    ref_base64 = load_image_base64(REFERENCE_IMAGE_PATH)

    # RunwayML API için doğru format - uri ve tag ile
    reference_images = [{
        "uri": f"data:image/jpeg;base64,{ref_base64}",
        "tag": "reference_style"  # Tag opsiyonel, ama varsa prompt'ta @reference_style ile kullanılabilir
    }]

    generator = OfficialRunwayGenerator()

    print("🎨 Görsel üretimi başlatılıyor...")

    try:
        task = await generator.async_client.text_to_image.create(
            model="gen4_image",  # Doğru model adı
            prompt_text=visual_prompt,
            ratio="1280:720",
            reference_images=reference_images
        )
    except Exception as e:
        print(f"❌ Task oluşturma hatası: {e}")
        # Eğer reference_images sorunu devam ederse, referans olmadan dene
        print("🔄 Referans görsel olmadan deneniyor...")
        try:
            task = await generator.async_client.text_to_image.create(
                model="gen4_image",  # Doğru model adı
                prompt_text=visual_prompt,
                ratio="1280:720"
            )
        except Exception as e2:
            print(f"❌ Referanssız task de başarısız: {e2}")
            return

    if not task:
        print("❌ Task oluşturulamadı.")
        return

    image_url, error = await generator.wait_for_completion(task.id, "image", max_wait=300)
    if not image_url:
        print(f"❌ Görsel üretimi başarısız: {error}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success, download_error, file_size = await generator.download_real_file(image_url, str(OUTPUT_IMAGE_PATH))
    if success:
        print(f"\n✅ Görsel başarıyla indirildi: {OUTPUT_IMAGE_PATH} ({file_size:.2f} MB)")
    else:
        print(f"❌ İndirme hatası: {download_error}")


if __name__ == "__main__":
    asyncio.run(main())