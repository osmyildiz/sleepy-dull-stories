import requests
import time
import os
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()


class ImprovedPiapiThumbnail:
    def __init__(self):
        self.api_key = os.getenv('PIAPI_KEY')
        self.base_url = "https://api.piapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # SLEEPY DULL STORIES SAFE & FULL BANNER - KISA VE ÖZ PROMPT
        self.templates = {
            # Safe area için kısa prompt (sadece text area)
            100: "Sleepy Dull Stories YouTube banner, elegant serif typography, dreamy night background, soft Studio Ghibli style, navy blue and gold colors, ancient books and stars",

            # Full background için detaylı prompt
            101: "YouTube banner background without text, dreamy night sky, ancient castle silhouettes, floating books, pocket watches, moon phases, owls, Studio Ghibli painterly style, navy midnight blue faded gold lavender colors, cinematic lighting"
        }

        self.cinematic_boosters = [
            "epic movie poster style",
            "award winning illustration",
            "matte painting aesthetic",
            "storybook art style"
        ]

        self.lighting_enhancers = [
            "dreamy soft lighting",
            "twilight atmosphere",
            "moonlit glow",
            "cinematic lighting"
        ]

    def enhance_prompt(self, base_prompt, topic_id, aspect_ratio="16:9"):
        import random
        booster = random.choice(self.cinematic_boosters)
        lighting = random.choice(self.lighting_enhancers)

        # DÜZELTME: Virgülden sonra boşluk ekledik ve parametreleri ayırdık
        enhanced = f"{base_prompt}, {booster}, {lighting} --ar {aspect_ratio} --v 6"

        # Prompt uzunluk kontrolü
        if len(enhanced) > 250:  # Biraz daha kısa tutuyoruz
            enhanced = f"{base_prompt}, {booster} --ar {aspect_ratio} --v 6"

        print(f"\n🧪 Prompt (ID {topic_id}, AR {aspect_ratio}):\n{enhanced}\n")
        return enhanced

    def submit_task_with_retry(self, prompt, aspect_ratio, max_retries=3):
        for attempt in range(max_retries):
            print(f"\n🚀 Attempt {attempt + 1}: Submitting prompt...\n{prompt}\n")

            # DÜZELTME: aspect_ratio'yu payload içinde doğru kullanıyoruz
            payload = {
                "model": "midjourney",
                "task_type": "imagine",
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,  # Bu önemli!
                    "process_mode": "relax"
                }
            }

            try:
                response = requests.post(
                    f"{self.base_url}/task",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                print(f"🔁 Response Status: {response.status_code}")
                print(f"📩 Response Body: {response.text}")

                if response.status_code == 200 and response.json().get("code") == 200:
                    return response.json()["data"]["task_id"]

            except Exception as e:
                print(f"❌ Exception occurred: {e}")

            time.sleep(10)
        return None

    def check_status(self, task_id):
        try:
            response = requests.get(f"{self.base_url}/task/{task_id}", headers=self.headers, timeout=10)
            if response.status_code == 200 and response.json().get("code") == 200:
                data = response.json()["data"]
                if data.get("status") == "completed":
                    return {"status": "completed", "urls": data.get("output", {}).get("temporary_image_urls", [])}
                elif data.get("status") == "failed":
                    return {"status": "failed"}
                else:
                    return {"status": "processing"}
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return {"status": "error"}

    def download_image(self, url, filename):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'https://discord.com/'
            }
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {filename}")
                return True
            else:
                print(f"❌ Download failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Download error: {e}")
        return False

    def generate_banner_versions(self, topic_id, aspect_ratio, versions=1, output_dir="banners"):
        Path(output_dir).mkdir(exist_ok=True)

        base_prompt = self.templates.get(topic_id)
        if not base_prompt:
            print(f"Template not found for topic {topic_id}")
            return False

        for i in range(versions):
            prompt = self.enhance_prompt(base_prompt, topic_id, aspect_ratio)
            task_id = self.submit_task_with_retry(prompt, aspect_ratio)  # aspect_ratio'yu geçiyoruz

            if not task_id:
                print("Task submission failed")
                continue

            # Status kontrolü
            for check_count in range(30):  # 10 dakika bekleme
                result = self.check_status(task_id)

                if result["status"] == "completed" and result["urls"]:
                    url = result["urls"][0]
                    filename = f"{output_dir}/topic_{topic_id}_{aspect_ratio.replace(':', 'x')}_v{i + 1}.png"
                    if self.download_image(url, filename):
                        print(f"✅ Successfully generated: {filename}")
                        return True
                    break

                elif result["status"] == "failed":
                    print("❌ Generation failed for task.")
                    break

                else:
                    print(f"⏳ Processing... ({check_count + 1}/30)")

                time.sleep(20)

        return False


if __name__ == "__main__":
    generator = ImprovedPiapiThumbnail()

    print("🎨 YouTube Banner Generator for Sleepy Dull Stories")
    print("=" * 50)

    # 1. Safe area banner için compact design (text odaklı)
    print("\n📏 Generating SAFE AREA banner (compact design)...")
    generator.generate_banner_versions(
        topic_id=100,  # Safe area template
        aspect_ratio="32:9",
        versions=1
    )

    # 2. Full background banner (text olmadan)
    print("\n🖼️ Generating BACKGROUND banner (full design without text)...")
    generator.generate_banner_versions(
        topic_id=101,  # Background template
        aspect_ratio="16:9",
        versions=1
    )

    print("\n✨ İki dosyayı Illustrator'da birleştirip safe area'ya text ekleyebilirsiniz!")