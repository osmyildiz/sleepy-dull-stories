import requests
import time
import os
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()


class YouTubeLogoGenerator:
    def __init__(self):
        self.api_key = os.getenv('PIAPI_KEY')
        self.base_url = "https://api.piapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # LOGO TEMPLATES - Daha kısa ve odaklı promptlar
        self.logo_templates = {
            # Ana konsept: Şato kulesi + asılı objeler
            200: "YouTube logo circular design, medieval castle tower center, hanging pocket watches books lanterns with golden chains, navy blue gold colors, dreamy night atmosphere",

            # Sadece asılı objeler odaklı
            201: "Circular logo, hanging vintage objects around center, antique pocket watches old books vintage lanterns suspended by chains, navy blue gold scheme, magical atmosphere",

            # Şato silueti + objeler
            202: "YouTube logo castle silhouette center, surrounded by hanging objects floating books pocket watches vintage lamps, circular badge, navy blue gold colors, mystical theme",

            # Tek kule + objeler konsepti
            203: "Single castle tower logo, tall spire center, vintage objects hanging around edges books lanterns pocket watches, suspended by golden threads, navy blue warm gold palette"
        }

        self.style_enhancers = [
            "magical hanging objects",
            "suspended vintage items",
            "floating antique elements",
            "dangling decorative pieces"
        ]

        self.quality_boosters = [
            "Studio Ghibli inspired",
            "dreamy atmosphere",
            "mystical night theme",
            "enchanted castle aesthetic"
        ]

    def enhance_logo_prompt(self, base_prompt, logo_id):
        import random
        style = random.choice(self.style_enhancers)
        quality = random.choice(self.quality_boosters)

        # DÜZELTME: Midjourney parametrelerini kaldırdık
        enhanced = f"{base_prompt}, {style}, {quality}"

        # Prompt uzunluk kontrolü
        if len(enhanced) > 250:
            enhanced = f"{base_prompt}, {style}"

        print(f"\n🎨 Logo Prompt (ID {logo_id}):\n{enhanced}\n")
        return enhanced

    def submit_logo_task(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            print(f"\n🚀 Logo Generation Attempt {attempt + 1}...")

            payload = {
                "model": "midjourney",
                "task_type": "imagine",
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",  # Square format for logo
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

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("code") == 200:
                        task_id = response_data["data"]["task_id"]
                        print(f"✅ Task submitted successfully: {task_id}")
                        return task_id
                    else:
                        print(f"❌ API Error: {response_data}")
                else:
                    print(f"❌ HTTP Error: {response.status_code}")
                    print(f"Response: {response.text}")

            except Exception as e:
                print(f"❌ Exception: {e}")

            time.sleep(10)

        return None

    def check_task_status(self, task_id):
        try:
            response = requests.get(
                f"{self.base_url}/task/{task_id}",
                headers=self.headers,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 200:
                    task_data = data["data"]
                    status = task_data.get("status")

                    if status == "completed":
                        urls = task_data.get("output", {}).get("temporary_image_urls", [])
                        return {"status": "completed", "urls": urls}
                    elif status == "failed":
                        return {"status": "failed", "error": task_data.get("error")}
                    else:
                        progress = task_data.get("output", {}).get("progress", 0)
                        return {"status": "processing", "progress": progress}

        except Exception as e:
            print(f"❌ Status check error: {e}")

        return {"status": "error"}

    def download_all_variants(self, urls, logo_id, output_dir="logos"):
        """4 varyantı da indir"""
        Path(output_dir).mkdir(exist_ok=True)
        downloaded_files = []

        for i, url in enumerate(urls[:4]):  # En fazla 4 varyant
            try:
                print(f"⬇️ Downloading variant {i + 1}/4...")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://discord.com/'
                }

                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 200:
                    filename = f"{output_dir}/logo_{logo_id}_variant_{i + 1}.png"

                    with open(filename, 'wb') as f:
                        f.write(response.content)

                    # Dosya boyutunu kontrol et
                    file_size = os.path.getsize(filename)
                    file_size_mb = file_size / (1024 * 1024)

                    print(f"✅ Downloaded: {filename} ({file_size_mb:.2f} MB)")

                    if file_size_mb > 4:
                        print(f"⚠️ Warning: File size exceeds 4MB YouTube limit!")

                    downloaded_files.append(filename)

                else:
                    print(f"❌ Download failed: HTTP {response.status_code}")

            except Exception as e:
                print(f"❌ Download error for variant {i + 1}: {e}")

            time.sleep(2)  # Rate limiting

        return downloaded_files

    def generate_logo_set(self, logo_id, output_dir="logos"):
        """Belirtilen logo template'ini oluştur"""

        base_prompt = self.logo_templates.get(logo_id)
        if not base_prompt:
            print(f"❌ Logo template {logo_id} not found!")
            return False

        print(f"🎨 Generating logo set for template {logo_id}...")

        # Prompt'u enhance et
        enhanced_prompt = self.enhance_logo_prompt(base_prompt, logo_id)

        # Task submit et
        task_id = self.submit_logo_task(enhanced_prompt)
        if not task_id:
            print("❌ Failed to submit logo generation task")
            return False

        # Status monitoring
        print("⏳ Waiting for logo generation...")
        for check_count in range(40):  # 13+ dakika bekleme
            time.sleep(20)

            result = self.check_task_status(task_id)
            status = result["status"]

            if status == "completed":
                urls = result.get("urls", [])
                if urls:
                    print(f"🎉 Logo generation completed! Found {len(urls)} variants")

                    # Tüm varyantları indir
                    downloaded = self.download_all_variants(urls, logo_id, output_dir)

                    if downloaded:
                        print(f"✅ Successfully downloaded {len(downloaded)} logo variants:")
                        for file in downloaded:
                            print(f"  📁 {file}")
                        return True
                    else:
                        print("❌ Failed to download any variants")
                        return False
                else:
                    print("❌ No image URLs in completed task")
                    return False

            elif status == "failed":
                error = result.get("error", "Unknown error")
                print(f"❌ Logo generation failed: {error}")
                return False

            else:
                progress = result.get("progress", 0)
                print(f"⏳ Processing... {progress}% ({check_count + 1}/40)")

        print("❌ Timeout waiting for logo generation")
        return False

    def generate_all_logo_concepts(self):
        """Tüm logo konseptlerini oluştur"""

        print("🎨 YouTube Logo Generator for Sleepy Dull Stories")
        print("=" * 60)

        success_count = 0

        for logo_id in self.logo_templates.keys():
            print(f"\n🎯 Generating Logo Concept {logo_id}...")

            if self.generate_logo_set(logo_id):
                success_count += 1
                print(f"✅ Logo concept {logo_id} completed successfully!")
            else:
                print(f"❌ Logo concept {logo_id} failed!")

            # Konseptler arası bekleme
            print("⏸️ Waiting before next concept...")
            time.sleep(30)

        print(f"\n🎉 Logo generation completed!")
        print(f"📊 Success rate: {success_count}/{len(self.logo_templates)} concepts")

        if success_count > 0:
            print("\n📁 Check the 'logos' folder for all generated variants!")
            print("💡 Choose the best one for your YouTube channel!")


def main():
    generator = YouTubeLogoGenerator()

    # API key kontrolü
    if not generator.api_key:
        print("❌ PIAPI_KEY environment variable not found!")
        print("Please set your API key in .env file")
        return

    print("🎯 Starting YouTube logo generation...")
    print("📋 Will generate 4 different logo concepts")
    print("🔢 Each concept will have 4 variants (total ~16 logos)")
    print("⏱️ Estimated time: 20-30 minutes")

    input("\n▶️ Press Enter to start generation...")

    generator.generate_all_logo_concepts()


if __name__ == "__main__":
    main()