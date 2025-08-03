import requests
import time
import os
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()


class SleepyDullStoriesThumbnailGenerator:
    def __init__(self):
        self.api_key = os.getenv('PIAPI_KEY')
        self.base_url = "https://api.piapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # JSON dosyasından prompt'ları yükle
        self.load_prompts()

    def load_prompts(self):
        """thumbnail_prompts.json'dan prompt'ları yükle"""
        try:
            with open('src/data/thumbnail_prompts.json', 'r', encoding='utf-8') as f:
                self.prompts_data = json.load(f)
                self.prompts = self.prompts_data['prompts']
            print(f"✅ {len(self.prompts)} thumbnail prompt loaded")
        except Exception as e:
            print(f"❌ JSON load error: {e}")
            self.prompts = {}

    def enhance_prompt_v7(self, base_prompt):
        """V7 için optimize edilmiş prompt iyileştirme - Parametre çakışması düzeltmesi"""
        import random

        # V7 için kısa booster'lar
        boosters = [
            "dramatic lighting",
            "cinematic composition",
            "epic scale",
            "movie poster style",
            "intense atmosphere"
        ]

        booster = random.choice(boosters)

        # Base prompt'taki --ar 16:9'u çıkar (çakışmayı önlemek için)
        clean_prompt = base_prompt.replace(" --ar 16:9", "")

        # V7 optimized prompt - TEK parametre seti
        enhanced = f"{clean_prompt}, {booster}, photorealistic --ar 16:9 --v 7.0"

        print(f"🎬 Enhanced V7 prompt:")
        print(f"📝 {enhanced}")
        print(f"📊 Length: {len(enhanced)} characters")

        return enhanced

    def submit_midjourney_task(self, prompt, max_retries=3):
        """Midjourney task'ı PIAPI'ye gönder - Debug bilgisiyle"""

        for attempt in range(max_retries):
            print(f"🚀 Submitting to Midjourney V7 (attempt {attempt + 1}/{max_retries})...")

            payload = {
                "model": "midjourney",
                "task_type": "imagine",
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "process_mode": "relax"
                }
            }

            # DEBUG: Payload'u göster
            print(f"📤 DEBUG Payload:")
            print(f"   🎯 Model: {payload['model']}")
            print(f"   🎯 Task Type: {payload['task_type']}")
            print(f"   🎯 Prompt: {payload['input']['prompt']}")
            print(f"   🎯 Aspect Ratio: {payload['input']['aspect_ratio']}")
            print(f"   🎯 Process Mode: {payload['input']['process_mode']}")

            try:
                response = requests.post(
                    f"{self.base_url}/task",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                # DEBUG: Response detayları
                print(f"📥 DEBUG Response:")
                print(f"   🔢 Status Code: {response.status_code}")
                print(f"   📊 Headers: {dict(response.headers)}")

                try:
                    response_json = response.json()
                    print(f"   📋 JSON Response: {response_json}")
                except:
                    print(f"   📋 Raw Response: {response.text}")

                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 200:
                        task_id = result["data"]["task_id"]
                        print(f"✅ Task submitted successfully: {task_id}")
                        return task_id
                    else:
                        print(f"❌ API Error Details:")
                        print(f"   📋 Code: {result.get('code')}")
                        print(f"   📋 Message: {result.get('message')}")
                        print(f"   📋 Full Response: {result}")

                elif response.status_code == 500:
                    print(f"❌ HTTP 500 Server Error Details:")
                    try:
                        error_json = response.json()
                        print(f"   📋 Error JSON: {error_json}")
                    except:
                        print(f"   📋 Raw Error: {response.text}")

                    wait_time = (attempt + 1) * 30
                    print(f"⚠️ Rate limit detected - Waiting {wait_time}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue

                elif response.status_code == 429:
                    print(f"❌ HTTP 429 Rate Limit Details:")
                    try:
                        error_json = response.json()
                        print(f"   📋 Rate Limit JSON: {error_json}")
                    except:
                        print(f"   📋 Raw Rate Limit: {response.text}")

                    wait_time = (attempt + 1) * 60
                    print(f"⚠️ Official rate limit - Waiting {wait_time}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue

                else:
                    print(f"❌ HTTP Error {response.status_code} Details:")
                    try:
                        error_json = response.json()
                        print(f"   📋 Error JSON: {error_json}")
                    except:
                        print(f"   📋 Raw Error: {response.text}")

                    if attempt < max_retries - 1:
                        print(f"⚠️ Retrying in 15 seconds...")
                        time.sleep(15)
                        continue

            except requests.exceptions.Timeout:
                print(f"❌ Request Timeout (30s)")
                if attempt < max_retries - 1:
                    print(f"⚠️ Retrying in 10 seconds...")
                    time.sleep(10)
                    continue

            except requests.exceptions.ConnectionError:
                print(f"❌ Connection Error")
                if attempt < max_retries - 1:
                    print(f"⚠️ Retrying in 10 seconds...")
                    time.sleep(10)
                    continue

            except Exception as e:
                print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    print(f"⚠️ Retrying in 10 seconds...")
                    time.sleep(10)
                    continue

        print(f"❌ All {max_retries} attempts failed")
        return None

    def check_task_status(self, task_id):
        """Task durumunu kontrol et - Debug bilgisiyle"""
        try:
            response = requests.get(
                f"{self.base_url}/task/{task_id}",
                headers=self.headers,
                timeout=10
            )

            # DEBUG: Status response
            print(f"🔍 DEBUG Status Check:")
            print(f"   🔢 Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"   📋 Status Response: {result}")

                if result.get("code") == 200:
                    task_data = result["data"]
                    status = task_data.get("status", "").lower()

                    print(f"   🎯 Task Status: '{status}'")
                    print(f"   📊 Progress: {task_data.get('progress', 0)}%")

                    if status == "completed":
                        output = task_data.get("output", {})
                        temp_urls = output.get("temporary_image_urls", [])
                        image_url = output.get("image_url", "")

                        print(f"   ✅ Completed! Found URLs:")
                        print(f"      📷 Temp URLs: {len(temp_urls) if temp_urls else 0}")
                        print(f"      📷 Image URL: {'Yes' if image_url else 'No'}")

                        # Midjourney genelde 4 varyasyon döndürür
                        if temp_urls and len(temp_urls) >= 4:
                            return {"status": "completed", "urls": temp_urls[:4]}
                        elif image_url:
                            return {"status": "completed", "urls": [image_url]}
                        else:
                            return {"status": "completed", "urls": []}

                    elif status == "failed":
                        error_info = task_data.get("error", {})
                        print(f"   ❌ Failed! Error info:")
                        print(f"      📋 Error: {error_info}")
                        return {"status": "failed"}

                    elif status in ["pending", "processing", "in_progress"]:
                        print(f"   ⏳ Still processing...")
                        return {"status": "processing"}

                    else:
                        print(f"   ⚠️ Unknown status: '{status}'")
                        return {"status": "processing"}  # Bilinmeyen durumları processing say

                else:
                    print(f"   ❌ API Error Code: {result.get('code')}")
                    print(f"   📋 API Message: {result.get('message')}")
                    return {"status": "error"}

            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   📋 Response: {response.text}")
                return {"status": "error"}

        except Exception as e:
            print(f"❌ Status check exception: {type(e).__name__}: {e}")
            return {"status": "error"}

    def download_thumbnail(self, url, filename):
        """Thumbnail'ı indir"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
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
                return False

        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

    def generate_thumbnail_variants(self, topic_id, output_dir="thumbnails"):
        """Belirli topic için 4 varyant thumbnail oluştur"""

        Path(output_dir).mkdir(exist_ok=True)

        # Topic ID'yi string'e çevir (JSON'da string olarak saklanıyor)
        topic_key = str(topic_id)

        if topic_key not in self.prompts:
            print(f"❌ Topic {topic_id} not found in prompts!")
            print(f"Available IDs: {list(self.prompts.keys())}")
            return False

        prompt_data = self.prompts[topic_key]
        base_prompt = prompt_data['prompt']
        topic_name = prompt_data['topic']

        print(f"\n🎨 GENERATING THUMBNAILS")
        print(f"🆔 Topic ID: {topic_id}")
        print(f"📝 Topic: {topic_name}")
        print(f"🎬 Base Prompt: {base_prompt}")
        print("=" * 60)

        # Enhanced prompt oluştur
        enhanced_prompt = self.enhance_prompt_v7(base_prompt)

        # Task gönder
        task_id = self.submit_midjourney_task(enhanced_prompt)
        if not task_id:
            print(f"❌ Task submission failed for topic {topic_id}")
            return False

        # Status monitoring
        print(f"⏳ Monitoring task: {task_id}")
        max_cycles = 40  # 20 dakika (40 * 30 saniye)

        for cycle in range(max_cycles):
            result = self.check_task_status(task_id)

            if result["status"] == "completed":
                urls = result["urls"]
                print(f"✅ Generation completed! Found {len(urls)} variants")

                # 4 varyantı indir
                successful_downloads = 0
                for i, url in enumerate(urls[:4], 1):
                    filename = f"{output_dir}/thumbnail_{topic_id}_variant_{i}.png"
                    if self.download_thumbnail(url, filename):
                        successful_downloads += 1

                print(f"🎉 Downloaded {successful_downloads}/4 variants for topic {topic_id}")
                return successful_downloads > 0

            elif result["status"] == "failed":
                print(f"❌ Generation failed for topic {topic_id}")
                return False

            else:
                if cycle % 4 == 0:  # Her 2 dakikada rapor
                    print(f"⏳ Processing... ({cycle + 1}/{max_cycles})")
                time.sleep(30)

        print(f"⏰ Timeout for topic {topic_id}")
        return False


def main():
    """Ana fonksiyon - Manuel topic ID girişi"""

    generator = SleepyDullStoriesThumbnailGenerator()

    if not generator.api_key:
        print("❌ PIAPI_KEY environment variable not found!")
        return

    if not generator.prompts:
        print("❌ No prompts loaded! Check thumbnail_prompts.json")
        return

    print("🎬 SLEEPY DULL STORIES THUMBNAIL GENERATOR")
    print("=" * 60)
    print("🎯 Midjourney V7 - 1920x1080 Thumbnails")
    print("🎨 4 Variants per topic")
    print("💾 Auto-download all variants")
    print("=" * 60)

    # Available topics göster
    print(f"📋 Available topics ({len(generator.prompts)}):")
    for topic_id, data in sorted(generator.prompts.items(), key=lambda x: int(x[0])):
        print(f"   🎭 {topic_id}: {data['topic']}")

    print("=" * 60)

    # MANUEL TOPIC GİRİŞİ - BURASI DEĞİŞTİRİLECEK
    topic_id = 31  # <-- BU SATIRDA TOPIC ID'Yİ DEĞİŞTİR

    print(f"🎯 Selected Topic: {topic_id}")

    # Topic ID'yi string'e çevir
    topic_key = str(topic_id)

    if topic_key in generator.prompts:
        topic_data = generator.prompts[topic_key]
        print(f"📝 Topic Name: {topic_data['topic']}")
        print(f"🎬 Prompt: {topic_data['prompt']}")

        success = generator.generate_thumbnail_variants(
            topic_id=topic_id,
            output_dir="thumbnails"
        )

        if success:
            print(f"\n🎉 SUCCESS! Topic {topic_id} thumbnails generated!")
            print(f"📁 Check 'thumbnails' folder for 4 variants")
            print(f"📱 Ready for YouTube upload!")
        else:
            print(f"\n❌ Failed to generate thumbnails for topic {topic_id}")

    else:
        print(f"❌ Topic {topic_id} not found!")
        print(f"Available IDs: {list(generator.prompts.keys())}")


if __name__ == "__main__":
    main()