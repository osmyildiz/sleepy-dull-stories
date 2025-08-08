"""
Simple Social Media Visual Generator
6 görsel oluşturur (thumbnail + intro + 4 story visual + outro)
Her görsel için 4 varyasyon indirir
"""

import requests
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import logging

load_dotenv()

class SimpleSocialMediaGenerator:
    """Basit sosyal medya görsel oluşturucu"""

    def __init__(self):
        self.api_key = self.get_api_key()
        self.base_url = "https://api.piapi.ai/api/v1"

        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        self.api_calls_made = 0
        self.successful_downloads = 0

        print("🚀 Simple Social Media Generator initialized")
        print(f"🔑 API Key: {self.api_key[:8]}...")

    def get_api_key(self):
        """API key al"""
        api_key = (
            os.getenv('PIAPI_KEY') or
            os.getenv('MIDJOURNEY_API_KEY') or
            os.getenv('PIAPI_API_KEY')
        )

        if not api_key:
            raise ValueError("❌ PIAPI_KEY gerekli!")

        print("✅ API key loaded")
        return api_key

    def load_social_media_json(self, json_path: str) -> dict:
        """Social media JSON'u yükle"""
        print(f"📂 Loading JSON: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("✅ JSON loaded successfully")
        return data

    def extract_visual_prompts(self, data: dict) -> list:
        """JSON'dan tüm visual prompt'ları çıkar"""
        print("🎨 Extracting visual prompts...")

        universal = data.get("universal_60s_template", {})
        visuals = []

        # 1. Thumbnail
        thumbnail_prompt = universal.get("thumbnail_design", {}).get("main_visual_prompt", "")
        if thumbnail_prompt:
            visuals.append({
                "name": "thumbnail",
                "prompt": thumbnail_prompt,
                "type": "thumbnail"
            })

        # 2. Intro
        intro_prompt = universal.get("intro_section", {}).get("visual_prompt", "")
        if intro_prompt:
            visuals.append({
                "name": "intro",
                "prompt": intro_prompt,
                "type": "intro"
            })

        # 3. Story visuals (4 tane)
        story_sequence = universal.get("story_section", {}).get("visual_sequence", [])
        for i, story_visual in enumerate(story_sequence, 1):
            story_prompt = story_visual.get("visual_prompt", "")
            if story_prompt:
                visuals.append({
                    "name": f"story_{i}",
                    "prompt": story_prompt,
                    "type": "story",
                    "timing": story_visual.get("timing", ""),
                    "sync": story_visual.get("story_sync", "")
                })

        # 4. Outro
        outro_prompt = universal.get("outro_section", {}).get("visual_prompt", "")
        if outro_prompt:
            visuals.append({
                "name": "outro",
                "prompt": outro_prompt,
                "type": "outro"
            })

        print(f"✅ Extracted {len(visuals)} visual prompts:")
        for visual in visuals:
            print(f"   📎 {visual['name']}: {visual['prompt'][:60]}...")

        return visuals

    def enhance_prompt_for_social_media(self, prompt: str, visual_type: str) -> str:
        """Prompt'u sosyal medya için optimize et"""

        # Temel sosyal medya optimizasyonları
        social_additions = [
            "9:16 aspect ratio",
            "mobile optimized",
            "high quality",
            "cinematic lighting",
            "professional photography"
        ]

        # Tip'e göre özel eklemeler
        if visual_type == "thumbnail":
            social_additions.extend([
                "thumbnail worthy",
                "eye catching",
                "dramatic composition"
            ])
        elif visual_type == "intro":
            social_additions.extend([
                "establishing shot",
                "cinematic opening",
                "engaging intro"
            ])
        elif visual_type == "story":
            social_additions.extend([
                "storytelling visual",
                "narrative scene",
                "emotional connection"
            ])
        elif visual_type == "outro":
            social_additions.extend([
                "satisfying conclusion",
                "peaceful ending",
                "call to action ready"
            ])

        # V7 parametreleri ekle
        enhanced_prompt = f"{prompt}, {', '.join(social_additions)}, --ar 9:16 --v 7.0"

        return enhanced_prompt

    def clean_prompt_for_api(self, prompt: str) -> str:
        """API için prompt'u temizle"""
        import re

        # --ar ve --v parametrelerini çıkar (API kendi ekliyor)
        prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt)
        prompt = re.sub(r'--v\s+[\d.]+', '', prompt)
        prompt = re.sub(r'--\w+(?:\s+[\w:.]+)?', '', prompt)

        # Fazla boşlukları temizle
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip()

        return prompt

    def submit_midjourney_task(self, prompt: str, visual_name: str) -> str:
        """Midjourney task gönder"""

        # Prompt'u temizle ve optimize et
        cleaned_prompt = self.clean_prompt_for_api(prompt)

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": cleaned_prompt,
                "aspect_ratio": "9:16",
                "process_mode": "relax"
            }
        }

        print(f"🎬 Submitting {visual_name}...")
        print(f"   📝 Prompt: {cleaned_prompt[:100]}...")

        try:
            self.api_calls_made += 1
            response = requests.post(f"{self.base_url}/task", headers=self.headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_id = result.get("data", {}).get("task_id")
                    print(f"✅ {visual_name}: Task submitted - {task_id}")
                    return task_id
                else:
                    print(f"❌ {visual_name}: API Error - {result.get('message', 'Unknown')}")
                    return None
            else:
                print(f"❌ {visual_name}: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ {visual_name}: Request failed - {e}")
            return None

    def check_task_status(self, task_id: str, visual_name: str) -> dict:
        """Task durumunu kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/task/{task_id}", headers=self.headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    status = task_data.get("status", "").lower()

                    if status == "completed":
                        output = task_data.get("output", {})
                        temp_urls = output.get("temporary_image_urls", [])
                        main_url = output.get("image_url", "")

                        # Tüm URL'leri topla
                        all_urls = []
                        if temp_urls:
                            all_urls.extend(temp_urls)
                        if main_url and main_url not in all_urls:
                            all_urls.append(main_url)

                        if all_urls:
                            return {
                                "status": "completed",
                                "urls": all_urls,
                                "count": len(all_urls)
                            }
                        else:
                            return {"status": "completed_no_urls"}

                    elif status == "failed":
                        error_msg = task_data.get("error", {}).get("message", "Unknown error")
                        return {"status": "failed", "error": error_msg}

                    else:
                        return {"status": "processing"}

            return {"status": "error", "error": "HTTP error"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def download_all_variations(self, urls: list, visual_name: str, output_dir: Path) -> int:
        """Tüm varyasyonları indir"""
        downloaded_count = 0

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://discord.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }

        print(f"📥 {visual_name}: Downloading {len(urls)} variations...")

        for i, url in enumerate(urls):
            if i >= 4:  # Maksimum 4 varyasyon
                break

            # Dosya adı: thumbnail_1.png, thumbnail_2.png, etc.
            if i == 0:
                filename = f"{visual_name}.png"
            else:
                filename = f"{visual_name}_v{i+1}.png"

            save_path = output_dir / filename

            try:
                response = requests.get(url, headers=headers, timeout=30, stream=True)

                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    file_size = os.path.getsize(save_path)
                    print(f"✅ {visual_name} v{i+1}: Downloaded ({file_size} bytes)")
                    downloaded_count += 1
                    self.successful_downloads += 1

                    # Metadata kaydet
                    metadata = {
                        "visual_name": visual_name,
                        "variation": i + 1,
                        "url": url,
                        "downloaded_at": datetime.now().isoformat(),
                        "file_size": file_size
                    }

                    json_path = save_path.with_suffix('.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                else:
                    print(f"❌ {visual_name} v{i+1}: Download failed HTTP {response.status_code}")

            except Exception as e:
                print(f"❌ {visual_name} v{i+1}: Download error - {e}")

        print(f"🎉 {visual_name}: Downloaded {downloaded_count}/{len(urls)} variations")
        return downloaded_count

    def generate_social_media_visuals(self, json_path: str, output_dir: str):
        """Ana generation fonksiyonu"""
        print("🚀" * 50)
        print("SIMPLE SOCIAL MEDIA VISUAL GENERATOR")
        print("🎬 6 Visual Generation (Thumbnail + Intro + 4 Story + Outro)")
        print("📱 4 Variations per Visual")
        print("🖼️ Total: 24 Images Expected")
        print("🚀" * 50)

        # Setup
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # JSON yükle ve prompts çıkar
        data = self.load_social_media_json(json_path)
        visual_prompts = self.extract_visual_prompts(data)

        if not visual_prompts:
            print("❌ No visual prompts found!")
            return False

        print(f"🎯 Generating {len(visual_prompts)} visuals...")

        # Task'ları gönder
        tasks = {}

        for visual in visual_prompts:
            visual_name = visual["name"]
            visual_type = visual["type"]
            original_prompt = visual["prompt"]

            # Prompt'u sosyal medya için optimize et
            enhanced_prompt = self.enhance_prompt_for_social_media(original_prompt, visual_type)

            print(f"\n🎬 Processing {visual_name}...")

            # Task gönder
            task_id = self.submit_midjourney_task(enhanced_prompt, visual_name)

            if task_id:
                tasks[visual_name] = {
                    "task_id": task_id,
                    "visual_data": visual,
                    "enhanced_prompt": enhanced_prompt
                }
                print(f"✅ {visual_name}: Task queued")
            else:
                print(f"❌ {visual_name}: Task submission failed")

            # Rate limiting
            time.sleep(5)

        if not tasks:
            print("❌ No tasks submitted!")
            return False

        print(f"\n⏳ Monitoring {len(tasks)} tasks...")

        # Task'ları izle
        completed = {}
        max_cycles = 50

        for cycle in range(max_cycles):
            if not tasks:
                break

            print(f"📊 Cycle {cycle + 1}: {len(completed)}/{len(completed) + len(tasks)} completed")

            tasks_to_remove = []

            for visual_name, task_data in tasks.items():
                task_id = task_data["task_id"]

                result = self.check_task_status(task_id, visual_name)

                if result["status"] == "completed":
                    if "urls" in result:
                        print(f"✅ {visual_name}: Task completed!")
                        completed[visual_name] = {
                            "urls": result["urls"],
                            "task_data": task_data
                        }
                        tasks_to_remove.append(visual_name)
                    else:
                        print(f"⚠️ {visual_name}: Completed but no URLs")
                        tasks_to_remove.append(visual_name)

                elif result["status"] == "failed":
                    print(f"❌ {visual_name}: Task failed - {result.get('error', 'Unknown')}")
                    tasks_to_remove.append(visual_name)

                elif result["status"] == "error":
                    print(f"⚠️ {visual_name}: Status check error - {result.get('error', 'Unknown')}")

            for visual_name in tasks_to_remove:
                del tasks[visual_name]

            if not tasks:
                break

            time.sleep(30)

        # Download completed visuals
        if not completed:
            print("❌ No visuals completed!")
            return False

        print(f"\n📥 Downloading {len(completed)} completed visuals...")

        total_downloaded = 0

        for visual_name, visual_data in completed.items():
            urls = visual_data["urls"]
            downloaded_count = self.download_all_variations(urls, visual_name, output_path)
            total_downloaded += downloaded_count

        # Final report
        report = {
            "generation_completed": datetime.now().isoformat(),
            "total_visuals_requested": len(visual_prompts),
            "total_visuals_completed": len(completed),
            "total_images_downloaded": total_downloaded,
            "api_calls_made": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "output_directory": str(output_path),
            "visuals_generated": list(completed.keys())
        }

        report_path = output_path / "generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 GENERATION COMPLETED!")
        print(f"✅ Visuals completed: {len(completed)}/{len(visual_prompts)}")
        print(f"📥 Images downloaded: {total_downloaded}")
        print(f"📁 Output directory: {output_path}")
        print(f"📊 Report saved: {report_path}")

        return len(completed) >= len(visual_prompts) * 0.8  # 80% success rate

def main():
    """Ana fonksiyon"""

    # Kullanım
    json_path = "social_media_content.json"  # JSON dosya yolu
    output_dir = "social_media_output"        # Çıktı klasörü

    try:
        generator = SimpleSocialMediaGenerator()
        success = generator.generate_social_media_visuals(json_path, output_dir)

        if success:
            print("🎊 Social media visual generation successful!")
        else:
            print("⚠️ Some visuals failed to generate")

    except Exception as e:
        print(f"💥 Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()