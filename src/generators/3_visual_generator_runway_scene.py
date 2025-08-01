"""
Runway Official SDK Generator - Topic 1 Scene Generation
Uses official runwayml Python SDK (not HTTP requests)
Based on official documentation: https://docs.dev.runwayml.com/api-details/sdks/
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Official SDK import
try:
    from runwayml import RunwayML
    print("✅ Official RunwayML SDK imported")
except ImportError:
    print("❌ RunwayML SDK not found")
    print("💡 Install with: pip install runwayml")
    exit(1)

# Configuration
RUNWAY_CONFIG = {
    "api_key": os.getenv("RUNWAYML_API_SECRET"),
    "target_topic_id": 1,
    "output_dir": "../output/1/scenes",
    "max_retries": 3,
    "retry_delay": 10
}

class OfficialRunwayGenerator:
    """Official Runway SDK generator"""

    def __init__(self):
        self.api_key = RUNWAY_CONFIG["api_key"]
        if not self.api_key:
            raise ValueError("❌ RUNWAYML_API_SECRET not found in environment")

        # Initialize official client
        self.client = RunwayML(
            api_key=self.api_key
        )

        self.generation_log = []
        self.generated_images = []

        print("✅ Official Runway SDK initialized")

    def log_step(self, description: str, status: str = "START"):
        """Log steps"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {description}")

    def load_topic_1_data(self):
        """Load Topic 1 data"""
        self.log_step("Loading Topic 1 data")

        visual_prompts_path = "../output/1/visual_generation_prompts.json"
        character_profiles_path = "../output/1/character_profiles.json"

        try:
            with open(visual_prompts_path, 'r', encoding='utf-8') as f:
                visual_prompts = json.load(f)
            print(f"✅ Loaded {len(visual_prompts)} visual prompts")
        except Exception as e:
            print(f"❌ Failed to load visual prompts: {e}")
            return None, None

        try:
            with open(character_profiles_path, 'r', encoding='utf-8') as f:
                character_data = json.load(f)
            print(f"✅ Loaded character data")
        except Exception as e:
            print(f"⚠️ Character data not found: {e}")
            character_data = {"main_characters": []}

        return visual_prompts, character_data

    def optimize_prompt(self, original_prompt: str, scene_data: dict) -> str:
        """Optimize prompt - COMPACT VERSION"""

        scene_number = scene_data.get('scene_number', 0)
        characters_in_scene = scene_data.get('characters_in_scene', [])

        # VERY SHORT prefix
        prefix = "79 CE Pompeii: "

        # Add character if present
        if characters_in_scene:
            char_names = [c.get('name', '') for c in characters_in_scene]
            char_detail = f"Roman {char_names[0]}, period clothing. "
            optimized = prefix + char_detail + original_prompt
        else:
            optimized = prefix + original_prompt

        # Short ending
        if scene_number != 99:
            ending = ". Roman architecture, historically accurate"
        else:
            ending = ". YouTube thumbnail, Vesuvius background"

        optimized += ending

        # HARD LIMIT - 300 characters max
        if len(optimized) > 300:
            optimized = optimized[:297] + "..."

        return optimized

    def generate_text_to_image(self, prompt: str, scene_number: int) -> dict:
        """Generate image using official SDK"""

        self.log_step(f"Generating Scene {scene_number} with official SDK")

        print(f"   📝 Prompt length: {len(prompt)} chars")
        print(f"   📝 Prompt: {prompt}")

        try:
            # Create task
            response = self.client.text_to_image.create(
                model="gen4_image",
                ratio="1920:1080",
                prompt_text=prompt
            )

            # Get task ID and poll
            if hasattr(response, 'id'):
                task_id = response.id
                print(f"   📋 Task ID: {task_id}")

                # Poll for completion
                for attempt in range(30):
                    time.sleep(10)

                    try:
                        task = self.client.tasks.retrieve(task_id)

                        if hasattr(task, 'status'):
                            status = task.status
                            print(f"   📊 Status: {status}")

                            if status == "SUCCEEDED":
                                if hasattr(task, 'output') and task.output:
                                    if isinstance(task.output, list) and len(task.output) > 0:
                                        image_url = task.output[0]
                                    elif isinstance(task.output, dict):
                                        image_url = task.output.get('url') or task.output.get('image_url')
                                    else:
                                        image_url = str(task.output)

                                    return {
                                        "success": True,
                                        "image_url": image_url,
                                        "scene_number": scene_number,
                                        "task_id": task_id
                                    }

                            elif status == "FAILED":
                                error_detail = getattr(task, 'error', 'Unknown error')
                                return {
                                    "success": False,
                                    "error": f"Task failed: {error_detail}",
                                    "scene_number": scene_number
                                }
                            elif status in ["PENDING", "RUNNING", "THROTTLED"]:
                                continue
                            else:
                                print(f"   ⚠️ Unknown status: {status}")
                                continue

                    except Exception as poll_error:
                        print(f"   ⚠️ Polling error: {poll_error}")
                        time.sleep(5)
                        continue

                return {
                    "success": False,
                    "error": "Timeout waiting for completion",
                    "scene_number": scene_number
                }
            else:
                return {
                    "success": False,
                    "error": "No task ID received",
                    "scene_number": scene_number
                }

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ SDK Error: {error_msg}")

            return {
                "success": False,
                "error": error_msg,
                "scene_number": scene_number
            }

    def download_image(self, image_url: str, scene_number: int) -> str:
        """Download and save image"""

        output_dir = RUNWAY_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        if scene_number == 99:
            filename = "thumbnail.png"
        else:
            filename = f"scene_{scene_number:02d}.png"

        file_path = os.path.join(output_dir, filename)

        try:
            import requests
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            file_size = os.path.getsize(file_path)
            print(f"   💾 Saved: {filename} ({file_size:,} bytes)")

            return file_path

        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            return None

    def generate_all_scenes(self):
        """Generate all scenes for Topic 1"""

        self.log_step("Starting Pompeii scene generation")

        # Load data
        visual_prompts, character_data = self.load_topic_1_data()
        if not visual_prompts:
            return False

        # Separate scenes and thumbnail
        scenes = [p for p in visual_prompts if p.get('scene_number', 0) != 99]
        thumbnail = next((p for p in visual_prompts if p.get('scene_number', 0) == 99), None)

        print(f"\n🎬 Found {len(scenes)} scenes + {'1 thumbnail' if thumbnail else 'no thumbnail'}")

        success_count = 0
        total_count = len(scenes) + (1 if thumbnail else 0)

        # Generate ALL scenes
        for scene_data in scenes:
            scene_number = scene_data.get('scene_number', 0)
            original_prompt = scene_data.get('prompt', '')

            if not original_prompt:
                print(f"❌ Scene {scene_number}: No prompt")
                continue

            # Optimize prompt
            optimized_prompt = self.optimize_prompt(original_prompt, scene_data)
            print(f"\n🎨 Scene {scene_number}: {optimized_prompt[:80]}...")

            # Generate
            result = self.generate_text_to_image(optimized_prompt, scene_number)

            if result["success"]:
                # Download
                file_path = self.download_image(result["image_url"], scene_number)
                if file_path:
                    success_count += 1
                    self.generated_images.append({
                        "scene_number": scene_number,
                        "file_path": file_path,
                        "prompt": optimized_prompt,
                        "runway_url": result["image_url"]
                    })
                    print(f"   ✅ Scene {scene_number} complete")
                else:
                    print(f"   ❌ Scene {scene_number} download failed")
            else:
                print(f"   ❌ Scene {scene_number} generation failed: {result.get('error', 'Unknown')}")

            # Rate limiting
            time.sleep(5)

        # Generate thumbnail
        if thumbnail:
            print(f"\n🖼️ Generating thumbnail...")

            original_prompt = thumbnail.get('prompt', '')
            if original_prompt:
                optimized_prompt = self.optimize_prompt(original_prompt, thumbnail)

                result = self.generate_text_to_image(optimized_prompt, 99)

                if result["success"]:
                    file_path = self.download_image(result["image_url"], 99)
                    if file_path:
                        success_count += 1
                        self.generated_images.append({
                            "scene_number": 99,
                            "file_path": file_path,
                            "prompt": optimized_prompt,
                            "runway_url": result["image_url"]
                        })
                        print(f"   ✅ Thumbnail complete")

        # Save report
        self.save_report(success_count, total_count)

        return success_count > 0

    def save_report(self, success_count: int, total_count: int):
        """Save generation report"""

        report = {
            "topic": "Pompeii's Final Night",
            "completed": datetime.now().isoformat(),
            "total_scenes": total_count,
            "successful": success_count,
            "success_rate": f"{(success_count/total_count)*100:.1f}%",
            "generated_images": self.generated_images,
            "output_dir": RUNWAY_CONFIG["output_dir"],
            "sdk_version": "official_runwayml_python_sdk"
        }

        report_path = os.path.join(RUNWAY_CONFIG["output_dir"], "generation_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 Report saved: {report_path}")

def main():
    """Main execution"""

    print("🚀 RUNWAY SDK GENERATOR - POMPEII'S FINAL NIGHT")
    print("=" * 60)
    print("🌋 Topic 1: Pompeii's Final Night")
    print("🔧 Method: Official RunwayML Python SDK")
    print(f"📁 Output: {RUNWAY_CONFIG['output_dir']}")
    print("🚀 PRODUCTION MODE: ALL SCENES")
    print("=" * 60)

    try:
        generator = OfficialRunwayGenerator()

        # Check data exists
        if not os.path.exists("../output/1"):
            print("❌ Topic 1 data not found")
            print("💡 Run story generator first")
            return

        # Generate scenes
        start_time = time.time()
        success = generator.generate_all_scenes()
        generation_time = time.time() - start_time

        # Results
        print("\n" + "🎉" * 60)
        print("RUNWAY GENERATION COMPLETE!")
        print("🎉" * 60)

        print(f"⏱️ Time: {generation_time:.1f}s")
        print(f"✅ Success: {'YES' if success else 'NO'}")
        print(f"📊 Generated: {len(generator.generated_images)} images")

        if generator.generated_images:
            print(f"\n📋 FILES:")
            for img in generator.generated_images:
                scene_num = img['scene_number']
                filename = os.path.basename(img['file_path'])
                if scene_num == 99:
                    print(f"🖼️ {filename}")
                else:
                    print(f"🎬 {filename}")

        if success:
            print(f"\n🚀 PRODUCTION COMPLETE!")
            print(f"📁 All scenes saved to: {RUNWAY_CONFIG['output_dir']}")
        else:
            print(f"\n❌ Production failed")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()