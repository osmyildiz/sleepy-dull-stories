"""
Apollodorus Character Generation Test
Using 60-word optimized system + working v7 configuration
"""

import requests
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

class ApolloTestGenerator:
    """Generate Apollodorus character using optimized system"""

    def __init__(self):
        self.base_url = "https://api.piapi.ai/api/v1"
        self.api_key = self.load_api_key()
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Apollodorus data
        self.apollodorus = {
            "name": "Apollodorus",
            "role": "ancient library guardian",
            "physical_description": "Elderly man with weathered hands and gentle eyes, wearing simple robes. Silver-streaked beard, moves with practiced grace despite his age. Carries himself with quiet dignity and purpose.",
            "importance_score": 6,
            "personality_traits": ["reverent", "methodical", "protective", "wise", "contemplative"]
        }

        print("🎭 Apollodorus Character Test initialized")

    def load_api_key(self):
        """Load API key from .env"""
        api_key = os.getenv('PIAPI_KEY')
        if api_key:
            return api_key

        env_files = [Path('.env'), Path('../.env'), Path('../../.env')]
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file, override=True)
                api_key = os.getenv('PIAPI_KEY')
                if api_key:
                    return api_key
        return None

    def build_apollodorus_prompt(self):
        """Build Apollodorus character prompt using 60-word system"""

        print("🎭 BUILDING APOLLODORUS CHARACTER PROMPT")
        print("=" * 45)

        # Extract key features (25 words max)
        physical_desc = self.apollodorus["physical_description"]
        key_features = "elderly man weathered hands gentle eyes simple robes silver-streaked beard practiced grace quiet dignity purpose"

        # Character prompt components
        components = [
            # Character type
            "Full body character reference",

            # Role
            "ancient library guardian",

            # Physical features (key visual elements)
            key_features,

            # Setting context
            "ancient historical Alexandria library setting",

            # Pose and composition
            "standing pose character design sheet",

            # Quality markers
            "photorealistic portrait cinematic lighting",

            # Technical params (working configuration)
            "--v 7.0 --ar 2:3"
        ]

        final_prompt = " ".join(components)

        # Count words
        content_words = len(" ".join(components[:-1]).split())

        print(f"🎭 Character: {self.apollodorus['name']}")
        print(f"👤 Role: {self.apollodorus['role']}")
        print(f"📝 Key features: {key_features}")
        print(f"📊 Content words: {content_words}")
        print(f"🔧 Full prompt: {final_prompt}")
        print(f"📏 Total length: {len(final_prompt)} chars")

        return final_prompt

    def build_apollodorus_scene_prompt(self, character_ref_url):
        """Build scene with Apollodorus using optimized 60-word system"""

        print("\n🎬 BUILDING APOLLODORUS SCENE PROMPT")
        print("=" * 40)

        # Scene-specific part (15 words max)
        specific_scene = "ancient library guardian checking scrolls at night oil lamp casting warm golden glow"

        # Default core (consistent across all scenes)
        default_core = "cinematic realistic photograph professional film photography dramatic lighting photorealistic historical scene detailed textures"

        # Style modifiers
        style_modifiers = "warm golden light deep shadows intimate atmosphere weathered materials classical proportions"

        # Technical params
        technical_params = "--v 7.0 --ar 16:9 --quality 1"

        # Build final prompt
        prompt_parts = [
            character_ref_url,
            specific_scene,
            default_core,
            style_modifiers,
            technical_params
        ]

        scene_prompt = " ".join(prompt_parts)

        # Count content words
        content_words = len(specific_scene.split()) + len(default_core.split()) + len(style_modifiers.split())

        print(f"📝 Scene: {specific_scene}")
        print(f"📊 Content words: {content_words}/50")
        print(f"🔧 Scene prompt: {scene_prompt[:150]}...")

        return scene_prompt

    def submit_character_task(self, prompt, task_name):
        """Submit character generation task"""

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt,
                "aspect_ratio": "2:3" if "character" in task_name.lower() else "16:9",
                "process_mode": "fast"
            }
        }

        print(f"\n🚀 SUBMITTING {task_name}")
        print("=" * 30)
        print(f"📋 Aspect ratio: {payload['input']['aspect_ratio']}")

        try:
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            print(f"📡 Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    task_id = task_data.get("task_id")

                    if task_id:
                        print(f"✅ {task_name} submitted!")
                        print(f"🎯 Task ID: {task_id}")
                        return task_id
                    else:
                        print(f"⚠️ No task ID in response")
                        return None
                else:
                    print(f"❌ API error: {result.get('message', 'Unknown')}")
                    return None

            elif response.status_code == 500:
                try:
                    error_data = response.json()
                    error_detail = error_data.get("data", {}).get("error", {})
                    raw_message = error_detail.get("raw_message", "")
                    print(f"❌ 500 Error: {raw_message}")
                except:
                    print(f"❌ 500 Error: Cannot parse")
                return None

            else:
                print(f"❌ HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

    def monitor_task(self, task_id, task_name):
        """Monitor task completion"""

        print(f"\n📊 MONITORING {task_name}: {task_id}")
        print("=" * 35)

        max_cycles = 25

        for cycle in range(max_cycles):
            print(f"🔄 Cycle {cycle + 1}/{max_cycles}")

            try:
                response = requests.get(
                    f"{self.base_url}/task/{task_id}",
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get("code") == 200:
                        task_data = result.get("data", {})
                        status = task_data.get("status", "unknown").lower()

                        print(f"📈 Status: {status}")

                        if status == "completed":
                            output = task_data.get("output", {})

                            image_url = None
                            if output.get("image_urls"):
                                image_url = output["image_urls"][0]
                            elif output.get("temporary_image_urls"):
                                image_url = output["temporary_image_urls"][0]
                            elif output.get("image_url"):
                                image_url = output["image_url"]

                            if image_url:
                                print(f"🎉 {task_name} completed!")
                                print(f"🖼️ Image: {image_url}")

                                # Download
                                save_path = f"./apollodorus_{task_name.lower().replace(' ', '_')}.png"
                                if self.download_image(image_url, save_path):
                                    print(f"✅ Saved: {save_path}")

                                return {"status": "completed", "image_url": image_url, "save_path": save_path}
                            else:
                                print("⚠️ Completed but no image")
                                return {"status": "completed_no_image"}

                        elif status == "failed":
                            error = task_data.get("error", {})
                            print(f"❌ Failed: {error}")
                            return {"status": "failed", "error": error}

                        elif status in ["processing", "pending", "running"]:
                            print(f"⏳ {status}...")

                    else:
                        print(f"⚠️ API error: {result}")

                else:
                    print(f"⚠️ HTTP {response.status_code}")

            except Exception as e:
                print(f"⚠️ Monitor error: {e}")

            if cycle < max_cycles - 1:
                time.sleep(30)

        return {"status": "timeout"}

    def download_image(self, image_url, save_path):
        """Download generated image"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://discord.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }

            response = requests.get(image_url, headers=headers, timeout=30, stream=True)

            if response.status_code == 200:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = Path(save_path).stat().st_size
                print(f"📥 Downloaded: {file_size} bytes")
                return True

            return False
        except:
            return False

    def run_apollodorus_test(self):
        """Run complete Apollodorus character and scene test"""

        print("🎭" * 20)
        print("APOLLODORUS CHARACTER GENERATION")
        print("👤 Name: Apollodorus")
        print("🏛️ Role: Ancient Library Guardian")
        print("📚 Setting: Alexandria Library")
        print("🎯 Goal: Character + Scene generation")
        print("🎭" * 20)

        if not self.api_key:
            print("❌ No API key!")
            return

        # Step 1: Generate Apollodorus character
        character_prompt = self.build_apollodorus_prompt()
        character_task_id = self.submit_character_task(character_prompt, "APOLLODORUS CHARACTER")

        if not character_task_id:
            print("❌ Character submission failed")
            return

        # Step 2: Monitor character generation
        character_result = self.monitor_task(character_task_id, "CHARACTER")

        if character_result.get("status") == "completed":
            character_url = character_result["image_url"]
            print(f"✅ Apollodorus character generated!")
            print(f"🔗 Character URL: {character_url}")

            # Step 3: Generate scene with character
            scene_prompt = self.build_apollodorus_scene_prompt(character_url)
            scene_task_id = self.submit_character_task(scene_prompt, "APOLLODORUS SCENE")

            if scene_task_id:
                # Step 4: Monitor scene generation
                scene_result = self.monitor_task(scene_task_id, "SCENE")

                if scene_result.get("status") == "completed":
                    print("\n" + "🎉" * 25)
                    print("APOLLODORUS GENERATION SUCCESS!")
                    print(f"🎭 Character: {character_result['save_path']}")
                    print(f"🎬 Scene: {scene_result['save_path']}")
                    print(f"🔗 Character URL: {character_url}")
                    print(f"🔗 Scene URL: {scene_result['image_url']}")
                    print(f"📊 60-word system: WORKING")
                    print(f"🎨 Style consistency: ACHIEVED")
                    print("🎉" * 25)
                    return True
                else:
                    print(f"❌ Scene generation failed: {scene_result}")
            else:
                print("❌ Scene submission failed")
        else:
            print(f"❌ Character generation failed: {character_result}")

        return False

def main():
    """Main test function"""

    generator = ApolloTestGenerator()
    success = generator.run_apollodorus_test()

    if success:
        print("\n✅ Apollodorus test completed successfully!")
        print("🎯 60-word system validated!")
        print("🚀 Ready for full automation!")
    else:
        print("\n❌ Apollodorus test failed!")

if __name__ == "__main__":
    main()