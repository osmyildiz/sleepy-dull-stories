import requests
import os
import json
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
import urllib.request

# Load environment
load_dotenv()


class MidjourneyVisualGenerator:
    def __init__(self):
        self.api_key = os.getenv("PIAPI_KEY")
        self.base_url = "https://api.piapi.ai/api/v1"

        # Auto-detect base directory from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(script_dir)
        self.topics_csv_path = os.path.join(self.base_dir, "data", "topics.csv")

        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Current project tracking
        self.current_row_index = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None
        self.current_historical_period = None

        # Generation tracking
        self.generation_log = []
        self.character_references = {}

        print("🚀 Midjourney Visual Generator v6.1 (Role-Based) Initialized")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"📊 Topics CSV: {self.topics_csv_path}")
        print(f"🔑 API Key: {self.api_key[:8]}...")

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps"""
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "project": self.current_row_index,
            "metadata": metadata or {}
        }
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step}")

    def get_next_topic(self) -> Tuple[bool, Optional[Dict]]:
        """Get next topic where done=1 and cover_image_created=0"""
        self.log_step("🔍 Reading topics.csv for next project")

        if not os.path.exists(self.topics_csv_path):
            self.log_step("❌ topics.csv not found", "ERROR")
            return False, None

        df = pd.read_csv(self.topics_csv_path)

        # Find first row where done=1 and cover_image_created=0 (or NaN)
        mask = (df["done"] == 1) & ((df["cover_image_created"] == 0) | df["cover_image_created"].isna())
        next_rows = df[mask]

        if next_rows.empty:
            self.log_step("✅ No topics ready for image generation", "INFO")
            return False, None

        # Get first matching row
        row = next_rows.iloc[0]
        row_index = next_rows.index[0] + 1  # Convert to 1-based indexing

        # Setup project paths
        self.current_row_index = row_index
        self.current_output_dir = os.path.join(self.base_dir, "output", str(row_index))
        self.current_topic = row["topic"]
        self.current_description = row["description"]
        self.current_historical_period = row.get("historical_period", "ancient times")

        project_info = {
            "row_index": row_index,
            "topic": self.current_topic,
            "description": self.current_description,
            "historical_period": self.current_historical_period,
            "output_dir": self.current_output_dir
        }

        self.log_step(f"✅ Found project: {self.current_topic}", "SUCCESS", project_info)
        return True, project_info

    def extract_character_role(self, character: Dict) -> str:
        """Extract character role from description dynamically"""
        description = character.get('physical_description', '').lower()
        historical_period = getattr(self, 'current_historical_period', 'ancient times')

        # Role detection based on description keywords
        role_keywords = {
            'baker': ['flour', 'bread', 'kneading', 'oven', 'dough', 'bakery'],
            'fisherman': ['fishing', 'nets', 'harbor', 'sea', 'boat', 'maritime'],
            'gladiator': ['sword', 'arena', 'combat', 'warrior', 'battle', 'muscular'],
            'senator': ['toga', 'dignified', 'authority', 'noble', 'distinguished'],
            'woman': ['elegant', 'graceful', 'flowing robes', 'gentle hands'],
            'priest': ['temple', 'robes', 'religious', 'ceremony', 'sacred'],
            'merchant': ['trade', 'goods', 'market', 'commerce', 'wealthy'],
            'soldier': ['armor', 'military', 'guard', 'captain', 'uniform'],
            'artisan': ['craft', 'tools', 'workshop', 'skilled', 'maker'],
            'healer': ['herbs', 'medicine', 'healing', 'physician']
        }

        # Check for role keywords in description
        detected_roles = []
        for role, keywords in role_keywords.items():
            if any(keyword in description for keyword in keywords):
                detected_roles.append(role)

        # Determine primary role
        if detected_roles:
            primary_role = detected_roles[0]  # First match

            # Context-specific role formatting
            if 'roman' in historical_period.lower() or '79 ad' in historical_period.lower() or 'century ce' in historical_period.lower():
                role_prefix = "ancient Roman"
            elif 'medieval' in historical_period.lower():
                role_prefix = "medieval"
            elif 'egyptian' in historical_period.lower():
                role_prefix = "ancient Egyptian"
            else:
                role_prefix = "historical"

            return f"{role_prefix} {primary_role}"

        # Fallback based on historical period
        if 'roman' in historical_period.lower():
            return "ancient Roman person"
        elif 'medieval' in historical_period.lower():
            return "medieval person"
        elif 'egyptian' in historical_period.lower():
            return "ancient Egyptian person"
        else:
            return "historical person"

    def load_project_data(self) -> Tuple[Dict, List[Dict]]:
        """Load character profiles and visual prompts for current project"""
        self.log_step("📂 Loading project data files")

        # Load character profiles
        char_profiles_path = os.path.join(self.current_output_dir, "character_profiles.json")
        with open(char_profiles_path, 'r', encoding='utf-8') as f:
            character_profiles = json.load(f)

        # Load visual generation prompts
        visual_prompts_path = os.path.join(self.current_output_dir, "visual_generation_prompts.json")
        with open(visual_prompts_path, 'r', encoding='utf-8') as f:
            visual_prompts = json.load(f)

        self.log_step("✅ Project data loaded", "SUCCESS", {
            "characters_count": len(character_profiles["main_characters"]),
            "scenes_count": len([s for s in visual_prompts if s["scene_number"] != 99]),
            "has_thumbnail": any(s["scene_number"] == 99 for s in visual_prompts)
        })

        return character_profiles, visual_prompts

    def generate_single_character_test(self, character_profiles: Dict):
        """Generate single character for testing"""
        self.log_step("🎭 Testing single character generation")

        main_characters = character_profiles["main_characters"]
        test_character = None

        # Find first marketing character
        for character in main_characters:
            if character.get("use_in_marketing", False):
                test_character = character
                break

        if not test_character:
            self.log_step("❌ No marketing characters found", "ERROR")
            return False

        char_name = test_character["name"]
        role = self.extract_character_role(test_character)
        physical = test_character['physical_description'].split(',')[0].strip()

        # Create prompt
        prompt = f"Full body character sheet, {role}, {physical}, {self.current_historical_period}, standing pose, character design reference --ar 2:3 --v 6.1"

        print(f"🎭 Testing character: {char_name}")
        print(f"📝 Role: {role}")
        print(f"📝 Prompt: {prompt[:100]}...")

        # TODO: Add actual generation here
        self.log_step(f"✅ Character prompt ready: {char_name}", "SUCCESS")
        return True
    def setup_directories(self):
        """Create necessary directories for current project"""
        self.characters_dir = os.path.join(self.current_output_dir, "characters")
        self.scenes_dir = os.path.join(self.current_output_dir, "scenes")

        os.makedirs(self.characters_dir, exist_ok=True)
        os.makedirs(self.scenes_dir, exist_ok=True)

        self.log_step("📁 Directories created", "SUCCESS")

    def test_character_role_detection(self, character_profiles: Dict):
        """Test character role detection for current project"""
        self.log_step("🧪 Testing character role detection")

        main_characters = character_profiles["main_characters"]

        for character in main_characters:
            if character.get("use_in_marketing", False):
                char_name = character["name"]
                role = self.extract_character_role(character)

                print(f"  🎭 {char_name} → {role}")

        self.log_step("✅ Role detection test complete", "SUCCESS")

    def submit_midjourney_task(self, prompt: str, aspect_ratio: str = "16:9") -> Optional[str]:
        """Submit task to Midjourney API"""
        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "process_mode": "relax",
                "skip_prompt_check": False
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    task_id = task_data.get("task_id")
                    self.log_step(f"✅ Task submitted: {task_id}", "SUCCESS")
                    return task_id
                else:
                    self.log_step(f"❌ API Error: {result}", "ERROR")
                    return None
            else:
                self.log_step(f"❌ HTTP Error: {response.status_code}", "ERROR")
                return None

        except Exception as e:
            self.log_step(f"❌ Request failed: {e}", "ERROR")
            return None

    def check_task_status(self, task_id: str) -> Optional[Dict]:
        """Check single task status"""
        try:
            status_url = f"{self.base_url}/task/{task_id}"
            response = requests.get(status_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    task_data = result.get("data", {})
                    status = task_data.get("status", "").lower()
                    output = task_data.get("output", {})

                    if status == "completed":
                        # Get image URLs with priority: temporary_image_urls > image_url
                        temp_urls = output.get("temporary_image_urls", [])
                        image_url = output.get("image_url", "")

                        if temp_urls and len(temp_urls) > 0:
                            selected_url = temp_urls[1] if len(temp_urls) >= 2 else temp_urls[0]
                            return {"url": selected_url, "source": "temporary_image_urls"}
                        elif image_url:
                            return {"url": image_url, "source": "image_url"}
                        else:
                            return False  # Completed but no images

                    elif status == "failed":
                        return False  # Failed
                    else:
                        return None  # Still processing

        except Exception as e:
            return None  # Error, treat as still processing

        return None

    def download_image(self, result_data: Dict, save_path: str) -> bool:
        """Download image with proper headers"""
        image_url = result_data["url"]

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://discord.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }

            response = requests.get(image_url, headers=headers, timeout=30, stream=True)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = os.path.getsize(save_path)
                self.log_step(f"✅ Downloaded: {os.path.basename(save_path)} ({file_size} bytes)", "SUCCESS")
                return True
            else:
                self.log_step(f"❌ HTTP {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Download failed: {e}", "ERROR")
            return False

    def generate_single_character_test(self, character_profiles: Dict):
        """Generate single character for testing"""
        self.log_step("🎭 Testing single character generation")

        main_characters = character_profiles["main_characters"]
        test_character = None

        # Find first marketing character
        for character in main_characters:
            if character.get("use_in_marketing", False):
                test_character = character
                break

        if not test_character:
            self.log_step("❌ No marketing characters found", "ERROR")
            return False

        char_name = test_character["name"]
        role = self.extract_character_role(test_character)
        physical = test_character['physical_description'].split(',')[0].strip()

        # Create prompt
        prompt = f"Full body character sheet, {role}, {physical}, {self.current_historical_period}, standing pose, character design reference --ar 2:3 --v 6.1"

        print(f"🎭 Testing character: {char_name}")
        print(f"📝 Role: {role}")
        print(f"📝 Prompt: {prompt[:100]}...")

        # Submit task
        task_id = self.submit_midjourney_task(prompt, aspect_ratio="2:3")
        if not task_id:
            return False

        # Monitor task (simplified - check a few times)
        print(f"⏳ Monitoring task: {task_id}")

        for i in range(20):  # Max 10 minutes
            result_data = self.check_task_status(task_id)

            if result_data and isinstance(result_data, dict):
                print(f"✅ Character generation complete!")
                print(f"🖼️ Image URL: {result_data['url']}")
                print(f"📊 Source: {result_data['source']}")

                # Download test
                safe_name = char_name.lower().replace(" ", "_")
                image_path = os.path.join(self.characters_dir, f"{safe_name}_test.png")

                if self.download_image(result_data, image_path):
                    self.log_step(f"✅ Single character test SUCCESS: {char_name}", "SUCCESS")
                    return True
                else:
                    return False

            elif result_data is False:
                self.log_step(f"❌ Character generation failed", "ERROR")
                return False
            else:
                print(f"⏳ Still processing... (check {i + 1}/20)")
                time.sleep(30)

        self.log_step("⏰ Character generation timeout", "ERROR")
        return False

    def test_api_connection(self) -> bool:
        """Test PIAPI connection"""
        self.log_step("🔍 Testing PIAPI connection")

        test_prompt = "red apple on white table --ar 1:1 --v 6.1"

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": test_prompt,
                "aspect_ratio": "1:1",
                "process_mode": "relax"
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/task",
                headers=self.headers,
                json=payload,
                timeout=10
            )

            print(f"📡 Test Response: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ API Connection OK: {result}")
                return True
            else:
                print(f"❌ API Error: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Connection Test Failed: {e}")
            return False

    def generate_all_characters_parallel(self, character_profiles: Dict):
        """Generate all marketing characters in parallel"""
        self.log_step("🎭 Starting parallel character generation")

        main_characters = character_profiles["main_characters"]
        marketing_characters = [char for char in main_characters if char.get("use_in_marketing", False)]

        if not marketing_characters:
            self.log_step("❌ No marketing characters found", "ERROR")
            return False

        # Submit all character tasks
        character_tasks = {}

        for character in marketing_characters:
            char_name = character["name"]
            role = self.extract_character_role(character)
            physical = character['physical_description'].split(',')[0].strip()

            prompt = f"Full body character sheet, {role}, {physical}, {self.current_historical_period}, standing pose, character design reference --ar 2:3 --v 6.1"

            print(f"🎭 Submitting: {char_name} → {role}")

            task_id = self.submit_midjourney_task(prompt, aspect_ratio="2:3")
            if task_id:
                character_tasks[char_name] = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "character_data": character
                }

            time.sleep(1)  # Brief rate limiting

        if not character_tasks:
            self.log_step("❌ No character tasks submitted", "ERROR")
            return False

        self.log_step(f"✅ Submitted {len(character_tasks)} character tasks", "SUCCESS")

        # Monitor all tasks
        completed_characters = {}
        max_cycles = 25  # 25 * 30 seconds = 12.5 minutes max

        for cycle in range(max_cycles):
            if not character_tasks:
                break

            completed_count = len(completed_characters)
            total_count = completed_count + len(character_tasks)
            self.log_step(f"📊 Cycle {cycle + 1}: {completed_count}/{total_count} completed")

            # Check each pending task
            chars_to_remove = []

            for char_name, task_data in character_tasks.items():
                task_id = task_data["task_id"]

                result_data = self.check_task_status(task_id)

                if result_data and isinstance(result_data, dict):
                    # Character completed!
                    self.log_step(f"✅ {char_name} completed!", "SUCCESS")
                    completed_characters[char_name] = {
                        "result_data": result_data,
                        "task_data": task_data
                    }
                    chars_to_remove.append(char_name)
                elif result_data is False:
                    # Character failed
                    self.log_step(f"❌ {char_name} failed", "ERROR")
                    chars_to_remove.append(char_name)

            # Remove completed/failed characters
            for char_name in chars_to_remove:
                del character_tasks[char_name]

            if not character_tasks:
                break

            # Wait before next cycle
            time.sleep(30)

        # Download all completed characters
        successful_downloads = 0

        for char_name, char_data in completed_characters.items():
            result_data = char_data["result_data"]

            safe_name = char_name.lower().replace(" ", "_").replace(".", "")
            image_path = os.path.join(self.characters_dir, f"{safe_name}.png")

            if self.download_image(result_data, image_path):
                # Save character reference for scenes
                self.character_references[char_name] = result_data["url"]
                successful_downloads += 1

                # Save metadata
                metadata = {
                    "name": char_name,
                    "role": self.extract_character_role(char_data["task_data"]["character_data"]),
                    "prompt": char_data["task_data"]["prompt"],
                    "image_url": result_data["url"],
                    "url_source": result_data["source"],
                    "local_path": image_path,
                    "generated_at": datetime.now().isoformat()
                }

                json_path = os.path.join(self.characters_dir, f"{safe_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Parallel character generation complete: {successful_downloads}/{len(marketing_characters)}",
                      "SUCCESS")

        return successful_downloads > 0

    def generate_all_scenes_parallel(self, visual_prompts: List[Dict]):
        """Generate all scenes in parallel with character references"""
        self.log_step("🎬 Starting parallel scene generation")

        # Get regular scenes (not thumbnail)
        regular_scenes = [s for s in visual_prompts if s["scene_number"] != 99]

        if not regular_scenes:
            self.log_step("❌ No scenes found", "ERROR")
            return False

        print(f"🎬 Found {len(regular_scenes)} scenes to generate")
        print(f"🎭 Available character references: {len(self.character_references)}")

        # Submit all scene tasks
        scene_tasks = {}

        for scene in regular_scenes:
            scene_num = scene["scene_number"]

            # Build scene prompt with character references
            base_prompt = scene.get("enhanced_prompt", scene["prompt"])

            # Add character references if characters are present in scene
            if scene.get("characters_present") and len(self.character_references) > 0:
                char_refs = []
                char_names = []
                for char_name in scene["characters_present"]:
                    if char_name in self.character_references:
                        char_refs.append(self.character_references[char_name])
                        char_names.append(char_name)

                if char_refs:
                    ref_string = " ".join(char_refs)
                    base_prompt = f"{base_prompt} {ref_string}"
                    print(f"🎬 Scene {scene_num}: Added {len(char_refs)} character refs ({', '.join(char_names)})")
                else:
                    print(f"ℹ️ Scene {scene_num}: Characters listed but no refs available")
            else:
                print(f"ℹ️ Scene {scene_num}: No character references needed")

            final_prompt = f"{base_prompt} --ar 16:9 --v 6.1"

            # Submit task
            task_id = self.submit_midjourney_task(final_prompt, aspect_ratio="16:9")
            if task_id:
                scene_tasks[scene_num] = {
                    "task_id": task_id,
                    "prompt": final_prompt,
                    "scene_data": scene
                }

            time.sleep(1)  # Brief rate limiting

        if not scene_tasks:
            self.log_step("❌ No scene tasks submitted", "ERROR")
            return False

        self.log_step(f"✅ Submitted {len(scene_tasks)} scene tasks", "SUCCESS")

        # Monitor all tasks (scenes take longer)
        completed_scenes = {}
        max_cycles = 30  # 30 * 30 seconds = 15 minutes max

        for cycle in range(max_cycles):
            if not scene_tasks:
                break

            completed_count = len(completed_scenes)
            total_count = completed_count + len(scene_tasks)
            self.log_step(f"📊 Scene Cycle {cycle + 1}: {completed_count}/{total_count} completed")

            # Check each pending scene
            scenes_to_remove = []

            for scene_num, task_data in scene_tasks.items():
                task_id = task_data["task_id"]

                result_data = self.check_task_status(task_id)

                if result_data and isinstance(result_data, dict):
                    # Scene completed!
                    self.log_step(f"✅ Scene {scene_num} completed!", "SUCCESS")
                    completed_scenes[scene_num] = {
                        "result_data": result_data,
                        "task_data": task_data
                    }
                    scenes_to_remove.append(scene_num)
                elif result_data is False:
                    # Scene failed
                    self.log_step(f"❌ Scene {scene_num} failed", "ERROR")
                    scenes_to_remove.append(scene_num)

            # Remove completed/failed scenes
            for scene_num in scenes_to_remove:
                del scene_tasks[scene_num]

            if not scene_tasks:
                break

            # Wait before next cycle
            time.sleep(30)

        # Download all completed scenes
        successful_downloads = 0

        for scene_num, scene_data in completed_scenes.items():
            result_data = scene_data["result_data"]

            image_path = os.path.join(self.scenes_dir, f"scene_{scene_num:02d}.png")

            if self.download_image(result_data, image_path):
                successful_downloads += 1

                # Save metadata
                metadata = {
                    "scene_number": scene_num,
                    "title": scene_data["task_data"]["scene_data"]["title"],
                    "prompt": scene_data["task_data"]["prompt"],
                    "image_url": result_data["url"],
                    "url_source": result_data["source"],
                    "local_path": image_path,
                    "generated_at": datetime.now().isoformat()
                }

                json_path = os.path.join(self.scenes_dir, f"scene_{scene_num:02d}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Parallel scene generation complete: {successful_downloads}/{len(regular_scenes)}", "SUCCESS")

        return successful_downloads > 0

    def run_complete_generation(self) -> bool:
        """Run complete image generation process"""
        print("🚀" * 30)
        print("MIDJOURNEY VISUAL GENERATOR v6.1")
        print("⚡ HYBRID PROCESSING MODE")
        print("🎭 Phase 1: Characters (parallel, role-based, full body)")
        print("🎬 Phase 2: Scenes (parallel + char refs)")
        print("🖼️ Phase 3: Thumbnail (with char refs)")
        print("🚀" * 30)

        # Step 0: Test API connection first
        if not self.test_api_connection():
            self.log_step("❌ API connection failed - aborting", "ERROR")
            return False

        # Step 1: Get next topic
        found, project_info = self.get_next_topic()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🏛️ Historical period: {project_info['historical_period']}")

        # Step 2: Setup directories
        self.setup_directories()

        # Step 3: Load project data
        character_profiles, visual_prompts = self.load_project_data()

        # Step 4: Test character role detection
        self.test_character_role_detection(character_profiles)

        print("\n" + "🎯" * 50)
        print("CHARACTER ROLE DETECTION TEST COMPLETE!")
        print("Next: Add character generation functionality")
        print("🎯" * 50)

        # Step 5: Generate all characters in parallel
        all_chars_success = self.generate_all_characters_parallel(character_profiles)

        if all_chars_success:
            print("\n" + "🎉" * 50)
            print("ALL CHARACTERS GENERATION SUCCESSFUL!")
            print(f"✅ Generated: {len(self.character_references)} characters")
            for char_name, url in self.character_references.items():
                print(f"  🎭 {char_name}: {url[:50]}...")
            print("Ready for scene generation!")
            print("🎉" * 50)
        else:
            print("\n" + "❌" * 50)
            print("CHARACTER GENERATION FAILED!")
            print("❌" * 50)

        return all_chars_success




if __name__ == "__main__":
    try:
        generator = MidjourneyVisualGenerator()
        success = generator.run_complete_generation()

        if success:
            print("🎊 Basic test completed successfully!")
        else:
            print("⚠️ Generation failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ Generation stopped by user")
    except Exception as e:
        print(f"💥 Generation failed: {e}")
        import traceback

        traceback.print_exc()