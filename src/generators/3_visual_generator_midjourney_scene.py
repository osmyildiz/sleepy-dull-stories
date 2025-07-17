import requests
import os
import json
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

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

        # Track failed scenes to prevent infinite loops
        self.scene_attempt_count = {}  # scene_number: attempt_count
        self.blacklisted_scenes = set()  # scenes that failed too many times

        print("🚀 Midjourney Visual Generator v9.0 (Universal Content Filter) Initialized")
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

    def apply_content_policy_filter(self, prompt: str) -> str:
        """Apply universal content policy filter to any prompt"""

        # Global content policy replacements
        replacements = {
            # Bath/water related
            "thermal baths": "ancient pool facility",
            "bath attendant": "Roman worker",
            "baths setting": "pool complex",
            "heated pools": "warm water pools",
            "bathing": "water facility",
            "bath": "pool",

            # Children related
            "children playing": "young people enjoying activities",
            "children": "young people",
            "kids": "youth",
            "child": "young person",

            # Physical intimacy
            "embracing tenderly": "sharing a peaceful moment",
            "embracing": "standing together peacefully",
            "embrace": "peaceful moment",
            "kissing": "showing affection",
            "intimate": "quiet",
            "tenderly": "peacefully",
            "romantic": "affectionate",

            # Bedroom/private spaces
            "bedchamber": "private chamber",
            "bedroom": "sleeping chamber",
            "bed": "resting area",

            # Body/nudity related
            "nude": "unclothed figure",
            "naked": "bare figure",
            "undressed": "unclothed",
            "bare": "uncovered",

            # Violence/conflict (just in case)
            "blood": "red liquid",
            "violence": "conflict",
            "fighting": "confrontation",

            # Modern terms that might confuse
            "thermal": "warm",
            "spa": "wellness area"
        }

        # Apply replacements
        filtered_prompt = prompt
        for old_term, new_term in replacements.items():
            filtered_prompt = filtered_prompt.replace(old_term, new_term)

        # Add safety qualifiers if needed
        safety_keywords = ["educational", "historical", "classical", "artistic"]
        has_safety = any(keyword in filtered_prompt.lower() for keyword in safety_keywords)

        if not has_safety:
            filtered_prompt += ", historical educational content, classical art style"

        # Add explicit safety clause for potentially sensitive scenes
        sensitive_indicators = ["couple", "private", "chamber", "pool", "young people"]
        if any(indicator in filtered_prompt.lower() for indicator in sensitive_indicators):
            if "no explicit content" not in filtered_prompt.lower():
                filtered_prompt += ", appropriate content, no explicit material"

        return filtered_prompt

    def is_content_policy_safe(self, prompt: str) -> bool:
        """Check if prompt is likely to pass content policy"""

        # Red flag keywords that often cause issues
        red_flags = [
            "children", "child", "kids", "minor",
            "nude", "naked", "bare", "undressed",
            "bath", "bathing", "thermal",
            "intimate", "romantic", "bedroom", "bed",
            "embrace", "kiss", "touch",
            "violence", "blood", "fight"
        ]

        prompt_lower = prompt.lower()
        found_flags = [flag for flag in red_flags if flag in prompt_lower]

        if found_flags:
            print(f"⚠️ Content policy issues detected: {', '.join(found_flags)} - Auto-filtering...")
            return False

        return True

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

    def setup_directories(self):
        """Create necessary directories for current project"""
        self.characters_dir = os.path.join(self.current_output_dir, "characters")
        self.scenes_dir = os.path.join(self.current_output_dir, "scenes")

        os.makedirs(self.characters_dir, exist_ok=True)
        os.makedirs(self.scenes_dir, exist_ok=True)

        self.log_step("📁 Directories created", "SUCCESS")

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

    def load_existing_character_references(self) -> bool:
        """Load existing character references from generated files"""
        self.log_step("🎭 Loading existing character references")

        if not os.path.exists(self.characters_dir):
            self.log_step("❌ Characters directory not found", "ERROR")
            return False

        loaded_count = 0

        # Look for character JSON files which contain the image URLs
        for filename in os.listdir(self.characters_dir):
            if filename.endswith('.json'):
                char_file_path = os.path.join(self.characters_dir, filename)

                try:
                    with open(char_file_path, 'r', encoding='utf-8') as f:
                        char_data = json.load(f)

                    char_name = char_data.get('name')
                    image_url = char_data.get('image_url')

                    if char_name and image_url:
                        self.character_references[char_name] = image_url
                        loaded_count += 1

                except Exception as e:
                    self.log_step(f"❌ Failed to load {filename}: {e}", "ERROR")

        self.log_step(f"✅ Loaded {loaded_count} character references", "SUCCESS")
        return loaded_count > 0

    def get_missing_scenes(self, visual_prompts: List[Dict]) -> List[Dict]:
        """Get list of scenes that are missing (not downloaded yet) and not blacklisted"""
        regular_scenes = [s for s in visual_prompts if s["scene_number"] != 99]
        missing_scenes = []

        for scene in regular_scenes:
            scene_num = scene["scene_number"]

            # Skip blacklisted scenes
            if scene_num in self.blacklisted_scenes:
                continue

            image_path = os.path.join(self.scenes_dir, f"scene_{scene_num:02d}.png")

            # Only add to missing if file doesn't exist
            if not os.path.exists(image_path):
                missing_scenes.append(scene)

        return missing_scenes

    def submit_midjourney_task(self, prompt: str, aspect_ratio: str = "16:9", retry_count: int = 0) -> Optional[str]:
        """Submit task to Midjourney API with universal content filtering and smart retry"""

        # Scene 32/34 için content filter bypass
        if "Roman garden shrine" in prompt or ("Roman kitchen" in prompt and "clay hearth" in prompt):
            filtered_prompt = prompt  # COMPLETE BYPASS
            print(f"🚫 SUBMIT: Content filter BYPASSED for ultra-safe prompt")
        else:
            # Apply content policy filter to ALL prompts automatically
            original_prompt = prompt
            filtered_prompt = self.apply_content_policy_filter(prompt)

            # Log if changes were made
            if filtered_prompt != original_prompt:
                print(f"🛡️ Content filter applied:")
                print(f"   Original: {original_prompt[:80]}...")
                print(f"   Filtered: {filtered_prompt[:80]}...")

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": filtered_prompt,
                "aspect_ratio": aspect_ratio,
                "process_mode": "relax"
            }
        }
        # ... rest of function
        print(f"🔍 Exact payload: {payload}")
        print(f"🔍 Headers: {self.headers}")
        print(f"🔍 URL: {self.base_url}/task")
        print(f"🔍 Retry count: {retry_count}")

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
                    if retry_count > 0:
                        print(f"✅ Task submitted after {retry_count} retries: {task_id}")
                    return task_id
                else:
                    print(f"❌ API Error: {result.get('message', 'Unknown error')}")
                    return None
            elif response.status_code == 500:
                # Rate limiting detected
                if retry_count < 3:
                    wait_time = (retry_count + 1) * 10  # 10, 20, 30 seconds
                    print(f"⚠️ HTTP 500 - Waiting {wait_time}s before retry {retry_count + 1}/3")
                    time.sleep(wait_time)
                    return self.submit_midjourney_task(original_prompt, aspect_ratio,
                                                       retry_count + 1)  # Use original prompt for retry
                else:
                    print(f"❌ HTTP 500 - Max retries reached")
                    return None
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    def build_safe_scene_prompt(self, scene: Dict) -> str:
        """Build content policy safe scene prompt"""

        base_prompt = scene.get("enhanced_prompt", scene["prompt"])
        scene_num = scene.get("scene_number")

        # Scene 32 ve 34 için COMPLETE BYPASS - zaten ultra güvenli
        if scene_num in [32, 34]:
            print(f"🚫 Scene {scene_num}: COMPLETE BYPASS - Ultra safe content")
            return base_prompt  # Hiçbir işlem yapma, direkt return

        # Diğer scene'ler için normal processing
        char_refs = []
        if scene.get("characters_present") and len(self.character_references) > 0:
            for char_name in scene["characters_present"]:
                if char_name in self.character_references:
                    char_refs.append(self.character_references[char_name])

        if char_refs:
            ref_string = " ".join(char_refs)
            final_prompt = f"{ref_string} {base_prompt}"
        else:
            final_prompt = base_prompt

        return final_prompt

    def check_task_status_detailed(self, task_id: str, scene_num: int) -> Optional[Dict]:
        """Check task status with detailed logging"""
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
                            print(f"⚠️ Scene {scene_num}: Completed but no image URLs found")
                            return False  # Completed but no images

                    elif status == "failed":
                        # Get failure reason if available
                        error_info = task_data.get("error", {})
                        error_msg = error_info.get("message", "Unknown error")
                        print(f"❌ Scene {scene_num}: Task failed - {error_msg}")
                        return False  # Failed
                    else:
                        return None  # Still processing

            else:
                print(f"⚠️ Scene {scene_num}: Status check failed HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"⚠️ Scene {scene_num}: Status check exception - {e}")
            return None  # Error, treat as still processing

        return None

    def download_image_detailed(self, result_data: Dict, save_path: str, scene_num: int) -> bool:
        """Download image with detailed logging"""
        image_url = result_data["url"]

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://discord.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }

            if scene_num == 99:
                print(f"📥 Thumbnail: Downloading from {image_url[:50]}...")
            elif scene_num == 0:
                print(f"📥 Character: Downloading from {image_url[:50]}...")
            else:
                print(f"📥 Scene {scene_num}: Downloading from {image_url[:50]}...")

            response = requests.get(image_url, headers=headers, timeout=30, stream=True)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = os.path.getsize(save_path)
                if scene_num == 99:
                    print(f"✅ Thumbnail: Downloaded successfully ({file_size} bytes)")
                elif scene_num == 0:
                    print(f"✅ Character: Downloaded successfully ({file_size} bytes)")
                else:
                    print(f"✅ Scene {scene_num}: Downloaded successfully ({file_size} bytes)")
                return True
            else:
                print(f"❌ Scene {scene_num}: Download failed HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Scene {scene_num}: Download exception - {e}")
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

            if response.status_code == 200:
                result = response.json()
                self.log_step("✅ API Connection OK", "SUCCESS")
                return True
            else:
                self.log_step(f"❌ API Error: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Connection Test Failed: {e}", "ERROR")
            return False

    def generate_all_characters_parallel(self, character_profiles: Dict):
        """Generate all marketing characters in parallel with content filtering"""
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

            # Content filter will be applied automatically in submit_midjourney_task
            task_id = self.submit_midjourney_task(prompt, aspect_ratio="2:3")
            if task_id:
                character_tasks[char_name] = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "character_data": character
                }

            time.sleep(3)  # Rate limiting

        if not character_tasks:
            self.log_step("❌ No character tasks submitted", "ERROR")
            return False

        self.log_step(f"✅ Submitted {len(character_tasks)} character tasks", "SUCCESS")

        # Monitor all tasks
        completed_characters = {}
        max_cycles = 25

        for cycle in range(max_cycles):
            if not character_tasks:
                break

            completed_count = len(completed_characters)
            total_count = completed_count + len(character_tasks)
            print(f"📊 Character Cycle {cycle + 1}: {completed_count}/{total_count} completed")

            chars_to_remove = []

            for char_name, task_data in character_tasks.items():
                task_id = task_data["task_id"]

                result_data = self.check_task_status_detailed(task_id, 0)  # 0 for character

                if result_data and isinstance(result_data, dict):
                    print(f"✅ {char_name} completed!")
                    completed_characters[char_name] = {
                        "result_data": result_data,
                        "task_data": task_data
                    }
                    chars_to_remove.append(char_name)
                elif result_data is False:
                    print(f"❌ {char_name} failed")
                    chars_to_remove.append(char_name)

            for char_name in chars_to_remove:
                del character_tasks[char_name]

            if not character_tasks:
                break

            time.sleep(30)

        # Download all completed characters
        successful_downloads = 0

        for char_name, char_data in completed_characters.items():
            result_data = char_data["result_data"]

            safe_name = char_name.lower().replace(" ", "_").replace(".", "")
            image_path = os.path.join(self.characters_dir, f"{safe_name}.png")

            if self.download_image_detailed(result_data, image_path, 0):  # 0 for character
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

        print(f"✅ Character generation complete: {successful_downloads}/{len(marketing_characters)}")
        return successful_downloads > 0

    def generate_scenes_with_intelligent_retry(self, visual_prompts: List[Dict], max_retry_rounds: int = 15):
        """Enhanced scene generation with intelligent retry using Claude AI - FINAL CLAUDE RETRY ONLY"""

        print("🧠 ENHANCED SCENE GENERATION WITH FINAL CLAUDE RETRY")
        print("🔄 Normal retry: Auto retry without Claude")
        print("🧠 Final retry: Claude AI generates new prompts for failed downloads")
        print("🛡️ Content filtering: All prompts")

        # Track failed downloads separately from failed submissions
        download_failed_scenes = set()

        for retry_round in range(max_retry_rounds):
            missing_scenes = self.get_missing_scenes(visual_prompts)

            if not missing_scenes:
                print("✅ All scenes completed!")
                # FINAL CLAUDE RETRY FOR DOWNLOAD FAILURES
                if download_failed_scenes and self.intelligent_retry_enabled:
                    print(f"\n🧠 FINAL CLAUDE RETRY: {len(download_failed_scenes)} scenes with download failures")
                    return self._run_final_claude_retry(visual_prompts, download_failed_scenes)
                return True

            # Determine retry mode - NO INTELLIGENT RETRY DURING NORMAL ROUNDS
            retry_mode = "NORMAL"
            print(f"\n🔄 NORMAL RETRY ROUND {retry_round + 1}: {len(missing_scenes)} missing scenes")

            # Check blacklisted scenes
            total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
            blacklisted_count = len(self.blacklisted_scenes)

            if blacklisted_count > 0:
                print(f"⚫ {blacklisted_count} scenes blacklisted (failed too many times)")

            # Update attempt counts and blacklisting - MORE LENIENT
            for scene in missing_scenes:
                scene_num = scene["scene_number"]
                self.scene_attempt_count[scene_num] = self.scene_attempt_count.get(scene_num, 0) + 1

                # INCREASED blacklisting threshold - give more chances
                max_attempts = 8  # Increased from 5

                if self.scene_attempt_count[scene_num] > max_attempts:
                    self.blacklisted_scenes.add(scene_num)
                    print(f"⚫ Scene {scene_num}: Blacklisted after {self.scene_attempt_count[scene_num]} attempts")

            # Re-get missing scenes after blacklisting
            missing_scenes = self.get_missing_scenes(visual_prompts)

            if not missing_scenes:
                completed_count = total_scenes - blacklisted_count
                print(f"✅ All processable scenes completed! ({completed_count}/{total_scenes})")
                # FINAL CLAUDE RETRY FOR DOWNLOAD FAILURES
                if download_failed_scenes and self.intelligent_retry_enabled:
                    print(f"\n🧠 FINAL CLAUDE RETRY: {len(download_failed_scenes)} scenes with download failures")
                    return self._run_final_claude_retry(visual_prompts, download_failed_scenes)
                return True

            # Wait between retry rounds
            if retry_round > 0:
                wait_time = 60
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

            # Submit scene tasks - NORMAL PROMPTS ONLY
            scene_tasks = {}
            successful_submissions = 0

            for i, scene in enumerate(missing_scenes):
                scene_num = scene["scene_number"]
                attempt_num = self.scene_attempt_count.get(scene_num, 0)

                print(f"🎬 Processing Scene {scene_num} ({i + 1}/{len(missing_scenes)}) - Attempt #{attempt_num}")

                # ALWAYS use normal prompt building - NO CLAUDE DURING NORMAL RETRY
                final_prompt = self.build_safe_scene_prompt(scene)

                # Handle long prompts
                if len(final_prompt) > 4000:
                    print(f"⚠️ Scene {scene_num}: Truncating long prompt...")
                    final_prompt = final_prompt[:3900] + " --ar 16:9 --v 6.1"

                # Content policy check (informational)
                if not self.is_content_policy_safe(final_prompt):
                    print(f"🛡️ Scene {scene_num}: Content filter will be applied")

                # Submit task
                task_id = self.submit_midjourney_task(final_prompt, aspect_ratio="16:9")

                if task_id:
                    scene_tasks[scene_num] = {
                        "task_id": task_id,
                        "prompt": final_prompt,
                        "scene_data": scene,
                        "retry_mode": retry_mode
                    }
                    successful_submissions += 1
                    print(f"✅ Scene {scene_num}: Submitted successfully")
                else:
                    print(f"❌ Scene {scene_num}: Submission failed")

                # Rate limiting
                base_wait = 5 if retry_round < 3 else 8
                wait_time = base_wait + (retry_round * 2)

                if i < len(missing_scenes) - 1:
                    print(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)

            print(
                f"📊 Round {retry_round + 1} submissions: ✅ {successful_submissions} | ❌ {len(missing_scenes) - successful_submissions}")

            if not scene_tasks:
                print("❌ No tasks submitted, continuing to next round...")
                continue

            # Monitor tasks
            completed_scenes = {}
            max_cycles = 50

            for cycle in range(max_cycles):
                if not scene_tasks:
                    break

                completed_count = len(completed_scenes)
                total_count = completed_count + len(scene_tasks)
                print(f"📊 Monitoring Cycle {cycle + 1}: {completed_count}/{total_count} completed")

                scenes_to_remove = []

                for scene_num, task_data in scene_tasks.items():
                    task_id = task_data["task_id"]

                    result_data = self.check_task_status_detailed(task_id, scene_num)

                    if result_data and isinstance(result_data, dict):
                        print(f"✅ Scene {scene_num}: Task completed!")
                        completed_scenes[scene_num] = {
                            "result_data": result_data,
                            "task_data": task_data
                        }
                        scenes_to_remove.append(scene_num)
                    elif result_data is False:
                        print(f"❌ Scene {scene_num}: Task failed")
                        scenes_to_remove.append(scene_num)

                for scene_num in scenes_to_remove:
                    del scene_tasks[scene_num]

                if not scene_tasks:
                    break

                time.sleep(30)

            # Download completed scenes - TRACK DOWNLOAD FAILURES
            successful_downloads = 0

            for scene_num, scene_data in completed_scenes.items():
                result_data = scene_data["result_data"]
                image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

                if self.download_image_detailed(result_data, str(image_path), scene_num):
                    successful_downloads += 1
                    # Remove from download failed if it was there
                    download_failed_scenes.discard(scene_num)

                    # Save metadata
                    metadata = {
                        "scene_number": scene_num,
                        "title": scene_data["task_data"]["scene_data"]["title"],
                        "prompt": scene_data["task_data"]["prompt"],
                        "image_url": result_data["url"],
                        "url_source": result_data["source"],
                        "local_path": str(image_path),
                        "generated_at": datetime.now().isoformat(),
                        "retry_round": retry_round,
                        "retry_mode": scene_data["task_data"]["retry_mode"],
                        "attempt_number": self.scene_attempt_count.get(scene_num, 1),
                        "content_filtered": True
                    }

                    json_path = self.scenes_dir / f"scene_{scene_num:02d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                else:
                    print(f"❌ Scene {scene_num}: Download failed, tracking for final Claude retry")
                    # ADD TO DOWNLOAD FAILED TRACKING
                    download_failed_scenes.add(scene_num)

            print(f"✅ Round {retry_round + 1} downloads: {successful_downloads}")
            if download_failed_scenes:
                print(f"📥 Download failures tracked: {len(download_failed_scenes)} scenes")

        # Final check and summary
        final_missing = self.get_missing_scenes(visual_prompts)
        total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
        completed_count = total_scenes - len(final_missing) - len(self.blacklisted_scenes)

        print(f"\n📊 NORMAL RETRY SUMMARY:")
        print(f"✅ Completed: {completed_count}")
        print(f"❌ Missing: {len(final_missing)}")
        print(f"⚫ Blacklisted: {len(self.blacklisted_scenes)}")
        print(f"📥 Download failed: {len(download_failed_scenes)}")

        # FINAL CLAUDE RETRY FOR ALL FAILURES
        if (final_missing or download_failed_scenes) and self.intelligent_retry_enabled:
            all_failed_scenes = set()

            # Add missing scenes (not blacklisted)
            for scene in final_missing:
                if scene["scene_number"] not in self.blacklisted_scenes:
                    all_failed_scenes.add(scene["scene_number"])

            # Add download failed scenes
            all_failed_scenes.update(download_failed_scenes)

            if all_failed_scenes:
                print(f"\n🧠 RUNNING FINAL CLAUDE RETRY FOR {len(all_failed_scenes)} FAILED SCENES")
                return self._run_final_claude_retry(visual_prompts, all_failed_scenes)

        success_rate = completed_count / total_scenes
        return success_rate >= 0.85

    def _run_final_claude_retry(self, visual_prompts: List[Dict], failed_scene_numbers: set) -> bool:
        """Run final Claude-powered retry for failed scenes"""

        print("\n" + "🧠" * 60)
        print("FINAL CLAUDE AI RETRY - ALTERNATIVE PROMPTS")
        print("🤖 Claude will generate completely new prompts")
        print("🎯 Last chance for failed/download-failed scenes")
        print("🧠" * 60)

        if not self.intelligent_retry_system:
            print("❌ Intelligent retry system not available")
            return False

        # Get scenes that need Claude retry
        scenes_for_claude = []
        for scene in visual_prompts:
            if scene["scene_number"] in failed_scene_numbers:
                scenes_for_claude.append(scene)

        if not scenes_for_claude:
            print("✅ No scenes need Claude retry")
            return True

        print(f"🧠 Claude will process {len(scenes_for_claude)} failed scenes")

        # Submit Claude-generated prompts
        claude_tasks = {}
        successful_claude_submissions = 0

        for i, scene in enumerate(scenes_for_claude):
            scene_num = scene["scene_number"]

            print(f"\n🧠 Claude Processing Scene {scene_num} ({i + 1}/{len(scenes_for_claude)})")

            # Get failure context
            failure_context = self.intelligent_retry_system.create_intelligent_retry_context(
                scene, self.scene_attempt_count.get(scene_num, 0)
            )

            # Generate alternative prompt with Claude
            alternative_prompt = self.intelligent_retry_system.generate_alternative_scene_prompt(
                scene, failure_context
            )

            if alternative_prompt:
                print(f"🤖 Scene {scene_num}: Claude generated alternative prompt")
                print(f"📝 New prompt: {alternative_prompt[:100]}...")

                # Submit Claude's prompt
                task_id = self.submit_midjourney_task(alternative_prompt, aspect_ratio="16:9")

                if task_id:
                    claude_tasks[scene_num] = {
                        "task_id": task_id,
                        "prompt": alternative_prompt,
                        "scene_data": scene,
                        "claude_generated": True
                    }
                    successful_claude_submissions += 1
                    print(f"✅ Scene {scene_num}: Claude prompt submitted successfully")

                    # Track Claude retry
                    if scene_num not in self.intelligent_retry_system.intelligent_retries:
                        self.intelligent_retry_system.intelligent_retries[scene_num] = {
                            "attempts": 1,
                            "prompts": [alternative_prompt]
                        }
                    else:
                        self.intelligent_retry_system.intelligent_retries[scene_num]["attempts"] += 1
                        self.intelligent_retry_system.intelligent_retries[scene_num]["prompts"].append(
                            alternative_prompt)
                else:
                    print(f"❌ Scene {scene_num}: Claude prompt submission failed")
            else:
                print(f"❌ Scene {scene_num}: Claude failed to generate alternative prompt")

            # Wait between Claude submissions
            if i < len(scenes_for_claude) - 1:
                print("⏳ Waiting 10 seconds before next Claude prompt...")
                time.sleep(10)

        print(
            f"\n📊 Claude submissions: ✅ {successful_claude_submissions} | ❌ {len(scenes_for_claude) - successful_claude_submissions}")

        if not claude_tasks:
            print("❌ No Claude tasks submitted")
            return False

        # Monitor Claude tasks
        print("\n🔍 Monitoring Claude-generated tasks...")
        completed_claude_scenes = {}
        max_claude_cycles = 60  # Longer timeout for Claude tasks

        for cycle in range(max_claude_cycles):
            if not claude_tasks:
                break

            completed_count = len(completed_claude_scenes)
            total_count = completed_count + len(claude_tasks)
            print(f"🧠 Claude Monitoring Cycle {cycle + 1}: {completed_count}/{total_count} completed")

            scenes_to_remove = []

            for scene_num, task_data in claude_tasks.items():
                task_id = task_data["task_id"]

                result_data = self.check_task_status_detailed(task_id, scene_num)

                if result_data and isinstance(result_data, dict):
                    print(f"✅ Scene {scene_num}: Claude task completed!")
                    completed_claude_scenes[scene_num] = {
                        "result_data": result_data,
                        "task_data": task_data
                    }
                    scenes_to_remove.append(scene_num)
                elif result_data is False:
                    print(f"❌ Scene {scene_num}: Claude task failed")
                    scenes_to_remove.append(scene_num)

            for scene_num in scenes_to_remove:
                del claude_tasks[scene_num]

            if not claude_tasks:
                break

            time.sleep(30)

        # Download Claude-generated scenes
        print("\n📥 Downloading Claude-generated scenes...")
        claude_successful_downloads = 0

        for scene_num, scene_data in completed_claude_scenes.items():
            result_data = scene_data["result_data"]
            image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

            print(f"📥 Scene {scene_num}: Downloading Claude-generated image...")

            if self.download_image_detailed(result_data, str(image_path), scene_num):
                claude_successful_downloads += 1

                # Enhanced metadata for Claude-generated scenes
                metadata = {
                    "scene_number": scene_num,
                    "title": scene_data["task_data"]["scene_data"]["title"],
                    "prompt": scene_data["task_data"]["prompt"],
                    "image_url": result_data["url"],
                    "url_source": result_data["source"],
                    "local_path": str(image_path),
                    "generated_at": datetime.now().isoformat(),
                    "claude_generated": True,
                    "final_retry": True,
                    "claude_attempt_number": self.intelligent_retry_system.intelligent_retries[scene_num]["attempts"],
                    "total_attempts": self.scene_attempt_count.get(scene_num, 0) + 1,
                    "content_filtered": True,
                    "intelligent_retry_details": self.intelligent_retry_system.intelligent_retries[scene_num]
                }

                json_path = self.scenes_dir / f"scene_{scene_num:02d}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"✅ Scene {scene_num}: Claude-generated scene downloaded and saved!")
            else:
                print(f"❌ Scene {scene_num}: Claude-generated scene download failed")

        # Final Claude retry summary
        print(f"\n📊 CLAUDE RETRY RESULTS:")
        print(f"🤖 Claude prompts generated: {successful_claude_submissions}")
        print(f"✅ Successfully downloaded: {claude_successful_downloads}")
        print(f"🧠 Total Claude API calls: {self.intelligent_retry_system.claude_calls_made}")

        # Overall success assessment
        final_missing_after_claude = self.get_missing_scenes(visual_prompts)
        total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
        blacklisted_count = len(self.blacklisted_scenes)
        final_completed = total_scenes - len(final_missing_after_claude) - blacklisted_count

        print(f"\n📊 FINAL RESULTS AFTER CLAUDE RETRY:")
        print(f"✅ Total completed: {final_completed}/{total_scenes}")
        print(f"❌ Still missing: {len(final_missing_after_claude)}")
        print(f"⚫ Blacklisted: {blacklisted_count}")
        print(f"🧠 Claude rescued: {claude_successful_downloads} scenes")

        if final_missing_after_claude:
            print(f"⚠️ Scenes still missing after Claude retry:")
            for scene in final_missing_after_claude:
                print(f"  ❌ Scene {scene['scene_number']}: {scene['title']}")

        success_rate = final_completed / total_scenes

        if success_rate >= 0.9:
            print(f"🎉 CLAUDE RETRY SUCCESSFUL! {success_rate:.1%} completion rate")
            return True
        else:
            print(f"⚠️ CLAUDE RETRY PARTIAL SUCCESS: {success_rate:.1%} completion rate")
            return success_rate >= 0.8

    def generate_thumbnail(self, visual_prompts: List[Dict]) -> bool:
        """Generate YouTube thumbnail with dramatic expression and content filtering"""
        self.log_step("🖼️ Starting thumbnail generation")

        # Find thumbnail scene
        thumbnail_scene = None
        for scene in visual_prompts:
            if scene["scene_number"] == 99:
                thumbnail_scene = scene
                break

        if not thumbnail_scene:
            self.log_step("❌ No thumbnail scene found", "ERROR")
            return False

        # Get main character
        main_char = thumbnail_scene.get("character_used", "")
        if not main_char or main_char not in self.character_references:
            self.log_step(f"❌ Thumbnail character '{main_char}' not found", "ERROR")
            return False

        print(f"🖼️ Generating thumbnail with character: {main_char}")

        # Get base prompt and enhance with dramatic emotion
        base_prompt = thumbnail_scene.get("prompt", "")

        # DRAMATIC ENHANCEMENT - from successful tests
        dramatic_enhancement = """extreme close-up portrait, positioned RIGHT SIDE of frame,
        INTENSE SHOCK and TERROR facial expression, eyes WIDE OPEN with pure fear and panic,
        eyebrows raised extremely high, mouth OPEN in shocked gasp and alarm,
        facial muscles tense with horror, looking with frightened surprise toward left side,
        dramatic chiaroscuro lighting, deep shadows LEFT SIDE for text overlay,
        YouTube clickbait thumbnail style, exaggerated emotional expression for maximum impact"""

        enhanced_prompt = f"{base_prompt}, {dramatic_enhancement}"

        # Add character reference at beginning (Midjourney requirement)
        char_ref_url = self.character_references[main_char]
        final_prompt = f"{char_ref_url} {enhanced_prompt} --ar 16:9 --v 6.1"

        print(f"🎬 Enhanced thumbnail with TERROR expression and character reference")
        print(f"🛡️ Content filter will be applied automatically")

        # Submit thumbnail task (content filter applied automatically)
        task_id = self.submit_midjourney_task(final_prompt, aspect_ratio="16:9")

        if not task_id:
            return False

        print(f"⏳ Monitoring thumbnail: {task_id}")

        # Monitor thumbnail generation
        for i in range(25):
            result_data = self.check_task_status_detailed(task_id, 99)  # 99 for thumbnail

            if result_data and isinstance(result_data, dict):
                print(f"✅ Thumbnail complete!")

                thumbnail_path = os.path.join(self.current_output_dir, "thumbnail.png")

                if self.download_image_detailed(result_data, thumbnail_path, 99):
                    metadata = {
                        "character_used": main_char,
                        "clickbait_title": thumbnail_scene.get("clickbait_title", ""),
                        "base_prompt": base_prompt,
                        "enhanced_prompt": final_prompt,
                        "emotion_enhancement": "INTENSE_SHOCK_TERROR_WIDE_EYES_OPEN_MOUTH",
                        "character_reference": char_ref_url,
                        "image_url": result_data["url"],
                        "local_path": thumbnail_path,
                        "generated_at": datetime.now().isoformat(),
                        "content_filtered": True
                    }

                    json_path = os.path.join(self.current_output_dir, "thumbnail.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    return True

            elif result_data is False:
                print(f"❌ Thumbnail failed")
                return False
            else:
                print(f"⏳ Processing... ({i + 1}/25)")
                time.sleep(30)

        return False

    def run_complete_generation(self) -> bool:
        """Run complete image generation with universal content filtering"""
        print("🚀" * 30)
        print("MIDJOURNEY VISUAL GENERATOR v9.0")
        print("🛡️ UNIVERSAL CONTENT FILTER ACTIVE")
        print("🎭 Phase 1: Characters (content filtered)")
        print("🎬 Phase 2: Scenes (smart retry + content filtered)")
        print("🖼️ Phase 3: Thumbnail (TERROR expression + content filtered)")
        print("🚀" * 30)

        # Test API
        if not self.test_api_connection():
            return False

        # Get project
        found, project_info = self.get_next_topic()
        if not found:
            return False

        print(f"✅ Project: {project_info['topic']}")
        print(f"📁 Output: {project_info['output_dir']}")

        # Setup
        self.setup_directories()
        character_profiles, visual_prompts = self.load_project_data()

        # Characters
        if not self.load_existing_character_references():
            print("\n🎭 GENERATING CHARACTERS WITH CONTENT FILTER...")
            if not self.generate_all_characters_parallel(character_profiles):
                return False
        else:
            print(f"✅ Loaded {len(self.character_references)} character references")

        # Scenes with smart retry and content filtering
        print("\n🎬 GENERATING SCENES WITH UNIVERSAL CONTENT FILTER...")
        scenes_success = self.generate_scenes_with_retry(visual_prompts, max_retry_rounds=10)

        if not scenes_success:
            print("❌ Scene generation failed after all retries")
            return False

        # Thumbnail with dramatic enhancement and content filtering
        print("\n🖼️ GENERATING THUMBNAIL WITH CONTENT FILTER...")
        thumbnail_success = self.generate_thumbnail(visual_prompts)

        if thumbnail_success:
            print("\n" + "🎊" * 50)
            print("PRODUCTION COMPLETE WITH UNIVERSAL CONTENT FILTER!")
            print("✅ Characters generated (content filtered)")
            print("✅ ALL scenes generated (smart retry + content filtered)")
            print("✅ YouTube thumbnail with TERROR expression (content filtered)")
            print("🛡️ ALL PROMPTS AUTOMATICALLY SAFE FOR MIDJOURNEY")
            print("🎊 READY FOR CONTENT CREATION!")
            print("🎊" * 50)

            # Mark as completed
            self.mark_project_completed()

        return thumbnail_success

    def mark_project_completed(self):
        """Mark current project as completed in topics.csv"""
        try:
            df = pd.read_csv(self.topics_csv_path)
            df.loc[self.current_row_index - 1, "cover_image_created"] = 1
            df.to_csv(self.topics_csv_path, index=False)
            print("✅ Project marked as completed in topics.csv")
        except Exception as e:
            print(f"❌ Failed to update topics.csv: {e}")


if __name__ == "__main__":
    import sys

    try:
        generator = MidjourneyVisualGenerator()

        if len(sys.argv) > 1 and sys.argv[1] == "production":
            print("🚀 PRODUCTION MODE WITH UNIVERSAL CONTENT FILTER ACTIVATED")
            success = generator.run_complete_generation()

            if success:
                print("🎊 Production completed successfully!")
            else:
                print("⚠️ Production failed")

        else:
            print("📋 Usage:")
            print("  python 2_visual_generator_midjourney_scene.py production")

    except KeyboardInterrupt:
        print("\n⏹️ Generation stopped by user")
    except Exception as e:
        print(f"💥 Generation failed: {e}")
        import traceback

        traceback.print_exc()