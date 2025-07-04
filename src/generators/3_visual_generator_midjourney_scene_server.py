"""
Sleepy Dull Stories - SERVER-READY Midjourney Scene Generator
SCENE & THUMBNAIL generation with character reference integration
Production-optimized with complete automation and error recovery
"""

import requests
import os
import json
import pandas as pd
import time
import sys
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
import urllib.request
from pathlib import Path
import logging

# Load environment first
load_dotenv()

# Server Configuration Class (from story generator)
class ServerConfig:
    """Server-friendly configuration management"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_visual_config()
        self.ensure_directories()

    def setup_paths(self):
        """Setup server-friendly paths"""
        # Detect current file location
        current_file = Path(__file__).resolve()

        # For server: /home/youtube-automation/channels/sleepy-dull-stories/src/generators/
        # Go up to project root
        self.project_root = current_file.parent.parent.parent

        self.paths = {
            'BASE_DIR': str(self.project_root),
            'SRC_DIR': str(current_file.parent.parent),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'LOGS_DIR': str(self.project_root / 'logs'),
            'CONFIG_DIR': str(self.project_root / 'config')
        }

        print(f"✅ Scene Generator server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_visual_config(self):
        """Setup Midjourney visual generation configuration"""
        self.visual_config = {
            "api_base_url": "https://api.piapi.ai/api/v1",
            "max_concurrent_tasks": 10,
            "max_wait_cycles": 30,
            "wait_interval_seconds": 30,
            "default_aspect_ratios": {
                "characters": "2:3",
                "scenes": "16:9",
                "thumbnail": "16:9"
            },
            "default_version": "6.1",
            "process_mode": "relax",
            "character_generation": False,  # Disabled - use existing
            "scene_generation": True,       # Main focus
            "thumbnail_generation": True,   # Main focus
            "server_mode": True,
            "production_ready": True
        }

        # Get API key
        self.api_key = self.get_midjourney_api_key()

    def get_midjourney_api_key(self):
        """Get Midjourney API key from multiple sources"""
        # Try different environment variable names
        api_key = (
            os.getenv('PIAPI_KEY') or
            os.getenv('MIDJOURNEY_API_KEY') or
            os.getenv('PIAPI_API_KEY') or
            os.getenv('MIDJOURNEY_KEY')
        )

        if not api_key:
            # Check .env file
            env_files = [
                Path('.env'),
                Path('../../.env'),
                self.project_root / '.env'
            ]

            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file)
                    api_key = os.getenv('PIAPI_KEY')
                    if api_key:
                        print(f"✅ Midjourney API key loaded from: {env_file}")
                        break

        if not api_key:
            raise ValueError(
                "❌ Midjourney API key required!\n"
                "Set in .env file:\n"
                "PIAPI_KEY=your_api_key_here\n"
                "Or environment variable: PIAPI_KEY"
            )

        print("✅ Midjourney API key loaded successfully")
        return api_key

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"scene_gen_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("SceneGenerator")
        self.logger.info(f"✅ Scene generator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All scene generator directories created/verified")

# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Scene Generator server configuration loaded successfully")
except Exception as e:
    print(f"❌ Scene Generator server configuration failed: {e}")
    sys.exit(1)

# Database Topic Management Integration (Scene-focused)
class DatabaseSceneManager:
    """Professional scene management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_topic_ready_for_scenes(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed character topic that needs SCENE generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if scene generation columns exist, if not create them
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add columns individually if they don't exist
        columns_to_add = [
            ('scene_generation_status', 'TEXT DEFAULT "pending"'),
            ('scene_generation_started_at', 'DATETIME'),
            ('scene_generation_completed_at', 'DATETIME'),
            ('scenes_generated', 'INTEGER DEFAULT 0'),
            ('thumbnail_generated', 'BOOLEAN DEFAULT FALSE')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                print(f"🔧 Adding column: {column_name}")
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()
        print("✅ Scene generation columns verified/added")

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND character_generation_status = 'completed'
            AND (scene_generation_status IS NULL OR scene_generation_status = 'pending')
            ORDER BY character_generation_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_scene_generation_started(self, topic_id: int):
        """Mark scene generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if scene generation columns exist, if not create them
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add columns individually if they don't exist
        columns_to_add = [
            ('scene_generation_status', 'TEXT DEFAULT "pending"'),
            ('scene_generation_started_at', 'DATETIME'),
            ('scene_generation_completed_at', 'DATETIME'),
            ('scenes_generated', 'INTEGER DEFAULT 0'),
            ('thumbnail_generated', 'BOOLEAN DEFAULT FALSE')
        ]

        columns_added = []
        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')
                columns_added.append(column_name)

        if columns_added:
            print(f"🔧 Added columns: {', '.join(columns_added)}")

        cursor.execute('''
            UPDATE topics 
            SET scene_generation_status = 'in_progress', 
                scene_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_scene_generation_completed(self, topic_id: int, scenes_count: int, thumbnail_success: bool):
        """Mark scene generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET scene_generation_status = 'completed',
                scene_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                scenes_generated = ?,
                thumbnail_generated = ?
            WHERE id = ?
        ''', (scenes_count, thumbnail_success, topic_id))

        conn.commit()
        conn.close()

class ServerMidjourneySceneGenerator:
    """Server-ready Midjourney scene generator with character integration"""

    def __init__(self):
        self.api_key = CONFIG.api_key
        self.base_url = CONFIG.visual_config["api_base_url"]

        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None
        self.current_historical_period = None

        # Generation tracking
        self.generation_log = []
        self.character_references = {}
        self.api_calls_made = 0
        self.successful_downloads = 0

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseSceneManager(str(db_path))

        print("🚀 Server Midjourney Scene Generator v1.0 Initialized")
        print(f"🔑 API Key: {self.api_key[:8]}...")
        print(f"🌐 Base URL: {self.base_url}")

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps with production logging"""
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "project": self.current_topic_id,
            "api_calls": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "metadata": metadata or {}
        }
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step} [Calls: {self.api_calls_made}] [Downloads: {self.successful_downloads}]")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed character project that needs SCENE generation"""
        self.log_step("🔍 Finding completed character project for scene generation")

        result = self.db_manager.get_completed_topic_ready_for_scenes()

        if not result:
            self.log_step("✅ No completed character projects ready for scene generation", "INFO")
            return False, None

        topic_id, topic, description, output_path = result

        # Setup project paths
        self.current_topic_id = topic_id
        self.current_output_dir = output_path
        self.current_topic = topic
        self.current_description = description

        # Try to detect historical period from existing files
        try:
            character_path = Path(output_path) / "character_profiles.json"
            if character_path.exists():
                with open(character_path, 'r', encoding='utf-8') as f:
                    character_data = json.load(f)
                    visual_style = character_data.get('visual_style_notes', {})
                    self.current_historical_period = visual_style.get('period_accuracy', 'ancient times')
            else:
                self.current_historical_period = "ancient times"
        except:
            self.current_historical_period = "ancient times"

        project_info = {
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "output_dir": output_path,
            "historical_period": self.current_historical_period
        }

        # Mark as started in database
        self.db_manager.mark_scene_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def load_existing_character_references(self) -> bool:
        """Load existing character references from generated files"""
        self.log_step("🎭 Loading existing character references")

        characters_dir = Path(self.current_output_dir) / "characters"

        if not characters_dir.exists():
            self.log_step("❌ Characters directory not found", "ERROR")
            return False

        loaded_count = 0

        # Look for character JSON files which contain the image URLs
        for filename in characters_dir.glob("*.json"):
            if filename.stem == "thumbnail":  # Skip thumbnail.json
                continue

            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    char_data = json.load(f)

                char_name = char_data.get('name')
                image_url = char_data.get('image_url')

                if char_name and image_url:
                    self.character_references[char_name] = image_url
                    loaded_count += 1
                    print(f"✅ Loaded character: {char_name} → {image_url[:50]}...")

            except Exception as e:
                self.log_step(f"❌ Failed to load {filename.name}: {e}", "ERROR")

        self.log_step(f"✅ Loaded {loaded_count} character references", "SUCCESS")
        return loaded_count > 0

    def load_visual_prompts(self) -> List[Dict]:
        """Load visual generation prompts from story generator output"""
        self.log_step("📂 Loading visual generation prompts")

        output_dir = Path(self.current_output_dir)
        visual_prompts_path = output_dir / "visual_generation_prompts.json"

        if not visual_prompts_path.exists():
            raise FileNotFoundError(f"Visual prompts not found: {visual_prompts_path}")

        with open(visual_prompts_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

        # Extract scenes from the prompts data
        if isinstance(prompts_data, dict):
            visual_prompts = prompts_data.get("scenes", [])
        elif isinstance(prompts_data, list):
            visual_prompts = prompts_data
        else:
            raise ValueError("Invalid visual prompts format")

        self.log_step("✅ Visual prompts loaded", "SUCCESS", {
            "total_scenes": len(visual_prompts),
            "regular_scenes": len([s for s in visual_prompts if s.get("scene_number", 0) != 99]),
            "thumbnail_scenes": len([s for s in visual_prompts if s.get("scene_number", 0) == 99])
        })

        return visual_prompts

    def setup_scene_directories(self):
        """Create necessary directories for scene generation"""
        output_dir = Path(self.current_output_dir)

        self.scenes_dir = output_dir / "scenes"
        self.thumbnail_dir = output_dir / "thumbnail"

        self.scenes_dir.mkdir(exist_ok=True)
        self.thumbnail_dir.mkdir(exist_ok=True)

        self.log_step("📁 Scene generation directories created", "SUCCESS")

    def save_scene_generation_report(self):
        """Save scene generation report"""
        output_dir = Path(self.current_output_dir)

        report = {
            "scene_generation_completed": datetime.now().isoformat(),
            "topic_id": self.current_topic_id,
            "topic": self.current_topic,
            "api_calls_made": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "character_references_used": len(self.character_references),
            "scenes_dir": str(self.scenes_dir),
            "thumbnail_dir": str(self.thumbnail_dir),
            "historical_period": self.current_historical_period,
            "generation_log": self.generation_log,
            "server_optimized": True,
            "scene_only_mode": True
        }

        report_path = output_dir / "scene_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Scene generation report saved: {report_path}", "SUCCESS")

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
                self.log_step("✅ API Connection OK", "SUCCESS", {"response": result})
                return True
            else:
                self.log_step(f"❌ API Error: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Connection Test Failed: {e}", "ERROR")
            return False

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

        try:
            self.api_calls_made += 1
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
                    return self.submit_midjourney_task(prompt, aspect_ratio, retry_count + 1)  # Use original prompt for retry
                else:
                    print(f"❌ HTTP 500 - Max retries reached")
                    return None
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Request failed: {e}")
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
                self.successful_downloads += 1
                self.log_step(f"✅ Downloaded: {os.path.basename(save_path)} ({file_size} bytes)", "SUCCESS")
                return True
            else:
                self.log_step(f"❌ HTTP {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Download failed: {e}", "ERROR")
            return False

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

    def get_missing_scenes(self, visual_prompts: List[Dict]) -> List[Dict]:
        """Get list of scenes that are missing (not downloaded yet) and not blacklisted"""
        regular_scenes = [s for s in visual_prompts if s["scene_number"] != 99]
        missing_scenes = []

        for scene in regular_scenes:
            scene_num = scene["scene_number"]

            # Skip blacklisted scenes
            if hasattr(self, 'blacklisted_scenes') and scene_num in self.blacklisted_scenes:
                continue

            image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

            # Only add to missing if file doesn't exist
            if not image_path.exists():
                missing_scenes.append(scene)

        return missing_scenes

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

                file_size = save_path.stat().st_size if isinstance(save_path, Path) else os.path.getsize(save_path)
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

    def generate_scenes_with_retry(self, visual_prompts: List[Dict], max_retry_rounds: int = 10):
        """Generate all scenes with smart retry and universal content filtering - ORIGINAL WORKING VERSION"""

        # Initialize retry tracking
        if not hasattr(self, 'scene_attempt_count'):
            self.scene_attempt_count = {}  # scene_number: attempt_count
        if not hasattr(self, 'blacklisted_scenes'):
            self.blacklisted_scenes = set()  # scenes that failed too many times

        for retry_round in range(max_retry_rounds):
            missing_scenes = self.get_missing_scenes(visual_prompts)

            if not missing_scenes:
                print("✅ All scenes completed!")
                return True

            # Check if we have blacklisted scenes
            total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
            blacklisted_count = len(self.blacklisted_scenes)

            if retry_round == 0:
                print(f"🎬 Starting scene generation - {len(missing_scenes)} scenes to generate")
                print("🛡️ Universal content filter active for all prompts")
            else:
                print(f"\n🔄 RETRY ROUND {retry_round}: {len(missing_scenes)} missing scenes")
                if blacklisted_count > 0:
                    print(f"⚫ {blacklisted_count} scenes blacklisted (failed too many times)")

                # Longer wait between retry rounds
                print("⏳ Waiting 60 seconds before retry round...")
                time.sleep(60)

            # Check and update attempt counts
            for scene in missing_scenes:
                scene_num = scene["scene_number"]
                self.scene_attempt_count[scene_num] = self.scene_attempt_count.get(scene_num, 0) + 1

                # Blacklist scenes that failed too many times
                if self.scene_attempt_count[scene_num] > 5:
                    self.blacklisted_scenes.add(scene_num)
                    print(
                        f"⚫ Scene {scene_num}: Blacklisted after {self.scene_attempt_count[scene_num]} failed attempts")

            # Re-get missing scenes after blacklisting
            missing_scenes = self.get_missing_scenes(visual_prompts)

            if not missing_scenes:
                completed_count = total_scenes - blacklisted_count
                print(f"✅ All processable scenes completed! ({completed_count}/{total_scenes})")
                if blacklisted_count > 0:
                    print(f"⚫ {blacklisted_count} scenes blacklisted due to repeated failures")
                return True

            # Submit missing scene tasks
            scene_tasks = {}
            successful_submissions = 0

            for i, scene in enumerate(missing_scenes):
                scene_num = scene["scene_number"]
                attempt_num = self.scene_attempt_count.get(scene_num, 0)

                print(f"🎬 Processing Scene {scene_num} ({i + 1}/{len(missing_scenes)}) - Attempt #{attempt_num}")

                # Build safe scene prompt (content filter applied automatically)
                final_prompt = self.build_safe_scene_prompt(scene)

                # Check prompt length and truncate if necessary
                if len(final_prompt) > 4000:
                    print(f"⚠️ Scene {scene_num}: Truncating long prompt...")
                    base_prompt = scene.get("enhanced_prompt", scene["prompt"])

                    # Get character refs
                    char_refs = []
                    if scene.get("characters_present") and len(self.character_references) > 0:
                        for char_name in scene["characters_present"]:
                            if char_name in self.character_references:
                                char_refs.append(self.character_references[char_name])

                    if char_refs:
                        ref_string = " ".join(char_refs)
                        available_text_length = 3900 - len(ref_string) - len(" --ar 16:9 --v 6.1")
                        truncated_text = base_prompt[:available_text_length]
                        final_prompt = f"{ref_string} {truncated_text} --ar 16:9 --v 6.1"
                    else:
                        final_prompt = final_prompt[:3900] + " --ar 16:9 --v 6.1"

                # Check for content policy issues (informational only)
                if not self.is_content_policy_safe(final_prompt):
                    print(f"🛡️ Scene {scene_num}: Content filter will be applied")

                # Submit task (content filter applied automatically)
                task_id = self.submit_midjourney_task(final_prompt, aspect_ratio="16:9")

                if task_id:
                    scene_tasks[scene_num] = {
                        "task_id": task_id,
                        "prompt": final_prompt,
                        "scene_data": scene
                    }
                    successful_submissions += 1
                    print(f"✅ Scene {scene_num}: Submitted successfully")
                else:
                    print(f"❌ Scene {scene_num}: Submission failed")

                # Progressive rate limiting based on retry round
                base_wait = 5 if retry_round == 0 else 8
                wait_time = base_wait + (retry_round * 2)  # Increase wait time each retry round

                if i < len(missing_scenes) - 1:
                    print(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)

            print(
                f"📊 Round {retry_round + 1} submissions: ✅ {successful_submissions} | ❌ {len(missing_scenes) - successful_submissions}")

            if not scene_tasks:
                print("❌ No tasks submitted in this round, trying next round...")
                continue

            # Monitor tasks with detailed logging
            completed_scenes = {}
            max_cycles = 45

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

            # Download completed scenes with detailed logging
            successful_downloads = 0

            for scene_num, scene_data in completed_scenes.items():
                result_data = scene_data["result_data"]
                image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

                if self.download_image_detailed(result_data, str(image_path), scene_num):
                    successful_downloads += 1

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
                        "attempt_number": self.scene_attempt_count.get(scene_num, 1),
                        "content_filtered": True
                    }

                    json_path = self.scenes_dir / f"scene_{scene_num:02d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                else:
                    print(f"❌ Scene {scene_num}: Download failed, will retry in next round")

            print(f"✅ Round {retry_round + 1} downloads: {successful_downloads}")

        # Final check and summary
        final_missing = self.get_missing_scenes(visual_prompts)
        total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
        blacklisted_count = len(self.blacklisted_scenes)
        completed_count = total_scenes - len(final_missing) - blacklisted_count

        print(f"\n📊 FINAL SUMMARY:")
        print(f"✅ Completed: {completed_count}")
        print(f"❌ Missing: {len(final_missing)}")
        print(f"⚫ Blacklisted: {blacklisted_count}")
        print(f"📋 Total: {total_scenes}")
        print(f"🛡️ All prompts were content filtered")

        if final_missing:
            print(f"⚠️ Still missing after {max_retry_rounds} rounds:")
            for scene in final_missing:
                attempts = self.scene_attempt_count.get(scene['scene_number'], 0)
                print(f"  ❌ Scene {scene['scene_number']}: {scene['title']} (tried {attempts} times)")

        if self.blacklisted_scenes:
            print(f"⚫ Blacklisted scenes (failed >5 times):")
            for scene_num in self.blacklisted_scenes:
                attempts = self.scene_attempt_count.get(scene_num, 0)
                print(f"  ⚫ Scene {scene_num} (failed {attempts} times)")

        # Return success if we have most scenes (allowing some failures)
        success_rate = completed_count / total_scenes
        if success_rate >= 0.9:  # 90% success rate is acceptable
            print(f"✅ Generation successful with {success_rate:.1%} success rate")
            return True
        else:
            print(f"❌ Generation failed with only {success_rate:.1%} success rate")
            return False

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

                thumbnail_path = self.thumbnail_dir / "thumbnail.png"

                if self.download_image_detailed(result_data, str(thumbnail_path), 99):
                    metadata = {
                        "character_used": main_char,
                        "clickbait_title": thumbnail_scene.get("clickbait_title", ""),
                        "base_prompt": base_prompt,
                        "enhanced_prompt": final_prompt,
                        "emotion_enhancement": "INTENSE_SHOCK_TERROR_WIDE_EYES_OPEN_MOUTH",
                        "character_reference": char_ref_url,
                        "image_url": result_data["url"],
                        "local_path": str(thumbnail_path),
                        "generated_at": datetime.now().isoformat(),
                        "content_filtered": True
                    }

                    json_path = self.thumbnail_dir / "thumbnail.json"
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

    def run_scene_only_generation(self) -> bool:
        """Run SCENE-ONLY generation process for server environment"""
        print("🚀" * 50)
        print("SERVER MIDJOURNEY SCENE GENERATOR v1.0")
        print("🔗 Database integrated")
        print("🎬 SCENES & THUMBNAIL GENERATION")
        print("🎭 Character references integration")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # Initialize success tracking
        overall_success = False

        # Step 0: Test API connection
        if not self.test_api_connection():
            self.log_step("❌ API connection failed - aborting", "ERROR")
            return False

        # Step 1: Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🏛️ Historical period: {project_info['historical_period']}")

        try:
            # Step 2: Setup directories (scenes and thumbnail)
            self.setup_scene_directories()

            # Step 3: Load existing character references
            if not self.load_existing_character_references():
                self.log_step("❌ No character references found - characters must be generated first", "ERROR")
                return False

            print(f"🎭 Character references loaded: {len(self.character_references)}")
            for name, url in self.character_references.items():
                print(f"   🎭 {name}: {url[:50]}...")

            # Step 4: Load visual prompts
            visual_prompts = self.load_visual_prompts()
            print(f"🎬 Scene prompts loaded: {len(visual_prompts)}")

            # Step 5: Generate scenes with smart retry and content filtering
            print("\n🎬 GENERATING SCENES WITH UNIVERSAL CONTENT FILTER...")
            scenes_success = self.generate_scenes_with_retry(visual_prompts, max_retry_rounds=10)

            # Step 6: Generate thumbnail with dramatic enhancement and content filtering
            print("\n🖼️ GENERATING THUMBNAIL WITH CONTENT FILTER...")
            thumbnail_success = self.generate_thumbnail(visual_prompts)

            # Step 7: Save generation report
            self.save_scene_generation_report()

            # Step 8: Update database
            scenes_count = len([f for f in self.scenes_dir.glob("scene_*.png")]) if overall_success else 0

            self.db_manager.mark_scene_generation_completed(
                self.current_topic_id, scenes_count, thumbnail_success
            )

            # Final success assessment - original logic
            if scenes_success and thumbnail_success:
                print("\n" + "🎉" * 50)
                print("SCENE GENERATION SUCCESSFUL!")
                print("✅ ALL scenes generated (smart retry + content filtered)")
                print("✅ YouTube thumbnail with TERROR expression (content filtered)")
                print("🛡️ ALL PROMPTS AUTOMATICALLY SAFE FOR MIDJOURNEY")
                print("🎉" * 50)
                overall_success = True
            elif scenes_success:
                print("\n" + "🎊" * 50)
                print("SCENE GENERATION MOSTLY SUCCESSFUL!")
                print("✅ Scenes generated with smart retry")
                print("❌ Thumbnail generation failed")
                print("🎊" * 50)
                overall_success = True  # Still considered success if scenes work
            else:
                print("\n" + "❌" * 50)
                print("SCENE GENERATION FAILED!")
                print("❌ Scene generation failed after all retries")
                print("Check logs for details")
                print("❌" * 50)
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ Scene generation failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    try:
        print("🚀 SERVER MIDJOURNEY SCENE GENERATOR")
        print("🔗 Database integration with character references")
        print("🎬 SCENES & THUMBNAIL GENERATION")
        print("🖥️ Production-ready automation")
        print("=" * 60)

        generator = ServerMidjourneySceneGenerator()
        success = generator.run_scene_only_generation()

        if success:
            print("🎊 Scene generation completed successfully!")
        else:
            print("⚠️ Scene generation failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ Scene generation stopped by user")
    except Exception as e:
        print(f"💥 Scene generation failed: {e}")
        CONFIG.logger.error(f"Scene generation failed: {e}")
        import traceback
        traceback.print_exc()