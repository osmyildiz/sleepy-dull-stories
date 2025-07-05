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

    def load_character_references_and_scene_prompts(self) -> Tuple[Dict, List[Dict]]:
        """Load existing character references and scene prompts"""
        self.log_step("📂 Loading character references and scene prompts")

        output_dir = Path(self.current_output_dir)

        # Load character references from characters directory
        characters_dir = output_dir / "characters"
        if characters_dir.exists():
            for char_file in characters_dir.glob("*.json"):
                if char_file.stem != "thumbnail":  # Skip thumbnail.json
                    try:
                        with open(char_file, 'r', encoding='utf-8') as f:
                            char_data = json.load(f)
                            char_name = char_data.get("name")
                            image_url = char_data.get("image_url")
                            if char_name and image_url:
                                self.character_references[char_name] = image_url
                    except Exception as e:
                        print(f"⚠️ Failed to load character: {char_file} - {e}")

        # Load visual generation prompts
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

        self.log_step("✅ Character references and scene prompts loaded", "SUCCESS", {
            "character_references": len(self.character_references),
            "total_scenes": len(visual_prompts),
            "regular_scenes": len([s for s in visual_prompts if s.get("scene_number", 0) != 99]),
            "thumbnail_scenes": len([s for s in visual_prompts if s.get("scene_number", 0) == 99])
        })

        return self.character_references, visual_prompts

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

    def submit_midjourney_task(self, prompt: str, aspect_ratio: str = "16:9") -> Optional[str]:
        """Submit task to Midjourney API"""
        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "process_mode": CONFIG.visual_config["process_mode"],
                "skip_prompt_check": False
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
                self.successful_downloads += 1
                self.log_step(f"✅ Downloaded: {os.path.basename(save_path)} ({file_size} bytes)", "SUCCESS")
                return True
            else:
                self.log_step(f"❌ HTTP {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Download failed: {e}", "ERROR")
            return False

    def generate_all_scenes_parallel(self, visual_prompts: List[Dict]):
        """Generate all scenes in parallel with character references"""
        self.log_step("🎬 Starting parallel scene generation")

        # Get regular scenes (not thumbnail)
        regular_scenes = [s for s in visual_prompts if s.get("scene_number", 0) != 99]

        if not regular_scenes:
            self.log_step("❌ No scenes found", "ERROR")
            return False

        print(f"🎬 Found {len(regular_scenes)} scenes to generate")
        print(f"🎭 Available character references: {len(self.character_references)}")

        # Submit all scene tasks
        scene_tasks = {}

        for scene in regular_scenes:
            scene_num = scene.get("scene_number", 0)

            # Build scene prompt with character references
            base_prompt = scene.get("enhanced_prompt", scene.get("prompt", ""))

            # Add character references if characters are present in scene
            characters_present = scene.get("characters_present", [])
            if characters_present and len(self.character_references) > 0:
                char_refs = []
                char_names = []
                for char_name in characters_present:
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
        max_cycles = CONFIG.visual_config["max_wait_cycles"]

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
            time.sleep(CONFIG.visual_config["wait_interval_seconds"])

        # Download all completed scenes
        successful_downloads = 0

        for scene_num, scene_data in completed_scenes.items():
            result_data = scene_data["result_data"]

            image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

            if self.download_image(result_data, str(image_path)):
                successful_downloads += 1

                # Save metadata
                metadata = {
                    "scene_number": scene_num,
                    "title": scene_data["task_data"]["scene_data"].get("title", f"Scene {scene_num}"),
                    "prompt": scene_data["task_data"]["prompt"],
                    "image_url": result_data["url"],
                    "url_source": result_data["source"],
                    "local_path": str(image_path),
                    "generated_at": datetime.now().isoformat()
                }

                json_path = self.scenes_dir / f"scene_{scene_num:02d}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Scene generation complete: {successful_downloads}/{len(regular_scenes)}", "SUCCESS")

        return successful_downloads > 0

    def generate_thumbnail(self, visual_prompts: List[Dict]):
        """Generate thumbnail image"""
        self.log_step("🖼️ Starting thumbnail generation")

        # Find thumbnail scene (scene_number == 99)
        thumbnail_scene = next((s for s in visual_prompts if s.get("scene_number", 0) == 99), None)

        if not thumbnail_scene:
            self.log_step("❌ No thumbnail scene found (scene 99)", "ERROR")
            return False

        # Build thumbnail prompt
        base_prompt = thumbnail_scene.get("prompt", "")

        # Add character references if available
        character_used = thumbnail_scene.get("character_used", "")
        if character_used and character_used in self.character_references:
            char_ref = self.character_references[character_used]
            base_prompt = f"{base_prompt} {char_ref}"
            print(f"🖼️ Thumbnail: Added character reference for {character_used}")

        final_prompt = f"{base_prompt} --ar 16:9 --v 6.1"

        # Submit thumbnail task
        task_id = self.submit_midjourney_task(final_prompt, aspect_ratio="16:9")
        if not task_id:
            return False

        self.log_step(f"✅ Thumbnail task submitted: {task_id}", "SUCCESS")

        # Monitor thumbnail task
        max_cycles = CONFIG.visual_config["max_wait_cycles"]

        for cycle in range(max_cycles):
            result_data = self.check_task_status(task_id)

            if result_data and isinstance(result_data, dict):
                # Thumbnail completed!
                self.log_step("✅ Thumbnail generation completed!", "SUCCESS")

                image_path = self.thumbnail_dir / "thumbnail.png"

                if self.download_image(result_data, str(image_path)):
                    # Save metadata
                    metadata = {
                        "scene_number": 99,
                        "type": "thumbnail",
                        "character_used": character_used,
                        "prompt": final_prompt,
                        "image_url": result_data["url"],
                        "url_source": result_data["source"],
                        "local_path": str(image_path),
                        "generated_at": datetime.now().isoformat()
                    }

                    json_path = self.thumbnail_dir / "thumbnail.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    return True
                else:
                    return False

            elif result_data is False:
                self.log_step("❌ Thumbnail generation failed", "ERROR")
                return False
            else:
                print(f"⏳ Thumbnail processing... (cycle {cycle + 1}/{max_cycles})")
                time.sleep(CONFIG.visual_config["wait_interval_seconds"])

        self.log_step("⏰ Thumbnail generation timeout", "ERROR")
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

            # Step 3: Load character references and scene prompts
            character_refs, visual_prompts = self.load_character_references_and_scene_prompts()

            print(f"🎭 Character references loaded: {len(character_refs)}")
            print(f"🎬 Scene prompts loaded: {len(visual_prompts)}")

            # Step 4: Generate scenes in parallel
            scenes_success = self.generate_all_scenes_parallel(visual_prompts)

            # Step 5: Generate thumbnail
            thumbnail_success = self.generate_thumbnail(visual_prompts)

            # Step 6: Save generation report
            self.save_scene_generation_report()

            # Step 7: Update database
            scenes_count = len([f for f in self.scenes_dir.glob("scene_*.png")])

            self.db_manager.mark_scene_generation_completed(
                self.current_topic_id, scenes_count, thumbnail_success
            )

            # Final success assessment
            overall_success = scenes_success or thumbnail_success

            if overall_success:
                print("\n" + "🎉" * 50)
                print("SCENE GENERATION SUCCESSFUL!")
                print(f"✅ Scenes Generated: {scenes_count}")
                print(f"✅ Thumbnail Generated: {'Yes' if thumbnail_success else 'No'}")
                print(f"✅ API Calls: {self.api_calls_made}")
                print(f"✅ Downloads: {self.successful_downloads}")
                print(f"📁 Scenes saved to: {self.scenes_dir}")
                print(f"📁 Thumbnail saved to: {self.thumbnail_dir}")
                print("🎉" * 50)
            else:
                print("\n" + "❌" * 50)
                print("SCENE GENERATION FAILED!")
                print("Check logs for details")
                print("❌" * 50)

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