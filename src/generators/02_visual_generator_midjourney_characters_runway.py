"""
Sleepy Dull Stories - ROBUST Runway ML Character Generator with Multi-Variant Support
✅ Uses official Runway ML SDK (pip install runwayml)
✅ Smart debug (only on errors)
✅ Skip existing characters
✅ Individual character error handling
✅ Partial success acceptance
✅ Multi-variant character poses (headshot, bust, profile, full_body, action)
✅ Runway ML SDK integration with Gen4 image references
"""

import os
import json
import time
import sys
import sqlite3
import signal
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import requests

# Load environment first
load_dotenv()

# Runway ML SDK import
try:
    from runwayml import RunwayML
    SDK_AVAILABLE = True
    print("✅ RunwayML SDK available")
except ImportError:
    print("❌ RunwayML SDK not found! Install with: pip install runwayml")
    sys.exit(1)


# Server Configuration Class
class ServerConfig:
    """Server-friendly configuration management for Runway ML"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_runway_config()
        self.ensure_directories()

    def setup_paths(self):
        """Setup server-friendly paths"""
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent

        self.paths = {
            'BASE_DIR': str(self.project_root),
            'SRC_DIR': str(current_file.parent.parent),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'LOGS_DIR': str(self.project_root / 'logs'),
            'CONFIG_DIR': str(self.project_root / 'config')
        }

        print(f"✅ Runway Character Generator server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_runway_config(self):
        """Setup Runway ML character generation configuration"""
        self.runway_config = {
            "model": "gen4_image",  # Gen4 Image model for character generation
            "max_concurrent_tasks": 3,  # Conservative for stability
            "max_wait_cycles": 60,  # Longer wait for Runway processing
            "wait_interval_seconds": 10,  # Check every 10 seconds
            "character_ratio": "1080:1920",  # Portrait ratio for characters
            "scene_ratio": "1920:1080",  # Landscape for scenes
            "character_generation": True,
            "scene_generation": False,  # Focus on characters only
            "server_mode": True,
            "production_ready": True,
            "runway_optimized": True
        }

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"runway_char_gen_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("RunwayCharacterGenerator")
        self.logger.info(f"✅ Runway character generator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = ['DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR']

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All Runway character generator directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Runway Character Generator server configuration loaded successfully")
except Exception as e:
    print(f"❌ Runway Character Generator server configuration failed: {e}")
    sys.exit(1)


# Database Topic Management Integration
class DatabaseTopicManager:
    """Professional topic management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_topic_ready_for_characters(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed story topic that needs CHARACTER generation only"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND (character_generation_status IS NULL OR character_generation_status = 'pending')
            ORDER BY production_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()
        return result if result else None

    def mark_character_generation_started(self, topic_id: int):
        """Mark character generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        if 'character_generation_status' not in columns:
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_status TEXT DEFAULT "pending"')
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_started_at DATETIME')
            cursor.execute('ALTER TABLE topics ADD COLUMN character_generation_completed_at DATETIME')
            cursor.execute('ALTER TABLE topics ADD COLUMN characters_generated INTEGER DEFAULT 0')

        cursor.execute('''
            UPDATE topics 
            SET character_generation_status = 'in_progress', 
                character_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_character_generation_completed(self, topic_id: int, characters_count: int):
        """Mark character generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET character_generation_status = 'completed',
                character_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                characters_generated = ?
            WHERE id = ?
        ''', (characters_count, topic_id))

        conn.commit()
        conn.close()

    def mark_character_generation_failed(self, topic_id: int, error_message: str):
        """Mark character generation as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET character_generation_status = 'failed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()


class ServerRunwayCharacterGenerator:
    """Server-ready Runway ML character generator using official SDK"""

    def __init__(self):
        # Initialize Runway ML client - SDK handles API key from env
        try:
            self.client = RunwayML()
            print("✅ Runway ML client initialized from RUNWAYML_API_SECRET")
        except Exception as e:
            print(f"❌ Failed to initialize Runway ML client: {e}")
            print("Make sure RUNWAYML_API_SECRET environment variable is set")
            sys.exit(1)

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
        self.db_manager = DatabaseTopicManager(str(db_path))

        print("🚀 Server Runway Character Generator v2.0 Initialized (SDK-powered)")

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

    def test_runway_connection(self) -> bool:
        """Test Runway ML SDK connection"""
        self.log_step("🔍 Testing Runway ML SDK connection")

        try:
            # Simple test generation
            test_response = self.client.text_to_image.create(
                model="gen4_image",
                prompt_text="A simple red apple on white background, photorealistic",
                ratio="1920:1080"
            )

            if hasattr(test_response, 'id'):
                self.log_step("✅ Runway SDK Connection OK", "SUCCESS", {"test_task_id": test_response.id})
                print(f"   📋 Test task ID: {test_response.id}")
                return True
            else:
                self.log_step("❌ SDK Response Error", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ SDK Connection Test Failed: {e}", "ERROR")
            return False

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed story project that needs CHARACTER generation"""
        self.log_step("🔍 Finding completed story project for character generation")

        result = self.db_manager.get_completed_topic_ready_for_characters()

        if not result:
            self.log_step("✅ No completed stories ready for character generation", "INFO")
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
        self.db_manager.mark_character_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def load_character_profiles_only(self) -> Dict:
        """Load character profiles only from story generator output"""
        self.log_step("📂 Loading character profiles only")

        output_dir = Path(self.current_output_dir)
        char_profiles_path = output_dir / "character_profiles.json"

        if not char_profiles_path.exists():
            raise FileNotFoundError(f"Character profiles not found: {char_profiles_path}")

        with open(char_profiles_path, 'r', encoding='utf-8') as f:
            character_profiles = json.load(f)

        main_characters = character_profiles.get("main_characters", [])

        self.log_step("✅ Character profiles loaded", "SUCCESS", {
            "characters_count": len(main_characters),
            "marketing_characters": len([c for c in main_characters if c.get("use_in_marketing", False)])
        })

        return character_profiles

    def setup_character_directories(self):
        """Create necessary directories for character generation only"""
        output_dir = Path(self.current_output_dir)
        self.characters_dir = output_dir / "characters"
        self.characters_dir.mkdir(exist_ok=True)
        self.log_step("📁 Character generation directory created", "SUCCESS")

    def extract_character_role(self, character: Dict) -> str:
        """Extract character role considering gender and age"""
        description = character.get('physical_description', '').lower()
        name = character.get('name', '').lower()

        # Role detection logic (simplified for this context)
        if 'baker' in description or 'bread' in description:
            role = "baker"
        elif 'soldier' in description or 'military' in description:
            role = "soldier"
        elif 'merchant' in description or 'trade' in description:
            role = "merchant"
        elif 'mother' in description or 'maternal' in description:
            role = "mother"
        else:
            role = "citizen"

        # Add historical context
        if 'roman' in self.current_historical_period.lower():
            return f"ancient Roman {role}"
        else:
            return f"historical {role}"

    def generate_runway_prompt(self, character: Dict, is_base: bool = False) -> str:
        """Generate Runway ML optimized prompt from character data"""
        name = character.get("name", "Unknown")
        physical = character.get("physical_description", "")
        importance_score = character.get("importance_score", 5)
        use_in_marketing = character.get("use_in_marketing", False)

        # Enhanced role detection
        character_role = self.extract_character_role(character)

        # Visual signature details for consistency
        visual_sig = character.get("visual_signature", {})
        hair = visual_sig.get("hair", "dark hair")
        face = visual_sig.get("face", "strong features")
        skin = visual_sig.get("skin", "olive skin")
        clothing = visual_sig.get("clothing", "Roman tunic")

        # Historical context
        period = self.current_historical_period

        # Pose selection based on importance
        if is_base or use_in_marketing or importance_score >= 8:
            pose = "front facing portrait, looking directly at camera"
        elif importance_score >= 7:
            pose = "three-quarter view portrait"
        else:
            pose = "profile portrait"

        # Runway ML optimized prompt for ancient Roman accuracy
        prompt = f"""79 CE Pompeii: {character_role} named {name}, {physical}, {hair}, {face}, {skin}, {clothing}, 
        {pose}, ancient Roman setting, historically accurate, cinematic lighting, photorealistic,
        8K resolution, professional portrait photography, no modern elements"""

        # Clean up the prompt
        clean_prompt = " ".join(prompt.split())

        print(f"   🎭 Generated Runway prompt for {name}: {clean_prompt[:100]}...")
        return clean_prompt

    def get_character_variants_for_consistency(self) -> List[Dict]:
        """Consistency focused variant list for Runway"""
        return [
            {"key": "profile", "pose": "side profile portrait", "framing": "clean profile view"},
            {"key": "three_quarter", "pose": "three-quarter view portrait", "framing": "slight turn, both eyes visible"},
            {"key": "upper_body", "pose": "upper body portrait, shoulders visible", "framing": "from waist up"},
            {"key": "full_body", "pose": "full-body standing pose", "framing": "full height, natural stance"}
        ]

    def generate_variant_prompt_with_reference(self, character: Dict, variant: Dict) -> str:
        """Generate variant prompt with Gen4 reference syntax"""
        name = character.get("name", "character")
        gender = character.get("gender", "person")

        # Visual signature details
        visual_sig = character.get("visual_signature", {})
        hair = visual_sig.get("hair", "dark hair")
        face = visual_sig.get("face", "strong features")
        skin = visual_sig.get("skin", "olive skin")

        # Generate prompt with reference syntax
        prompt = f"""79 CE Pompeii: Same character as @reference, ancient Roman {gender}, {hair}, {face}, {skin},
        {variant['pose']}, {variant['framing']}, historically accurate ancient Roman setting,
        photorealistic portrait, cinematic lighting, no modern elements"""

        clean_prompt = " ".join(prompt.split())
        return clean_prompt

    def upload_image_for_reference(self, image_path: str) -> Optional[str]:
        """Upload local image to get URL for reference (placeholder for now)"""
        # For now, we'll need to handle this differently
        # Runway SDK might require public URLs for references
        # This is a placeholder - in production you'd need image hosting
        return None

    def generate_character_with_runway(self, character: Dict, variant_info: Dict = None, reference_url: str = None) -> Optional[Dict]:
        """Generate single character using Runway ML SDK"""
        char_name = character["name"]
        variant_key = variant_info["key"] if variant_info else "base"

        try:
            self.api_calls_made += 1

            if variant_info and reference_url:
                # Generate variant with reference
                prompt = self.generate_variant_prompt_with_reference(character, variant_info)

                # Use Gen4 References format
                task_response = self.client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=prompt,
                    ratio=CONFIG.runway_config["character_ratio"],
                    reference_images=[{
                        "tag": "reference",
                        "uri": reference_url
                    }]
                )
            else:
                # Generate base portrait
                prompt = self.generate_runway_prompt(character, is_base=True)

                task_response = self.client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=prompt,
                    ratio=CONFIG.runway_config["character_ratio"]
                )

            if hasattr(task_response, 'id'):
                print(f"   ✅ Task submitted: {task_response.id} ({char_name} {variant_key})")

                # Wait for completion manually - SDK syntax is different
                try:
                    print(f"   ⏳ Waiting for {char_name} {variant_key} completion...")

                    # Manual polling since SDK method is different
                    max_wait_cycles = 60  # 10 minutes
                    wait_interval = 10  # 10 seconds

                    for cycle in range(max_wait_cycles):
                        try:
                            # Get task status
                            task_status = self.client.tasks.retrieve(task_response.id)

                            if hasattr(task_status, 'status'):
                                status = task_status.status.upper()

                                if status == "SUCCEEDED":
                                    if hasattr(task_status, 'output') and task_status.output:
                                        # Get image URL
                                        if isinstance(task_status.output, list) and len(task_status.output) > 0:
                                            image_url = task_status.output[0]
                                        else:
                                            image_url = str(task_status.output)

                                        print(f"   ✅ {char_name} {variant_key} completed after {cycle * wait_interval}s!")

                                        return {
                                            "url": image_url,
                                            "prompt": prompt,
                                            "task_id": task_response.id,
                                            "character": character,
                                            "variant": variant_info
                                        }
                                    else:
                                        print(f"   ❌ {char_name} {variant_key} completed but no output")
                                        return None

                                elif status == "FAILED":
                                    error_msg = getattr(task_status, 'error', 'Unknown error')
                                    print(f"   ❌ {char_name} {variant_key} failed: {error_msg}")
                                    return None

                                elif status in ["PENDING", "RUNNING", "THROTTLED"]:
                                    # Still processing, continue waiting
                                    if cycle % 6 == 0:  # Progress update every minute
                                        print(f"   ⏳ Still waiting for {char_name} {variant_key}... ({cycle * wait_interval}s)")
                                    time.sleep(wait_interval)
                                    continue

                                else:
                                    print(f"   ⚠️ Unknown status for {char_name} {variant_key}: {status}")
                                    time.sleep(wait_interval)
                                    continue
                            else:
                                print(f"   ⚠️ No status in response for {char_name} {variant_key}")
                                time.sleep(wait_interval)
                                continue

                        except Exception as poll_error:
                            print(f"   ⚠️ Poll error for {char_name} {variant_key}: {poll_error}")
                            time.sleep(wait_interval)
                            continue

                    print(f"   ⏰ {char_name} {variant_key} timed out after {max_wait_cycles * wait_interval}s")
                    return None

                except Exception as wait_error:
                    print(f"   ❌ {char_name} {variant_key} wait setup failed: {wait_error}")
                    return None
            else:
                print(f"   ❌ No task ID for {char_name} {variant_key}")
                return None

        except Exception as e:
            print(f"   ❌ Generation failed for {char_name} {variant_key}: {e}")
            return None

    def download_and_save_character(self, result_data: Dict, save_path: str, json_path: str) -> bool:
        """Download and save character image with metadata"""
        import requests

        try:
            # Download image
            response = requests.get(result_data["url"], timeout=60, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(save_path)
            self.successful_downloads += 1

            # Save metadata
            metadata = {
                "name": result_data["character"]["name"],
                "variant": result_data["variant"]["key"] if result_data["variant"] else "base",
                "prompt": result_data["prompt"],
                "image_url": result_data["url"],
                "local_path": str(save_path),
                "task_id": result_data["task_id"],
                "generated_at": datetime.now().isoformat(),
                "runway_powered": True,
                "runway_model": "gen4_image",
                "importance_score": result_data["character"].get("importance_score", 0),
                "use_in_marketing": result_data["character"].get("use_in_marketing", False)
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"   💾 Saved: {os.path.basename(save_path)} ({file_size:,} bytes)")
            return True

        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            return False

    def generate_all_characters_runway(self, character_profiles: Dict) -> bool:
        """Generate all characters using Runway ML SDK with consistency"""
        self.log_step("🎭 Starting Runway Character Consistency Generation")

        main_characters = character_profiles.get("main_characters", [])

        # Filter for marketing and important characters
        marketing_characters = [char for char in main_characters if char.get("use_in_marketing", False)]
        important_characters = [char for char in main_characters if char.get("importance_score", 0) >= 7 and not char.get("use_in_marketing", False)]

        characters_to_process = marketing_characters + important_characters

        if not characters_to_process:
            self.log_step("❌ No marketing or important characters found", "ERROR")
            return False

        print(f"📊 Processing {len(characters_to_process)} characters:")
        print(f"   🎯 Marketing: {len(marketing_characters)} - {[c['name'] for c in marketing_characters]}")
        print(f"   ⭐ Important: {len(important_characters)} - {[c['name'] for c in important_characters]}")

        # PHASE 1: Generate base portraits
        print(f"\n🎯 PHASE 1: Generating base portraits")
        base_portraits = {}

        for character in characters_to_process:
            char_name = character["name"]
            safe_name = char_name.lower().replace(" ", "_").replace(".", "")
            base_path = self.characters_dir / f"{safe_name}_base.png"
            json_path = self.characters_dir / f"{safe_name}_base.json"

            # Skip if exists
            if base_path.exists() and base_path.stat().st_size > 1024:
                print(f"   ✅ {char_name} base already exists")
                base_portraits[char_name] = str(base_path)
                continue

            print(f"   🎨 Generating base portrait: {char_name}")

            result_data = self.generate_character_with_runway(character)
            if result_data:
                if self.download_and_save_character(result_data, str(base_path), str(json_path)):
                    base_portraits[char_name] = result_data["url"]  # Store URL for reference
                    print(f"   ✅ Base portrait completed: {char_name}")
                else:
                    print(f"   ❌ Base portrait save failed: {char_name}")
            else:
                print(f"   ❌ Base portrait generation failed: {char_name}")

            time.sleep(2)  # Rate limiting

        if not base_portraits:
            self.log_step("❌ No base portraits generated", "ERROR")
            return False

        print(f"\n📊 Base portraits: {len(base_portraits)}/{len(characters_to_process)}")

        # PHASE 2: Generate variants with references
        print(f"\n🎯 PHASE 2: Generating consistent variants")

        total_variants = 0
        successful_variants = 0

        for character in characters_to_process:
            char_name = character["name"]
            safe_name = char_name.lower().replace(" ", "_").replace(".", "")

            if char_name not in base_portraits:
                print(f"   ⚠️ Skipping {char_name} - no base portrait")
                continue

            reference_url = base_portraits[char_name]
            variants = self.get_character_variants_for_consistency()

            print(f"\n🎭 Generating variants for: {char_name}")

            for variant in variants:
                variant_key = variant["key"]
                variant_path = self.characters_dir / f"{safe_name}_{variant_key}.png"
                variant_json = self.characters_dir / f"{safe_name}_{variant_key}.json"

                # Skip if exists
                if variant_path.exists() and variant_path.stat().st_size > 1024:
                    print(f"   ✅ {variant_key} already exists")
                    successful_variants += 1
                    total_variants += 1
                    continue

                print(f"   📝 Generating {variant_key}: {variant['pose']}")
                total_variants += 1

                result_data = self.generate_character_with_runway(
                    character,
                    variant_info=variant,
                    reference_url=reference_url
                )

                if result_data:
                    if self.download_and_save_character(result_data, str(variant_path), str(variant_json)):
                        successful_variants += 1
                        print(f"   ✅ {variant_key} completed")
                    else:
                        print(f"   ❌ {variant_key} save failed")
                else:
                    print(f"   ❌ {variant_key} generation failed")

                time.sleep(1)  # Rate limiting

        # Update character references
        self.character_references = base_portraits

        # Final assessment
        success_rate = successful_variants / total_variants if total_variants > 0 else 0
        is_successful = len(base_portraits) >= 1 and success_rate >= 0.5

        print(f"\n📊 FINAL RUNWAY CHARACTER GENERATION SUMMARY:")
        print(f"   🎨 Base portraits: {len(base_portraits)}")
        print(f"   📸 Total variants: {successful_variants}/{total_variants}")
        print(f"   📈 Success rate: {success_rate:.1%}")
        print(f"   🚀 Powered by: Runway ML Gen4 Image")

        self.log_step(
            f"✅ Runway Character Generation {'SUCCESS' if is_successful else 'PARTIAL'}: "
            f"{len(base_portraits)} base + {successful_variants}/{total_variants} variants",
            "SUCCESS" if is_successful else "ERROR"
        )

        return is_successful

    def save_character_generation_report(self):
        """Save character generation report"""
        output_dir = Path(self.current_output_dir)

        report = {
            "character_generation_completed": datetime.now().isoformat(),
            "topic_id": self.current_topic_id,
            "topic": self.current_topic,
            "api_calls_made": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "character_references_created": len(self.character_references),
            "characters_dir": str(self.characters_dir),
            "historical_period": self.current_historical_period,
            "generation_log": self.generation_log,
            "runway_sdk_version": True,
            "runway_model": "gen4_image",
            "server_optimized": True,
            "character_only_mode": True
        }

        report_path = output_dir / "character_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Character generation report saved: {report_path}", "SUCCESS")

    def run_character_only_generation(self) -> bool:
        """Run CHARACTER-ONLY generation process with Runway ML SDK"""
        print("🚀" * 50)
        print("SERVER RUNWAY CHARACTER GENERATOR v2.0 (SDK-powered)")
        print("🔗 Database integrated")
        print("🎭 ROBUST CHARACTER GENERATION")
        print("✅ Skip existing | ✅ Individual error handling | ✅ Smart debug")
        print("🖥️ Production-ready automation")
        print("🚀 Powered by Runway ML SDK")
        print("🚀" * 50)

        # Test SDK connection
        if not self.test_runway_connection():
            self.log_step("❌ Runway SDK connection failed - aborting", "ERROR")
            return False

        # Get next project
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🏛️ Historical period: {project_info['historical_period']}")

        try:
            # Setup and process
            self.setup_character_directories()
            character_profiles = self.load_character_profiles_only()
            characters_success = self.generate_all_characters_runway(character_profiles)
            self.save_character_generation_report()

            # Update database
            characters_count = len(self.character_references)
            if characters_success:
                self.db_manager.mark_character_generation_completed(self.current_topic_id, characters_count)
            else:
                self.db_manager.mark_character_generation_failed(self.current_topic_id, "Character generation failed")

            # Final status
            if characters_success:
                print("\n" + "🎉" * 50)
                print("RUNWAY CHARACTER GENERATION SUCCESSFUL!")
                print(f"✅ Characters Generated: {characters_count}")
                print(f"✅ API Calls: {self.api_calls_made}")
                print(f"✅ Downloads: {self.successful_downloads}")
                print(f"📁 Saved to: {self.characters_dir}")
                print(f"🚀 Powered by: Runway ML Gen4 Image SDK")
                print("🎉" * 50)
            else:
                print("\n" + "❌" * 50)
                print("RUNWAY CHARACTER GENERATION FAILED!")
                print("Check logs for details")
                print("❌" * 50)

            return characters_success

        except Exception as e:
            self.log_step(f"❌ Character generation failed: {e}", "ERROR")
            self.db_manager.mark_character_generation_failed(self.current_topic_id, str(e))
            import traceback
            traceback.print_exc()
            return False


def run_autonomous_mode():
    """Run autonomous mode - continuously process completed story topics"""
    print("🤖 AUTONOMOUS RUNWAY CHARACTER GENERATION MODE STARTED")
    print("🔄 Will process all completed story topics continuously")
    print("🚀 Powered by Runway ML SDK")
    print("⏹️ Press Ctrl+C to stop gracefully")

    running = True
    processed_count = 0
    cycles_count = 0
    start_time = time.time()
    last_activity_time = time.time()

    def signal_handler(signum, frame):
        nonlocal running
        print(f"\n⏹️ Received shutdown signal ({signum})")
        print("🔄 Finishing current character generation and shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            character_generator = ServerRunwayCharacterGenerator()
            found, project_info = character_generator.get_next_project_from_database()

            if found:
                cycle_had_work = False
                cycle_start_time = time.time()

                print(f"\n🔄 CYCLE {cycles_count + 1}: Found completed story topics ready for character generation")

                while running and found:
                    success = character_generator.run_character_only_generation()

                    if success:
                        processed_count += 1
                        cycle_had_work = True
                        last_activity_time = time.time()
                        print(f"✅ Character generation completed! (Topic {processed_count})")
                    else:
                        print(f"⚠️ Character generation failed")
                        break

                    if running:
                        time.sleep(5)

                    character_generator = ServerRunwayCharacterGenerator()
                    found, project_info = character_generator.get_next_project_from_database()

                cycles_count += 1
                cycle_time = time.time() - cycle_start_time

                if cycle_had_work:
                    print(f"\n📊 CYCLE {cycles_count} COMPLETED:")
                    print(f"   ✅ Topics processed this cycle: {processed_count}")
                    print(f"   ⏱️ Cycle time: {cycle_time:.1f} seconds")

                if running:
                    print("⏳ Pausing 15 seconds before next cycle...")
                    time.sleep(15)

            else:
                time_since_activity = time.time() - last_activity_time

                if time_since_activity < 300:
                    wait_time = 60
                    print("😴 No topics ready. Recent activity detected - waiting 60s...")
                else:
                    wait_time = 3600
                    print("😴 No topics ready for extended period - waiting 1 hour...")

                for i in range(wait_time):
                    if not running:
                        break
                    if i > 0 and i % 300 == 0:
                        remaining = (wait_time - i) / 60
                        print(f"⏳ Still waiting... {remaining:.1f} minutes remaining")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️ Keyboard interrupt received")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("⏳ Waiting 30 seconds before retry...")
            time.sleep(30)

    runtime = time.time() - start_time
    print(f"\n🏁 AUTONOMOUS RUNWAY CHARACTER GENERATION SHUTDOWN")
    print(f"⏱️ Total runtime: {runtime / 3600:.1f} hours")
    print(f"🔄 Total cycles: {cycles_count}")
    print(f"✅ Topics processed: {processed_count}")
    print(f"🚀 Powered by: Runway ML SDK")
    print("👋 Goodbye!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--autonomous':
        run_autonomous_mode()
    else:
        try:
            print("🚀 SERVER RUNWAY CHARACTER GENERATOR v2.0")
            print("🔗 Database integration with story generator")
            print("🎭 ROBUST CHARACTER GENERATION")
            print("✅ Skip existing | ✅ Individual error handling | ✅ Smart debug")
            print("🖥️ Production-ready automation")
            print("🚀 Powered by Runway ML SDK")
            print("=" * 60)

            generator = ServerRunwayCharacterGenerator()
            success = generator.run_character_only_generation()

            if success:
                print("🎊 Runway character generation completed successfully!")
            else:
                print("⚠️ Character generation failed or no projects ready")

        except KeyboardInterrupt:
            print("\n⏹️ Character generation stopped by user")
        except Exception as e:
            print(f"💥 Character generation failed: {e}")
            CONFIG.logger.error(f"Character generation failed: {e}")
            import traceback
            traceback.print_exc()