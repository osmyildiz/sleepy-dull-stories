"""
Sleepy Dull Stories - RUNWAY ML Scene Generator with Character References
Uses existing character portraits from characters/ folder as references for scene generation
Runway ML Gen4 Image with reference system for consistent characters across scenes
"""

import os
import json
import time
import sys
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import base64

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
    """Server-friendly configuration management for Runway ML Scene Generation"""

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

        print(f"✅ Runway Scene Generator server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_runway_config(self):
        """Setup Runway ML scene generation configuration"""
        self.runway_config = {
            "model": "gen4_image",  # Gen4 Image model
            "max_concurrent_tasks": 2,  # Conservative for scenes
            "max_wait_cycles": 80,  # Longer wait for complex scenes
            "wait_interval_seconds": 15,  # Check every 15 seconds
            "scene_ratio": "1920:1080",  # Landscape for scenes
            "character_ratio": "1080:1920",  # Portrait for character closeups
            "scene_generation": True,
            "thumbnail_generation": True,
            "server_mode": True,
            "production_ready": True,
            "runway_optimized": True
        }

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"runway_scene_gen_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("RunwaySceneGenerator")
        self.logger.info(f"✅ Runway scene generator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = ['DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR']

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All Runway scene generator directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Runway Scene Generator server configuration loaded successfully")
except Exception as e:
    print(f"❌ Runway Scene Generator server configuration failed: {e}")
    sys.exit(1)


# Database Scene Management
class DatabaseSceneManager:
    """Professional scene management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_topic_ready_for_scenes(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed character topic that needs SCENE generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check and add scene generation columns
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

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

    def mark_scene_generation_failed(self, topic_id: int, error_message: str):
        """Mark scene generation as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET scene_generation_status = 'failed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()


# Character Reference Manager
class CharacterReferenceManager:
    """Manages character references from characters/ folder for scene generation"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.characters_dir = self.output_dir / "characters"
        self.character_references = {}
        self.primary_character = None
        self.load_character_references()

    def load_character_references(self) -> bool:
        """Load character references from JSON files in characters/ folder"""
        print("🎭 Loading character references from characters/ folder")

        if not self.characters_dir.exists():
            print("❌ Characters directory not found")
            return False

        loaded_count = 0

        # Load all character JSON files
        for json_file in self.characters_dir.glob("*_base.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    char_data = json.load(f)

                char_name = char_data.get('name')
                image_url = char_data.get('image_url')
                local_path = char_data.get('local_path')
                use_in_marketing = char_data.get('use_in_marketing', False)
                importance_score = char_data.get('importance_score', 0)

                if char_name and (image_url or local_path):
                    # Check if URL is still valid (Runway URLs expire 24-48h)
                    reference_source = self.get_best_reference_source(image_url, local_path)

                    if reference_source:
                        self.character_references[char_name] = {
                            "image_url": image_url,
                            "local_path": local_path,
                            "reference_source": reference_source,
                            "use_in_marketing": use_in_marketing,
                            "importance_score": importance_score,
                            "variants": {}
                        }
                        loaded_count += 1
                        print(f"   ✅ {char_name}: {reference_source['type']} reference loaded")

                        # Set primary character (highest importance or marketing)
                        if (use_in_marketing or importance_score >= 8) and not self.primary_character:
                            self.primary_character = char_name
                            print(f"   🎯 Primary character: {char_name}")

            except Exception as e:
                print(f"   ❌ Failed to load {json_file.name}: {e}")

        # Load variants for each character
        self.load_character_variants()

        print(f"✅ Loaded {loaded_count} character references")
        if not self.primary_character and self.character_references:
            # Fallback to first character
            self.primary_character = list(self.character_references.keys())[0]
            print(f"🎯 Fallback primary character: {self.primary_character}")

        return loaded_count > 0

    def load_character_variants(self):
        """Load character variants (profile, three_quarter, etc.) for each character"""
        for char_name in self.character_references.keys():
            safe_name = char_name.lower().replace(" ", "_").replace(".", "")

            # Look for variant files
            variant_types = ["profile", "three_quarter", "upper_body", "full_body"]

            for variant_type in variant_types:
                variant_json = self.characters_dir / f"{safe_name}_{variant_type}.json"

                if variant_json.exists():
                    try:
                        with open(variant_json, 'r', encoding='utf-8') as f:
                            variant_data = json.load(f)

                        image_url = variant_data.get('image_url')
                        local_path = variant_data.get('local_path')
                        reference_source = self.get_best_reference_source(image_url, local_path)

                        if reference_source:
                            self.character_references[char_name]["variants"][variant_type] = {
                                "image_url": image_url,
                                "local_path": local_path,
                                "reference_source": reference_source
                            }
                            print(f"   📐 {char_name} {variant_type}: loaded")

                    except Exception as e:
                        print(f"   ⚠️ Failed to load {char_name} {variant_type}: {e}")

    def get_best_reference_source(self, image_url: str, local_path: str) -> Optional[Dict]:
        """Get the best available reference source (URL vs local file)"""

        # First, try to use the original Runway URL if it's recent
        if image_url and self.is_runway_url_likely_valid(image_url):
            return {"type": "url", "source": image_url}

        # Fallback to local file converted to data URI
        if local_path and Path(local_path).exists():
            try:
                data_uri = self.convert_local_to_data_uri(local_path)
                if data_uri:
                    return {"type": "data_uri", "source": data_uri}
            except Exception as e:
                print(f"   ⚠️ Failed to convert {local_path} to data URI: {e}")

        return None

    def is_runway_url_likely_valid(self, url: str) -> bool:
        """Check if Runway URL is likely still valid (simple heuristic)"""
        if not url:
            return False

        # Runway URLs expire in 24-48 hours
        # For now, assume they're valid if they look like Runway URLs
        runway_domains = ["dnznrvs05pmza.cloudfront.net", "storage.googleapis.com", "cdn.runwayml.com"]

        return any(domain in url for domain in runway_domains)

    def convert_local_to_data_uri(self, image_path: str) -> Optional[str]:
        """Convert local image to data URI for Runway API"""
        try:
            with open(image_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"❌ Failed to encode {image_path}: {e}")
            return None

    def get_character_reference(self, char_name: str, variant: str = "base") -> Optional[str]:
        """Get character reference URI for scene generation"""

        if char_name not in self.character_references:
            return None

        char_data = self.character_references[char_name]

        # Get specific variant if requested and available
        if variant != "base" and variant in char_data["variants"]:
            reference_data = char_data["variants"][variant]
        else:
            # Use base character reference
            reference_data = char_data

        reference_source = reference_data.get("reference_source")
        if reference_source:
            return reference_source["source"]

        return None

    def get_scene_character_references(self, characters_present: List[str], scene_content: str = "") -> List[Dict]:
        """Get character references for scene generation in Runway format"""
        references = []

        # PHASE 1: Add specific characters that are actually in the scene
        valid_characters = []
        for char_name in characters_present:
            if char_name == "Entire City" or char_name not in self.character_references:
                continue
            valid_characters.append(char_name)

        # Add references for valid characters
        for i, char_name in enumerate(valid_characters):
            reference_uri = self.get_character_reference(char_name)
            if reference_uri:
                references.append({
                    "tag": f"char{i+1}",  # char1, char2, etc.
                    "uri": reference_uri
                })
                print(f"   📌 Added reference: {char_name} as @char{i+1}")

        # PHASE 2: Decide if we should use primary character for consistency
        # Only use primary character if:
        # 1. No specific characters AND
        # 2. Scene content suggests character presence (not pure landscape/cityscape)
        if not references:
            should_use_primary = self.should_use_primary_character_reference(scene_content, characters_present)

            if should_use_primary and self.primary_character:
                primary_ref = self.get_character_reference(self.primary_character)
                if primary_ref:
                    references.append({
                        "tag": "primary",
                        "uri": primary_ref
                    })
                    print(f"   🎯 Using primary character: {self.primary_character} as @primary")
            else:
                print(f"   🌆 No character references - pure scene/landscape generation")

        return references

    def should_use_primary_character_reference(self, scene_content: str, characters_present: List[str]) -> bool:
        """Determine if primary character reference should be used for consistency"""

        # Check if characters_present indicates no characters or unknown characters
        non_character_indicators = [
            "Entire City", "No Characters", "Landscape", "Cityscape",
            "other servants", "family members", "crowd", "people",
            "citizens", "soldiers", "children", "villagers", "townspeople"
        ]

        for indicator in non_character_indicators:
            if indicator in characters_present:
                print(f"   👥 Generic characters detected: '{indicator}' - skipping character reference")
                return False

        # Check if characters_present contains character names we don't have references for
        known_characters = set(self.character_references.keys())
        scene_characters = set(characters_present)

        # If scene has specific character names but we don't have references for them
        if scene_characters and not scene_characters.intersection(known_characters):
            print(f"   👤 Unknown characters: {list(scene_characters)} - skipping character reference")
            return False

        # Check scene content for landscape/cityscape indicators
        landscape_indicators = [
            "wide shot", "wide establishing shot", "aerial view", "bird's eye view",
            "cityscape", "landscape", "panoramic", "overview", "distant view",
            "entire city", "whole city", "city from above", "harbor view",
            "mountain view", "volcano", "vesuvius", "streets from above",
            "rooftops", "city walls", "distant", "far shot", "establishing shot"
        ]

        scene_lower = scene_content.lower()

        # If scene content suggests pure landscape/cityscape, don't use character reference
        for indicator in landscape_indicators:
            if indicator in scene_lower:
                print(f"   🌆 Detected landscape scene: '{indicator}' - skipping character reference")
                return False

        # Check for scenes that should NOT have our main characters
        other_character_indicators = [
            "servants", "slaves", "different family", "other household",
            "stranger", "unknown person", "guest", "visitor", "merchant",
            "soldier", "guard", "priest", "official", "woman", "man",
            "child", "elder", "young", "old"
        ]

        for indicator in other_character_indicators:
            if indicator in scene_lower:
                print(f"   👥 Other character scene: '{indicator}' - skipping our character reference")
                return False

        # Check for indoor/character-focused scenes with our known characters
        character_scene_indicators = [
            "close-up", "medium shot", "interior", "inside", "chamber", "room",
            "kitchen", "bakery", "workshop", "house", "home", "intimate",
            "conversation", "talking", "speaking", "working", "eating",
            "sleeping", "resting", "thinking", "contemplating"
        ]

        # Only use primary character if it's a character scene AND we have no specific characters
        for indicator in character_scene_indicators:
            if indicator in scene_lower:
                # But only if the scene could reasonably include our main character
                if any(known_char.lower() in scene_lower for known_char in known_characters):
                    print(f"   👤 Our character scene: '{indicator}' - using primary reference")
                    return True
                else:
                    print(f"   👤 Other character scene: '{indicator}' - not our character")
                    return False

        # Default: if unclear, don't use character reference to avoid unwanted appearances
        print(f"   🤷 Unclear scene type - defaulting to no character reference")
        return False

    def get_summary(self) -> Dict:
        """Get summary of loaded character references"""
        total_chars = len(self.character_references)
        total_variants = sum(len(char["variants"]) for char in self.character_references.values())

        return {
            "total_characters": total_chars,
            "total_variants": total_variants,
            "primary_character": self.primary_character,
            "characters": list(self.character_references.keys())
        }


class ServerRunwaySceneGenerator:
    """Server-ready Runway ML scene generator with character references"""

    def __init__(self):
        # Initialize Runway ML client
        try:
            self.client = RunwayML()
            print("✅ Runway ML client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Runway ML client: {e}")
            sys.exit(1)

        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None
        self.current_historical_period = None

        # Generation tracking
        self.generation_log = []
        self.api_calls_made = 0
        self.successful_downloads = 0

        # Character reference manager
        self.character_manager = None

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseSceneManager(str(db_path))

        print("🚀 Server Runway Scene Generator v1.0 Initialized")

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
            test_response = self.client.text_to_image.create(
                model="gen4_image",
                prompt_text="Simple test: red apple on white background",
                ratio="1920:1080"
            )

            if hasattr(test_response, 'id'):
                self.log_step("✅ Runway SDK Connection OK", "SUCCESS", {"test_task_id": test_response.id})
                return True
            else:
                self.log_step("❌ SDK Response Error", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ SDK Connection Test Failed: {e}", "ERROR")
            return False

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

        # Detect historical period
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT historical_period FROM topics WHERE id = ?', (topic_id,))
            period_result = cursor.fetchone()
            conn.close()

            if period_result and period_result[0]:
                self.current_historical_period = period_result[0]
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

    def setup_scene_directories(self):
        """Create necessary directories for scene generation"""
        output_dir = Path(self.current_output_dir)
        self.scenes_dir = output_dir / "scenes"
        self.thumbnail_dir = output_dir / "thumbnail"

        self.scenes_dir.mkdir(exist_ok=True)
        self.thumbnail_dir.mkdir(exist_ok=True)

        self.log_step("📁 Scene generation directories created", "SUCCESS")

    def load_character_references(self) -> bool:
        """Load character references using CharacterReferenceManager"""
        self.log_step("🎭 Loading character references")

        self.character_manager = CharacterReferenceManager(self.current_output_dir)

        if not self.character_manager.character_references:
            self.log_step("❌ No character references found", "ERROR")
            return False

        summary = self.character_manager.get_summary()
        self.log_step("✅ Character references loaded", "SUCCESS", summary)

        print(f"📊 Character Reference Summary:")
        print(f"   👥 Characters: {summary['total_characters']}")
        print(f"   📐 Variants: {summary['total_variants']}")
        print(f"   🎯 Primary: {summary['primary_character']}")
        print(f"   📋 List: {', '.join(summary['characters'])}")

        return True

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
            visual_prompts = prompts_data.get("visual_prompts", [])
        elif isinstance(prompts_data, list):
            visual_prompts = prompts_data
        else:
            raise ValueError("Invalid visual prompts format")

        # Filter out scene 99 (thumbnail) for now
        regular_scenes = [s for s in visual_prompts if s.get("scene_number", 0) != 99]

        self.log_step("✅ Visual prompts loaded", "SUCCESS", {
            "total_scenes": len(visual_prompts),
            "regular_scenes": len(regular_scenes)
        })

        return regular_scenes

    def build_runway_scene_prompt(self, scene: Dict) -> Tuple[str, List[Dict]]:
        """Build Runway ML scene prompt with character references"""

        scene_num = scene.get("scene_number", 0)
        base_prompt = scene.get("enhanced_prompt", scene.get("prompt", ""))
        characters_present = scene.get("characters_present", [])

        print(f"🎬 Building Runway scene prompt for Scene {scene_num}")

        # Get character references for this scene
        character_references = self.character_manager.get_scene_character_references(
            characters_present,
            base_prompt  # Pass scene content for analysis
        )

        # Build prompt with character reference syntax
        prompt_parts = []

        # Add character reference syntax to prompt
        if character_references:
            char_mentions = []
            for ref in character_references:
                char_mentions.append(f"@{ref['tag']}")

            # Inject character references into prompt
            if char_mentions:
                char_text = f"featuring {', '.join(char_mentions)} with consistent character appearance"
                prompt_parts.append(char_text)

        # Add historical context
        historical_context = f"79 CE Pompeii, {self.current_historical_period}"
        prompt_parts.append(historical_context)

        # Add main scene content (cleaned)
        clean_scene_content = self.clean_scene_prompt(base_prompt)
        prompt_parts.append(clean_scene_content)

        # Add technical specifications
        prompt_parts.append("cinematic composition, dramatic lighting, historical accuracy")
        prompt_parts.append("professional cinematography, realistic, detailed, 8K quality")

        # Combine all parts
        final_prompt = ", ".join(prompt_parts)

        print(f"   📝 Scene {scene_num}: {len(character_references)} character references")
        print(f"   📏 Prompt length: {len(final_prompt)} characters")
        print(f"   🔍 Preview: {final_prompt[:120]}...")

        return final_prompt, character_references

    def clean_scene_prompt(self, prompt: str) -> str:
        """Clean scene prompt for Runway ML (remove problematic content)"""

        # Basic cleaning for Runway ML
        import re

        # Remove Midjourney-specific parameters
        prompt = re.sub(r'--\w+\s+[\w:.]+', '', prompt)

        # Replace potentially problematic terms
        replacements = {
            "intimate": "private",
            "romantic": "tender",
            "sensual": "graceful",
            "nude": "classical",
            "naked": "unclothed",
            "bath": "washing area",
            "bedroom": "chamber",
            "violence": "conflict",
            "blood": "red stains"
        }

        for old, new in replacements.items():
            prompt = prompt.replace(old, new)

        # Clean up spacing
        prompt = re.sub(r'\s+', ' ', prompt).strip()

        return prompt

    def generate_scene_with_runway(self, scene: Dict) -> Optional[str]:
        """Generate single scene using Runway ML SDK with character references"""

        scene_num = scene.get("scene_number", 0)

        try:
            self.api_calls_made += 1

            # Build prompt with character references
            prompt, character_references = self.build_runway_scene_prompt(scene)

            # Create generation task
            if character_references:
                # Generate with character references
                print(f"   🎭 Scene {scene_num}: Generating with {len(character_references)} character references")

                task_response = self.client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=prompt,
                    ratio=CONFIG.runway_config["scene_ratio"],
                    reference_images=character_references
                )
            else:
                # Generate without references
                print(f"   🎬 Scene {scene_num}: Generating without character references")

                task_response = self.client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=prompt,
                    ratio=CONFIG.runway_config["scene_ratio"]
                )

            if hasattr(task_response, 'id'):
                print(f"   ✅ Scene {scene_num}: Task submitted - {task_response.id}")

                # Wait for completion
                print(f"   ⏳ Scene {scene_num}: Waiting for completion...")

                # Manual polling
                max_wait_cycles = CONFIG.runway_config["max_wait_cycles"]
                wait_interval = CONFIG.runway_config["wait_interval_seconds"]

                for cycle in range(max_wait_cycles):
                    try:
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

                                    print(f"   ✅ Scene {scene_num}: Completed after {cycle * wait_interval}s!")

                                    # Download and save
                                    saved_path = self.download_and_save_scene(
                                        image_url, scene_num, prompt, character_references
                                    )

                                    if saved_path:
                                        return saved_path
                                    else:
                                        return None
                                else:
                                    print(f"   ❌ Scene {scene_num}: Completed but no output")
                                    return None

                            elif status == "FAILED":
                                error_msg = getattr(task_status, 'error', 'Unknown error')
                                print(f"   ❌ Scene {scene_num}: Failed - {error_msg}")
                                return None

                            elif status in ["PENDING", "RUNNING", "THROTTLED"]:
                                if cycle % 4 == 0:  # Progress update every minute
                                    print(f"   ⏳ Scene {scene_num}: Still processing... ({cycle * wait_interval}s)")
                                time.sleep(wait_interval)
                                continue

                            else:
                                print(f"   ⚠️ Scene {scene_num}: Unknown status - {status}")
                                time.sleep(wait_interval)
                                continue
                        else:
                            print(f"   ⚠️ Scene {scene_num}: No status in response")
                            time.sleep(wait_interval)
                            continue

                    except Exception as poll_error:
                        print(f"   ⚠️ Scene {scene_num}: Poll error - {poll_error}")
                        time.sleep(wait_interval)
                        continue

                print(f"   ⏰ Scene {scene_num}: Timed out after {max_wait_cycles * wait_interval}s")
                return None
            else:
                print(f"   ❌ Scene {scene_num}: No task ID received")
                return None

        except Exception as e:
            print(f"   ❌ Scene {scene_num}: Generation failed - {e}")
            return None

    def download_and_save_scene(self, image_url: str, scene_num: int, prompt: str, character_references: List[Dict]) -> Optional[str]:
        """Download and save scene image with metadata"""

        save_path = self.scenes_dir / f"scene_{scene_num:02d}.png"
        json_path = self.scenes_dir / f"scene_{scene_num:02d}.json"

        try:
            # Download image
            response = requests.get(image_url, timeout=60, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(save_path)
            self.successful_downloads += 1

            # Save metadata
            metadata = {
                "scene_number": scene_num,
                "prompt": prompt,
                "image_url": image_url,
                "local_path": str(save_path),
                "generated_at": datetime.now().isoformat(),
                "character_references": character_references,
                "runway_powered": True,
                "runway_model": "gen4_image",
                "file_size": file_size,
                "ratio": CONFIG.runway_config["scene_ratio"]
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"   💾 Scene {scene_num}: Saved ({file_size:,} bytes)")
            return str(save_path)

        except Exception as e:
            print(f"   ❌ Scene {scene_num}: Download failed - {e}")
            return None

    def get_missing_scenes(self, visual_prompts: List[Dict]) -> List[Dict]:
        """Get list of scenes that need to be generated"""
        missing_scenes = []

        for scene in visual_prompts:
            scene_num = scene["scene_number"]
            scene_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

            if not scene_path.exists() or scene_path.stat().st_size < 1024:
                missing_scenes.append(scene)

        return missing_scenes

    def generate_all_scenes_runway(self, visual_prompts: List[Dict]) -> bool:
        """Generate all scenes using Runway ML with character references"""
        self.log_step("🎬 Starting Runway Scene Generation with Character References")

        # Check which scenes are missing
        missing_scenes = self.get_missing_scenes(visual_prompts)
        total_scenes = len(visual_prompts)
        existing_scenes = total_scenes - len(missing_scenes)

        print(f"📊 Scene Generation Status:")
        print(f"   ✅ Existing scenes: {existing_scenes}")
        print(f"   🎬 Missing scenes: {len(missing_scenes)}")
        print(f"   📋 Total scenes: {total_scenes}")

        if not missing_scenes:
            print("✅ All scenes already exist!")
            return True

        # Show existing scenes
        if existing_scenes > 0:
            existing_nums = []
            for scene in visual_prompts:
                scene_num = scene["scene_number"]
                if (self.scenes_dir / f"scene_{scene_num:02d}.png").exists():
                    existing_nums.append(scene_num)
            print(f"⏭️  Skipping existing scenes: {sorted(existing_nums)}")

        # Generate missing scenes
        successful_scenes = 0
        failed_scenes = 0

        for i, scene in enumerate(missing_scenes):
            scene_num = scene["scene_number"]

            print(f"\n🎬 Generating Scene {scene_num} ({i+1}/{len(missing_scenes)})")

            saved_path = self.generate_scene_with_runway(scene)

            if saved_path:
                successful_scenes += 1
                print(f"✅ Scene {scene_num}: Generation successful")
            else:
                failed_scenes += 1
                print(f"❌ Scene {scene_num}: Generation failed")

            # Rate limiting between scenes
            if i < len(missing_scenes) - 1:
                print("   ⏳ Waiting 10 seconds before next scene...")
                time.sleep(10)

        # Final assessment
        total_completed = existing_scenes + successful_scenes
        success_rate = total_completed / total_scenes

        print(f"\n📊 FINAL RUNWAY SCENE GENERATION SUMMARY:")
        print(f"   📋 Total scenes: {total_scenes}")
        print(f"   ✅ Completed scenes: {total_completed}")
        print(f"   🆕 Newly generated: {successful_scenes}")
        print(f"   📦 Pre-existing: {existing_scenes}")
        print(f"   ❌ Failed: {failed_scenes}")
        print(f"   📈 Success rate: {success_rate:.1%}")
        print(f"   🚀 Powered by: Runway ML Gen4 Image")

        is_successful = success_rate >= 0.8

        self.log_step(
            f"✅ Runway Scene Generation {'SUCCESS' if is_successful else 'PARTIAL'}: "
            f"{total_completed}/{total_scenes} scenes completed",
            "SUCCESS" if is_successful else "ERROR"
        )

        return is_successful

    def generate_thumbnail_runway(self) -> bool:
        """Generate thumbnail using Runway ML with primary character reference"""
        self.log_step("🖼️ Generating thumbnail with Runway ML")

        # Check if thumbnail already exists
        thumbnail_path = self.thumbnail_dir / "thumbnail.png"
        if thumbnail_path.exists() and thumbnail_path.stat().st_size > 1024:
            print("⏭️  Thumbnail already exists, skipping")
            return True

        try:
            # Get primary character reference
            primary_char = self.character_manager.primary_character
            if primary_char:
                primary_ref = self.character_manager.get_character_reference(primary_char)
                character_references = [{
                    "tag": "main_character",
                    "uri": primary_ref
                }] if primary_ref else []
            else:
                character_references = []

            # Build thumbnail prompt
            topic_title = self.current_topic
            historical_period = self.current_historical_period

            if character_references:
                thumbnail_prompt = f"""Epic movie poster style image featuring @main_character, {topic_title}, 
                {historical_period}, dramatic cinematic composition, 
                movie poster lighting, heroic pose, detailed background, 
                professional movie poster design, captivating, 8K quality"""
            else:
                thumbnail_prompt = f"""Epic movie poster style image, {topic_title}, 
                {historical_period}, dramatic cinematic composition, 
                movie poster lighting, detailed background, 
                professional movie poster design, captivating, 8K quality"""

            print(f"🎨 Thumbnail prompt: {thumbnail_prompt[:100]}...")

            # Generate thumbnail
            self.api_calls_made += 1

            if character_references:
                task_response = self.client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=thumbnail_prompt,
                    ratio=CONFIG.runway_config["scene_ratio"],
                    reference_images=character_references
                )
            else:
                task_response = self.client.text_to_image.create(
                    model="gen4_image",
                    prompt_text=thumbnail_prompt,
                    ratio=CONFIG.runway_config["scene_ratio"]
                )

            if hasattr(task_response, 'id'):
                print(f"✅ Thumbnail task submitted: {task_response.id}")

                # Wait for completion
                for cycle in range(60):  # 15 minutes max
                    try:
                        task_status = self.client.tasks.retrieve(task_response.id)

                        if hasattr(task_status, 'status'):
                            status = task_status.status.upper()

                            if status == "SUCCEEDED":
                                if hasattr(task_status, 'output') and task_status.output:
                                    if isinstance(task_status.output, list):
                                        image_url = task_status.output[0]
                                    else:
                                        image_url = str(task_status.output)

                                    # Download thumbnail
                                    response = requests.get(image_url, timeout=60)
                                    response.raise_for_status()

                                    with open(thumbnail_path, 'wb') as f:
                                        f.write(response.content)

                                    # Save metadata
                                    metadata = {
                                        "type": "thumbnail",
                                        "prompt": thumbnail_prompt,
                                        "image_url": image_url,
                                        "local_path": str(thumbnail_path),
                                        "generated_at": datetime.now().isoformat(),
                                        "character_reference": primary_char,
                                        "runway_powered": True
                                    }

                                    json_path = self.thumbnail_dir / "thumbnail.json"
                                    with open(json_path, 'w', encoding='utf-8') as f:
                                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                                    print("✅ Thumbnail generated and saved successfully")
                                    return True

                            elif status == "FAILED":
                                print(f"❌ Thumbnail generation failed")
                                return False

                            elif status in ["PENDING", "RUNNING", "THROTTLED"]:
                                if cycle % 4 == 0:
                                    print(f"⏳ Thumbnail still generating... ({cycle * 15}s)")
                                time.sleep(15)
                                continue

                    except Exception as e:
                        print(f"⚠️ Thumbnail polling error: {e}")
                        time.sleep(15)
                        continue

                print("⏰ Thumbnail generation timed out")
                return False
            else:
                print("❌ Thumbnail task submission failed")
                return False

        except Exception as e:
            print(f"❌ Thumbnail generation error: {e}")
            return False

    def save_scene_generation_report(self):
        """Save scene generation report"""
        output_dir = Path(self.current_output_dir)

        report = {
            "scene_generation_completed": datetime.now().isoformat(),
            "topic_id": self.current_topic_id,
            "topic": self.current_topic,
            "api_calls_made": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "character_references_used": len(self.character_manager.character_references) if self.character_manager else 0,
            "scenes_dir": str(self.scenes_dir),
            "thumbnail_dir": str(self.thumbnail_dir),
            "historical_period": self.current_historical_period,
            "generation_log": self.generation_log,
            "runway_sdk_version": True,
            "runway_model": "gen4_image",
            "character_reference_system": True,
            "server_optimized": True
        }

        report_path = output_dir / "scene_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Scene generation report saved: {report_path}", "SUCCESS")

    def run_runway_scene_generation(self) -> bool:
        """Run complete scene generation process with Runway ML"""
        print("🚀" * 50)
        print("RUNWAY ML SCENE GENERATOR v1.0")
        print("🎭 CHARACTER REFERENCE SYSTEM")
        print("🔗 Database integrated")
        print("🎬 SCENE GENERATION")
        print("🖼️ THUMBNAIL GENERATION")
        print("🚀 Powered by Runway ML Gen4 Image")
        print("🚀" * 50)

        # Test connection
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
            # Setup directories
            self.setup_scene_directories()

            # Load character references
            if not self.load_character_references():
                self.log_step("❌ Character references required for scene generation", "ERROR")
                return False

            # Load visual prompts
            visual_prompts = self.load_visual_prompts()
            print(f"🎬 Scene prompts loaded: {len(visual_prompts)}")

            # Generate scenes
            scenes_success = self.generate_all_scenes_runway(visual_prompts)

            # Generate thumbnail
            thumbnail_success = self.generate_thumbnail_runway()

            # Save report
            self.save_scene_generation_report()

            # Update database
            scenes_count = len([f for f in self.scenes_dir.glob("scene_*.png")]) if scenes_success else 0

            self.db_manager.mark_scene_generation_completed(
                self.current_topic_id, scenes_count, thumbnail_success
            )

            # Final status
            if scenes_success and thumbnail_success:
                print("\n" + "🎉" * 50)
                print("RUNWAY SCENE GENERATION COMPLETELY SUCCESSFUL!")
                print("✅ All scenes generated + Thumbnail successful")
                print(f"🎭 Character references: {len(self.character_manager.character_references)}")
                print(f"✅ API Calls: {self.api_calls_made}")
                print(f"✅ Downloads: {self.successful_downloads}")
                print(f"📁 Saved to: {self.scenes_dir}")
                print(f"🚀 Powered by: Runway ML Gen4 Image")
                print("🎉" * 50)
                overall_success = True
            elif scenes_success:
                print("\n" + "🎊" * 50)
                print("RUNWAY SCENE GENERATION SUCCESSFUL!")
                print("✅ Scenes generated successfully")
                print("❌ Thumbnail failed")
                print("🎊" * 50)
                overall_success = True
            else:
                print("\n" + "❌" * 50)
                print("RUNWAY SCENE GENERATION FAILED!")
                print("❌ Scene generation failed")
                print("❌" * 50)
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ Scene generation failed: {e}", "ERROR")
            self.db_manager.mark_scene_generation_failed(self.current_topic_id, str(e))
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    try:
        print("🚀 RUNWAY ML SCENE GENERATOR v1.0")
        print("🎭 CHARACTER REFERENCE SYSTEM")
        print("🔗 Database integration")
        print("🎬 SCENE GENERATION")
        print("🖼️ THUMBNAIL GENERATION")
        print("🚀 Powered by Runway ML Gen4 Image SDK")
        print("=" * 60)

        generator = ServerRunwaySceneGenerator()
        success = generator.run_runway_scene_generation()

        if success:
            print("🎊 Runway scene generation completed successfully!")
        else:
            print("⚠️ Scene generation failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ Scene generation stopped by user")
    except Exception as e:
        print(f"💥 Scene generation failed: {e}")
        CONFIG.logger.error(f"Scene generation failed: {e}")
        import traceback
        traceback.print_exc()