"""
Sleepy Dull Stories - ENHANCED Midjourney Scene Generator with Claude AI Prompt Correction
REAL-TIME prompt correction using Claude Sonnet 4 when Midjourney rejects prompts
Production-optimized with intelligent error recovery and JSON updating
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

# ADD CLAUDE API INTEGRATION
class ClaudePromptCorrector:
    """Claude Sonnet 4 powered prompt correction for Midjourney content policy"""

    def __init__(self):
        self.setup_claude_config()

        if not self.api_key:
            print("⚠️ CLAUDE_API_KEY not found - Claude prompt correction disabled")
            self.enabled = False
        else:
            print("🧠 Claude Sonnet 4 prompt corrector enabled")
            self.enabled = True

        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Track correction attempts per scene
        self.correction_attempts = {}
        self.max_attempts = 4

    def setup_claude_config(self):
        """Setup Claude configuration with PROVEN SETTINGS from successful version"""
        self.claude_config = {
            "model": "claude-sonnet-4-20250514",  # ✅ CLAUDE 4 - PROVEN SUCCESSFUL
            "max_tokens": 64000,  # ✅ HIGH TOKEN LIMIT - PROVEN SUCCESSFUL
            "temperature": 0.7,
            "streaming_response": True,  # ✅ PROVEN CRITICAL
            "long_timeout": True  # ✅ PROVEN CRITICAL
        }

        # Get API key
        self.api_key = self.get_claude_api_key()

    def get_claude_api_key(self):
        """Get Claude API key from multiple sources"""
        # Try different environment variable names
        api_key = (
                os.getenv('CLAUDE_API_KEY') or
                os.getenv('ANTHROPIC_API_KEY') or
                os.getenv('CLAUDE_4_API_KEY') or
                os.getenv('CLAUDE_SONNET_API_KEY')
        )

        if not api_key:
            # Check .env file
            env_files = [
                Path('.env'),
                Path('../../.env'),
                Path('../../.env')
            ]

            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file)
                    api_key = (
                        os.getenv('CLAUDE_API_KEY') or
                        os.getenv('ANTHROPIC_API_KEY') or
                        os.getenv('CLAUDE_4_API_KEY') or
                        os.getenv('CLAUDE_SONNET_API_KEY')
                    )
                    if api_key:
                        print(f"✅ Claude API key loaded from: {env_file}")
                        break

        if not api_key:
            raise ValueError(
                "❌ Claude API key required!\n"
                "Set in .env file:\n"
                "CLAUDE_API_KEY=sk-ant-api03-xxxxx\n"
                "Or environment variable: CLAUDE_API_KEY"
            )

        print("✅ Claude API key loaded successfully")
        return api_key

    def correct_prompt_with_claude(self, scene_data: Dict, banned_word: str, attempt_number: int) -> Optional[str]:
        """Use Claude Sonnet 4 to correct a banned prompt"""

        if not self.enabled:
            return None

        scene_num = scene_data.get("scene_number", 0)
        original_prompt = scene_data.get("enhanced_prompt", scene_data.get("prompt", ""))

        # Different severity levels based on attempt
        if attempt_number == 1:
            severity = "carefully review and make minimal changes to avoid the banned word"
            tone = "Keep the essence and visual details intact"
        elif attempt_number == 2:
            severity = "more aggressively rewrite problem areas and similar risky words"
            tone = "Be more conservative with word choices"
        elif attempt_number == 3:
            severity = "completely rewrite the problematic sections with safe alternatives"
            tone = "Prioritize Midjourney safety over original wording"
        else:
            severity = "completely restructure the entire prompt to be maximally safe"
            tone = "Create entirely new wording that achieves the same visual goal"

        system_message = f"""You are a Midjourney prompt expert specializing in content policy compliance. Midjourney has rejected a prompt for containing the banned word: "{banned_word}"

Your task: {severity}. {tone}.

CRITICAL RULES:
1. Keep the same visual scene and cinematic style
2. Maintain character references [CHARACTERNAME] if present
3. Keep technical parameters (--ar 16:9, --v 7.0, etc)
4. Remove or replace ANY potentially problematic words
5. Add safety phrases like "historical educational content, appropriate content"
6. Keep the prompt under 4000 characters
7. This is attempt #{attempt_number}/4 - {'be more aggressive' if attempt_number > 2 else 'be careful but thorough'}

COMMON MIDJOURNEY BANNED WORDS TO AVOID:
intimate, romantic, bath, bathing, bedroom, bed, nude, naked, bare, undressed, children, child, kids, embrace, embracing, kiss, kissing, violence, blood, fight, sensual, seductive

Return ONLY the corrected prompt, nothing else."""

        user_message = f"""SCENE #{scene_num} - ATTEMPT #{attempt_number}

BANNED WORD DETECTED: "{banned_word}"

ORIGINAL PROMPT:
{original_prompt}

SCENE CONTEXT:
- Title: {scene_data.get('title', 'Unknown')}
- Location: {scene_data.get('location', 'Unknown')}
- Characters: {scene_data.get('characters_present', [])}
- Emotion: {scene_data.get('emotion', 'neutral')}

Please provide the corrected prompt that will pass Midjourney's content policy."""

        payload = {
            "model": self.claude_config["model"],
            "max_tokens": 4000,
            "temperature": self.claude_config["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": f"{system_message}\n\n{user_message}"
                }
            ]
        }

        try:
            print(f"🧠 Claude Sonnet 4: Correcting Scene {scene_num} prompt (attempt {attempt_number}/4)")
            print(f"   Banned word: '{banned_word}'")

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                corrected_prompt = result["content"][0]["text"].strip()

                print(f"✅ Claude Sonnet 4: Scene {scene_num} prompt corrected")
                print(f"   Original length: {len(original_prompt)}")
                print(f"   Corrected length: {len(corrected_prompt)}")
                print(f"   Preview: {corrected_prompt[:100]}...")

                return corrected_prompt
            else:
                print(f"❌ Claude API error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Claude correction failed: {e}")
            return None

    def update_visual_prompts_json(self, visual_prompts_path: str, scene_number: int, corrected_prompt: str) -> bool:
        """Update the visual_generation_prompts.json file with corrected prompt"""

        try:
            # Read current JSON
            with open(visual_prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)

            # Find and update the scene
            updated = False
            for scene in prompts_data:
                if scene.get("scene_number") == scene_number:
                    scene["enhanced_prompt"] = corrected_prompt
                    scene["claude_corrected"] = True
                    scene["original_prompt_backup"] = scene.get("prompt", "")
                    updated = True
                    break

            if updated:
                # Save updated JSON
                with open(visual_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(prompts_data, f, indent=2, ensure_ascii=False)

                print(f"✅ Updated visual_generation_prompts.json for Scene {scene_number}")
                return True
            else:
                print(f"❌ Scene {scene_number} not found in JSON")
                return False

        except Exception as e:
            print(f"❌ Failed to update JSON: {e}")
            return False

# ADD INDEPENDENT THUMBNAIL IMPORT
try:
    from independent_thumbnail_generator import IndependentThumbnailGenerator
    INDEPENDENT_THUMBNAIL_AVAILABLE = True
    print("✅ Independent thumbnail generator imported")
except ImportError:
    INDEPENDENT_THUMBNAIL_AVAILABLE = False
    print("⚠️ Independent thumbnail generator not found")

# ADD INTELLIGENT RETRY SYSTEM IMPORT
try:
    from intelligent_scene_retry import IntelligentSceneRetrySystem, EnhancedSceneGeneratorWithIntelligentRetry
    INTELLIGENT_RETRY_AVAILABLE = True
    print("✅ Intelligent scene retry system imported")
except ImportError:
    INTELLIGENT_RETRY_AVAILABLE = False
    print("⚠️ Intelligent scene retry system not found")

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
            "default_version": "7.0",
            "process_mode": "relax",
            "character_generation": False,  # Disabled - use existing
            "scene_generation": True,       # Main focus
            "thumbnail_generation": True,   # Main focus
            "server_mode": True,
            "production_ready": True
        }
        self.prompt_template = {
            "default_core": "cinematic realistic photograph professional film photography dramatic lighting photorealistic historical scene detailed textures",
            "style_modifiers": "warm golden light deep shadows atmospheric weathered materials classical proportions",
            "banned_words": [
                "intimate", "romantic", "sensual", "seductive",
                "nude", "naked", "bare", "undressed",
                "kiss", "kissing", "embrace", "embracing",
                "children", "child", "kids", "minor",
                "violence", "blood", "fight", "weapon",
                "bedroom", "bed", "bath", "bathing"
            ]
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
    """Enhanced server-ready Midjourney scene generator with Claude AI prompt correction"""

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

        # Track failed scenes to prevent infinite loops
        self.scene_attempt_count = {}
        self.blacklisted_scenes = set()

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseSceneManager(str(db_path))

        # Initialize Claude prompt corrector
        self.claude_corrector = ClaudePromptCorrector()

        # Initialize intelligent retry system if available
        if INTELLIGENT_RETRY_AVAILABLE:
            self.intelligent_retry_system = IntelligentSceneRetrySystem()
            self.intelligent_retry_enabled = self.intelligent_retry_system.claude_api_key is not None
            print("🧠 Intelligent retry system enabled")
        else:
            self.intelligent_retry_system = None
            self.intelligent_retry_enabled = False
            print("⚠️ Intelligent retry system disabled")

        print("🚀 Enhanced Midjourney Scene Generator v2.0 with Claude AI Initialized")
        print(f"🔑 Midjourney API Key: {self.api_key[:8]}...")
        print(f"🧠 Claude AI Correction: {'✅ Enabled' if self.claude_corrector.enabled else '❌ Disabled'}")
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

    def extract_banned_word_from_error(self, error_response: Dict) -> Optional[str]:
        """Extract banned word from Midjourney error response"""
        try:
            error_data = error_response.get("error", {})
            raw_message = error_data.get("raw_message", "")

            # Format: "Banned Prompt: word"
            if "Banned Prompt:" in raw_message:
                banned_word = raw_message.split("Banned Prompt:")[-1].strip()
                return banned_word

            return None
        except:
            return None

    def submit_midjourney_task_with_claude_correction(self, prompt: str, scene_data: Dict, aspect_ratio: str = "16:9") -> Optional[str]:
        """Submit task with Claude AI correction on banned prompts"""

        scene_num = scene_data.get("scene_number", 0)
        max_claude_attempts = self.claude_corrector.max_attempts

        # Track correction attempts for this scene
        if scene_num not in self.claude_corrector.correction_attempts:
            self.claude_corrector.correction_attempts[scene_num] = 0

        current_prompt = prompt

        for attempt in range(max_claude_attempts + 1):  # +1 for original attempt

            if attempt == 0:
                print(f"🎬 Scene {scene_num}: Submitting original prompt")
            else:
                print(f"🔄 Scene {scene_num}: Claude correction attempt {attempt}/{max_claude_attempts}")

            # Clean prompt for PiAPI
            cleaned_prompt = self.clean_prompt_for_piapi(current_prompt)

            payload = {
                "model": "midjourney",
                "task_type": "imagine",
                "input": {
                    "prompt": cleaned_prompt,
                    "aspect_ratio": aspect_ratio,
                    "process_mode": "relax"
                }
            }

            url = f"{self.base_url}/task"

            try:
                self.api_calls_made += 1
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 200:
                        task_data = result.get("data", {})
                        task_id = task_data.get("task_id")

                        if attempt > 0:
                            print(f"✅ Scene {scene_num}: Submitted after Claude correction #{attempt}")

                        return task_id
                    else:
                        # Check for banned prompt error
                        error_message = result.get('message', '')

                        if "failed to check prompt" in error_message or result.get("error", {}).get("code") == 10000:
                            # Extract banned word
                            banned_word = self.extract_banned_word_from_error(result)

                            if banned_word and self.claude_corrector.enabled and attempt < max_claude_attempts:
                                print(f"🛡️ Scene {scene_num}: Banned word detected: '{banned_word}'")

                                # Use Claude to correct prompt
                                corrected_prompt = self.claude_corrector.correct_prompt_with_claude(
                                    scene_data, banned_word, attempt + 1
                                )

                                if corrected_prompt:
                                    current_prompt = corrected_prompt
                                    self.claude_corrector.correction_attempts[scene_num] += 1

                                    # Update JSON file with corrected prompt
                                    visual_prompts_path = Path(self.current_output_dir) / "visual_generation_prompts.json"
                                    if visual_prompts_path.exists():
                                        self.claude_corrector.update_visual_prompts_json(
                                            str(visual_prompts_path), scene_num, corrected_prompt
                                        )

                                    print(f"🧠 Scene {scene_num}: Trying Claude-corrected prompt")
                                    time.sleep(2)  # Brief pause before retry
                                    continue
                                else:
                                    print(f"❌ Scene {scene_num}: Claude correction failed")
                                    return None
                            else:
                                print(f"❌ Scene {scene_num}: Banned prompt - no more correction attempts")
                                return None
                        else:
                            print(f"❌ Scene {scene_num}: API Error: {result.get('message', 'Unknown error')}")
                            return None

                elif response.status_code == 500:
                    print(f"❌ Scene {scene_num}: HTTP 500 - Server error")
                    return None
                else:
                    print(f"❌ Scene {scene_num}: HTTP Error: {response.status_code}")
                    return None

            except Exception as e:
                print(f"❌ Scene {scene_num}: Request failed: {e}")
                return None

        print(f"❌ Scene {scene_num}: Failed after {max_claude_attempts} Claude correction attempts")
        return None

    def clean_prompt_for_piapi(self, prompt: str) -> str:
        """Remove all characters that might confuse PiAPI's prompt parser"""
        import re

        # Remove all --ar parameters
        prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt)

        # Remove all --v parameters
        prompt = re.sub(r'--v\s+[\d.]+', '', prompt)

        # Remove any other -- parameters
        prompt = re.sub(r'--\w+(?:\s+[\w:.]+)?', '', prompt)

        # Remove problematic dash characters
        prompt = prompt.replace(' - ', ' ')
        prompt = prompt.replace('-', ' ')

        # Clean up spaces
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip()

        return prompt

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

            # Violence/conflict
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
                filtered_prompt += ", appropriate content"

        return filtered_prompt

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

        for filename in characters_dir.glob("*.json"):
            if filename.stem == "thumbnail":
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

        # Filter out scene 99 (thumbnail)
        regular_scenes = [s for s in visual_prompts if s.get("scene_number", 0) != 99]

        self.log_step("✅ Visual prompts loaded", "SUCCESS", {
            "total_scenes": len(visual_prompts),
            "regular_scenes": len(regular_scenes),
            "thumbnail_scenes_filtered": len([s for s in visual_prompts if s.get("scene_number", 0) == 99])
        })

        return regular_scenes

    def setup_scene_directories(self):
        """Create necessary directories for scene generation"""
        output_dir = Path(self.current_output_dir)

        self.scenes_dir = output_dir / "scenes"
        self.thumbnail_dir = output_dir / "thumbnail"

        self.scenes_dir.mkdir(exist_ok=True)
        self.thumbnail_dir.mkdir(exist_ok=True)

        self.log_step("📁 Scene generation directories created", "SUCCESS")

    def build_safe_scene_prompt(self, scene: Dict) -> str:
        """Build safe scene prompt using clean template system"""

        base_prompt = scene.get("enhanced_prompt", scene.get("prompt", ""))
        scene_num = scene.get("scene_number")

        print(f"🎬 Building prompt for Scene {scene_num}")

        # Character references
        char_refs = []
        if scene.get("characters_present") and len(self.character_references) > 0:
            for char_name in scene["characters_present"]:
                if char_name in self.character_references:
                    char_refs.append(self.character_references[char_name])

        # Scene-specific (keep it simple - 20 words max)
        scene_words = base_prompt.split()
        if len(scene_words) > 20:
            scene_specific = " ".join(scene_words[:20])
        else:
            scene_specific = base_prompt

        # Build raw prompt first
        raw_prompt_parts = []

        if char_refs:
            raw_prompt_parts.extend(char_refs)

        raw_prompt_parts.append(scene_specific)
        raw_prompt_parts.append("cinematic realistic photograph professional film photography dramatic lighting")
        raw_prompt_parts.append("warm golden light deep shadows atmospheric")
        raw_prompt_parts.append("--v 7.0 --ar 16:9")

        raw_prompt = " ".join(raw_prompt_parts)

        # Apply content policy filter
        filtered_prompt = self.apply_content_policy_filter(raw_prompt)

        print(f"🔧 Final scene prompt: {filtered_prompt[:150]}...")

        return filtered_prompt

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
                        temp_urls = output.get("temporary_image_urls", [])
                        image_url = output.get("image_url", "")

                        if temp_urls and len(temp_urls) > 0:
                            selected_url = temp_urls[1] if len(temp_urls) >= 2 else temp_urls[0]
                            return {"url": selected_url, "source": "temporary_image_urls"}
                        elif image_url:
                            return {"url": image_url, "source": "image_url"}
                        else:
                            print(f"⚠️ Scene {scene_num}: Completed but no image URLs found")
                            return False

                    elif status == "failed":
                        error_info = task_data.get("error", {})
                        error_msg = error_info.get("message", "Unknown error")
                        print(f"❌ Scene {scene_num}: Task failed - {error_msg}")
                        return False
                    else:
                        return None

            else:
                print(f"⚠️ Scene {scene_num}: Status check failed HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"⚠️ Scene {scene_num}: Status check exception - {e}")
            return None

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

            print(f"📥 Scene {scene_num}: Downloading from {image_url[:50]}...")

            response = requests.get(image_url, headers=headers, timeout=30, stream=True)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = os.path.getsize(save_path)
                print(f"✅ Scene {scene_num}: Downloaded successfully ({file_size} bytes)")
                self.successful_downloads += 1
                return True
            else:
                print(f"❌ Scene {scene_num}: Download failed HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Scene {scene_num}: Download exception - {e}")
            return False

    def get_missing_scenes(self, visual_prompts: List[Dict]) -> List[Dict]:
        """Get list of scenes that are missing and not blacklisted"""
        regular_scenes = [s for s in visual_prompts if s["scene_number"] != 99]
        missing_scenes = []

        for scene in regular_scenes:
            scene_num = scene["scene_number"]

            if scene_num in self.blacklisted_scenes:
                continue

            image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

            if not image_path.exists():
                missing_scenes.append(scene)

        return missing_scenes

    def generate_scenes_with_claude_correction(self, visual_prompts: List[Dict], max_retry_rounds: int = 10):
        """Generate all scenes with Claude AI prompt correction"""

        print("🧠 ENHANCED SCENE GENERATION WITH CLAUDE AI PROMPT CORRECTION")
        print("🛡️ Real-time prompt correction when Midjourney blocks content")
        print("🔄 Up to 4 correction attempts per scene using Claude Sonnet 4")

        for retry_round in range(max_retry_rounds):
            missing_scenes = self.get_missing_scenes(visual_prompts)

            if not missing_scenes:
                print("✅ All scenes completed!")
                return True

            total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
            blacklisted_count = len(self.blacklisted_scenes)

            if retry_round == 0:
                print(f"🎬 Starting scene generation - {len(missing_scenes)} scenes to generate")
            else:
                print(f"\n🔄 RETRY ROUND {retry_round}: {len(missing_scenes)} missing scenes")
                if blacklisted_count > 0:
                    print(f"⚫ {blacklisted_count} scenes blacklisted")
                print("⏳ Waiting 60 seconds before retry round...")
                time.sleep(60)

            # Update attempt counts and blacklisting
            for scene in missing_scenes:
                scene_num = scene["scene_number"]
                self.scene_attempt_count[scene_num] = self.scene_attempt_count.get(scene_num, 0) + 1

                if self.scene_attempt_count[scene_num] > 8:  # Higher limit with Claude correction
                    self.blacklisted_scenes.add(scene_num)
                    print(f"⚫ Scene {scene_num}: Blacklisted after {self.scene_attempt_count[scene_num]} attempts")

            # Re-get missing scenes after blacklisting
            missing_scenes = self.get_missing_scenes(visual_prompts)

            if not missing_scenes:
                completed_count = total_scenes - blacklisted_count
                print(f"✅ All processable scenes completed! ({completed_count}/{total_scenes})")
                return True

            # Submit scene tasks with Claude correction
            scene_tasks = {}
            successful_submissions = 0

            for i, scene in enumerate(missing_scenes):
                scene_num = scene["scene_number"]
                attempt_num = self.scene_attempt_count.get(scene_num, 0)

                print(f"🎬 Processing Scene {scene_num} ({i + 1}/{len(missing_scenes)}) - Attempt #{attempt_num}")

                # Build safe scene prompt
                final_prompt = self.build_safe_scene_prompt(scene)

                # Submit task with Claude correction capability
                task_id = self.submit_midjourney_task_with_claude_correction(
                    final_prompt, scene, aspect_ratio="16:9"
                )

                if task_id:
                    scene_tasks[scene_num] = {
                        "task_id": task_id,
                        "prompt": final_prompt,
                        "scene_data": scene
                    }
                    successful_submissions += 1
                    print(f"✅ Scene {scene_num}: Submitted successfully")
                else:
                    print(f"❌ Scene {scene_num}: Submission failed after all correction attempts")

                # Rate limiting
                if i < len(missing_scenes) - 1:
                    wait_time = 8 + (retry_round * 2)
                    print(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)

            print(f"📊 Round {retry_round + 1} submissions: ✅ {successful_submissions} | ❌ {len(missing_scenes) - successful_submissions}")

            if not scene_tasks:
                print("❌ No tasks submitted in this round")
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

            # Download completed scenes
            successful_downloads = 0

            for scene_num, scene_data in completed_scenes.items():
                result_data = scene_data["result_data"]
                image_path = self.scenes_dir / f"scene_{scene_num:02d}.png"

                if self.download_image_detailed(result_data, str(image_path), scene_num):
                    successful_downloads += 1

                    # Enhanced metadata with Claude correction info
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
                        "claude_corrections": self.claude_corrector.correction_attempts.get(scene_num, 0),
                        "content_filtered": True
                    }

                    json_path = self.scenes_dir / f"scene_{scene_num:02d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"✅ Round {retry_round + 1} downloads: {successful_downloads}")

        # Final summary
        final_missing = self.get_missing_scenes(visual_prompts)
        total_scenes = len([s for s in visual_prompts if s["scene_number"] != 99])
        completed_count = total_scenes - len(final_missing) - len(self.blacklisted_scenes)

        print(f"\n📊 FINAL SUMMARY:")
        print(f"✅ Completed: {completed_count}")
        print(f"❌ Missing: {len(final_missing)}")
        print(f"⚫ Blacklisted: {len(self.blacklisted_scenes)}")
        print(f"🧠 Claude corrections used: {sum(self.claude_corrector.correction_attempts.values())}")

        success_rate = completed_count / total_scenes
        return success_rate >= 0.85

    def save_scene_generation_report(self):
        """Save enhanced scene generation report"""
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
            "claude_correction_enabled": self.claude_corrector.enabled,
            "claude_corrections_made": sum(self.claude_corrector.correction_attempts.values()),
            "claude_correction_attempts": dict(self.claude_corrector.correction_attempts),
            "server_optimized": True,
            "version": "2.0_claude_enhanced"
        }

        report_path = output_dir / "scene_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log_step(f"✅ Enhanced scene generation report saved: {report_path}", "SUCCESS")

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
                self.log_step("✅ API Connection OK", "SUCCESS", {"response": result})
                return True
            else:
                self.log_step(f"❌ API Error: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log_step(f"❌ Connection Test Failed: {e}", "ERROR")
            return False

    def run_enhanced_scene_generation(self) -> bool:
        """Run enhanced scene generation with Claude AI correction"""
        print("🚀" * 50)
        print("ENHANCED MIDJOURNEY SCENE GENERATOR v2.0")
        print("🧠 CLAUDE SONNET 4 PROMPT CORRECTION")
        print("🔗 Database integrated")
        print("🎬 SCENES GENERATION")
        print("🖼️ INDEPENDENT THUMBNAIL GENERATION")
        print("🛡️ REAL-TIME PROMPT CORRECTION")
        print("🚀" * 50)

        # Test API connection
        if not self.test_api_connection():
            self.log_step("❌ API connection failed - aborting", "ERROR")
            return False

        # Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🧠 Claude correction: {'✅ Enabled' if self.claude_corrector.enabled else '❌ Disabled'}")

        try:
            # Setup directories
            self.setup_scene_directories()

            # Load character references
            if not self.load_existing_character_references():
                self.log_step("❌ No character references found", "ERROR")
                return False

            print(f"🎭 Character references loaded: {len(self.character_references)}")

            # Load visual prompts
            visual_prompts = self.load_visual_prompts()
            print(f"🎬 Scene prompts loaded: {len(visual_prompts)}")

            # Generate scenes with Claude correction
            print("\n🎬 GENERATING SCENES WITH CLAUDE AI CORRECTION...")
            scenes_success = self.generate_scenes_with_claude_correction(visual_prompts, max_retry_rounds=12)

            # Generate independent thumbnail
            print("\n🖼️ GENERATING INDEPENDENT THUMBNAIL...")
            if INDEPENDENT_THUMBNAIL_AVAILABLE:
                try:
                    independent_thumbnail = IndependentThumbnailGenerator(
                        output_dir=str(self.current_output_dir),
                        api_key=self.api_key
                    )
                    thumbnail_success = independent_thumbnail.generate_thumbnail()
                except Exception as e:
                    print(f"❌ Independent thumbnail error: {e}")
                    thumbnail_success = False
            else:
                print("⚠️ Independent thumbnail generator not available")
                thumbnail_success = False

            # Save enhanced report
            self.save_scene_generation_report()

            # Update database
            scenes_count = len([f for f in self.scenes_dir.glob("scene_*.png")]) if scenes_success else 0

            self.db_manager.mark_scene_generation_completed(
                self.current_topic_id, scenes_count, thumbnail_success
            )

            # Final success assessment
            if scenes_success and thumbnail_success:
                print("\n" + "🎉" * 50)
                print("ENHANCED GENERATION COMPLETELY SUCCESSFUL!")
                print("✅ ALL scenes generated + Independent thumbnail successful")
                print(f"🧠 Claude corrections used: {sum(self.claude_corrector.correction_attempts.values())}")
                print("🛡️ REAL-TIME PROMPT CORRECTION WORKING")
                print("🎉" * 50)
                overall_success = True
            elif scenes_success:
                print("\n" + "🎊" * 50)
                print("ENHANCED SCENE GENERATION SUCCESSFUL!")
                print("✅ Scenes generated successfully")
                print(f"🧠 Claude corrections used: {sum(self.claude_corrector.correction_attempts.values())}")
                print("❌ Independent thumbnail failed")
                print("🎊" * 50)
                overall_success = True
            else:
                print("\n" + "❌" * 50)
                print("ENHANCED GENERATION FAILED!")
                print("❌ Scene generation failed despite Claude correction")
                print("❌" * 50)
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ Enhanced scene generation failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    try:
        print("🚀 ENHANCED MIDJOURNEY SCENE GENERATOR v2.0")
        print("🧠 CLAUDE SONNET 4 PROMPT CORRECTION")
        print("🔗 Database integration with character references")
        print("🎬 SCENES GENERATION")
        print("🖼️ INDEPENDENT THUMBNAIL GENERATION")
        print("🛡️ REAL-TIME PROMPT CORRECTION")
        print("=" * 60)

        generator = ServerMidjourneySceneGenerator()
        success = generator.run_enhanced_scene_generation()

        if success:
            print("🎊 Enhanced scene generation completed successfully!")
        else:
            print("⚠️ Enhanced scene generation failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ Enhanced scene generation stopped by user")
    except Exception as e:
        print(f"💥 Enhanced scene generation failed: {e}")
        CONFIG.logger.error(f"Enhanced scene generation failed: {e}")
        import traceback
        traceback.print_exc()