"""
Sleepy Dull Stories - Sosyal Medya Görsel Oluşturucu
YouTube Shorts, Instagram Reels ve TikTok videoları için Midjourney ile görsel oluşturma
Claude AI prompt düzeltme sistemi ile entegre
Database entegre otomatik workflow
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
import re

# Load environment first
load_dotenv()

class ClaudeSocialMediaPromptCorrector:
    """Claude Sonnet 4 powered prompt correction for social media content"""

    def __init__(self):
        self.setup_claude_config()

        if not self.api_key:
            print("⚠️ CLAUDE_API_KEY not found - Claude prompt correction disabled")
            self.enabled = False
        else:
            print("🧠 Claude Sonnet 4 social media prompt corrector enabled")
            self.enabled = True

        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Track correction attempts per content
        self.correction_attempts = {}
        self.max_attempts = 4

    def setup_claude_config(self):
        """Setup Claude configuration"""
        self.claude_config = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 64000,
            "temperature": 0.7,
            "streaming_response": True,
            "long_timeout": True
        }

        self.api_key = self.get_claude_api_key()

    def get_claude_api_key(self):
        """Get Claude API key from multiple sources"""
        api_key = (
                os.getenv('CLAUDE_API_KEY') or
                os.getenv('ANTHROPIC_API_KEY') or
                os.getenv('CLAUDE_4_API_KEY') or
                os.getenv('CLAUDE_SONNET_API_KEY')
        )

        if not api_key:
            env_files = [
                Path('.env'),
                Path('../../.env'),
                Path('../../../.env')
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
            print("⚠️ Claude API key not found - correction disabled")
            return None

        print("✅ Claude API key loaded successfully")
        return api_key

    def correct_social_media_prompt(self, content_data: Dict, banned_word: str, attempt_number: int) -> Optional[str]:
        """Use Claude to correct social media content prompts"""

        if not self.enabled:
            return None

        content_type = content_data.get("content_type", "unknown")
        content_id = content_data.get("content_id", 0)
        original_prompt = content_data.get("midjourney_prompt", "")

        if attempt_number == 1:
            severity = "carefully review and make minimal changes to avoid the banned word"
            tone = "Keep the aesthetic and visual details intact for social media"
        elif attempt_number == 2:
            severity = "more aggressively rewrite problem areas and similar risky words"
            tone = "Be more conservative but maintain social media appeal"
        elif attempt_number == 3:
            severity = "completely rewrite the problematic sections with safe alternatives"
            tone = "Prioritize Midjourney safety over original wording but keep social media optimized"
        else:
            severity = "completely restructure the entire prompt to be maximally safe"
            tone = "Create entirely new wording that achieves the same social media visual goal"

        system_message = f"""You are a Midjourney prompt expert specializing in social media content creation and content policy compliance. 

Midjourney has rejected a {content_type} prompt for containing the banned word: "{banned_word}"

Your task: {severity}. {tone}.

CRITICAL SOCIAL MEDIA RULES:
1. Keep the same aesthetic appeal for {content_type}
2. Maintain 9:16 aspect ratio optimization
3. Keep text overlay space considerations
4. Remove or replace ANY potentially problematic words
5. Add safety phrases like "appropriate content, family-friendly"
6. Keep the prompt under 4000 characters
7. This is attempt #{attempt_number}/4 - {'be more aggressive' if attempt_number > 2 else 'be careful but thorough'}
8. Optimize for mobile viewing and engagement

SOCIAL MEDIA SPECIFIC CONSIDERATIONS:
- Instagram: Aesthetic, clean, visually appealing
- YouTube Shorts: Engaging, thumbnail-worthy, dramatic
- TikTok: Educational, relatable, mobile-first

COMMON MIDJOURNEY BANNED WORDS TO AVOID:
intimate, romantic, bath, bathing, bedroom, bed, nude, naked, bare, undressed, children, child, kids, embrace, embracing, kiss, kissing, violence, blood, fight, sensual, seductive

Return ONLY the corrected prompt, nothing else."""

        user_message = f"""{content_type.upper()} #{content_id} - ATTEMPT #{attempt_number}

BANNED WORD DETECTED: "{banned_word}"

ORIGINAL PROMPT:
{original_prompt}

CONTENT CONTEXT:
- Title: {content_data.get('title', 'Unknown')}
- Platform: {content_type}
- Visual Style: {content_data.get('visual_elements', {}).get('instagram_aesthetic', 'social media optimized')}
- Duration: {content_data.get('duration_seconds', 60)} seconds

Please provide the corrected prompt that will pass Midjourney's content policy while maintaining social media appeal."""

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
            print(f"🧠 Claude: Correcting {content_type} {content_id} prompt (attempt {attempt_number}/4)")
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

                print(f"✅ Claude: {content_type} {content_id} prompt corrected")
                print(f"   Original length: {len(original_prompt)}")
                print(f"   Corrected length: {len(corrected_prompt)}")
                print(f"   Preview: {corrected_prompt[:100]}...")

                return corrected_prompt
            else:
                print(f"❌ Claude API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Claude correction failed: {e}")
            return None

class SocialMediaServerConfig:
    """Server configuration for social media visual generation"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_visual_config()
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
            'CONFIG_DIR': str(self.project_root / 'config'),
            'SOCIAL_MEDIA_DIR': str(self.project_root / 'output' / 'social_media')
        }

        print(f"✅ Social Media Generator paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_visual_config(self):
        """Setup Midjourney visual generation configuration"""
        self.visual_config = {
            "api_base_url": "https://api.piapi.ai/api/v1",
            "max_concurrent_tasks": 8,
            "max_wait_cycles": 30,
            "wait_interval_seconds": 30,
            "default_aspect_ratios": {
                "youtube_shorts": "9:16",
                "instagram_reels": "9:16",
                "tiktok_videos": "9:16"
            },
            "default_version": "7.0",
            "process_mode": "relax",
            "server_mode": True,
            "production_ready": True
        }

        self.api_key = self.get_midjourney_api_key()

    def get_midjourney_api_key(self):
        """Get Midjourney API key"""
        api_key = (
            os.getenv('PIAPI_KEY') or
            os.getenv('MIDJOURNEY_API_KEY') or
            os.getenv('PIAPI_API_KEY') or
            os.getenv('MIDJOURNEY_KEY')
        )

        if not api_key:
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
        logs_dir = Path(self.project_root) / 'logs' / 'social_media'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"social_media_gen_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("SocialMediaGenerator")
        self.logger.info(f"✅ Social media generator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR', 'SOCIAL_MEDIA_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All social media generator directories created/verified")

# Database Social Media Management Integration
class DatabaseSocialMediaManager:
    """Professional social media management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_social_media_columns()

    def setup_social_media_columns(self):
        """Add social media generation columns if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check existing columns
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add social media columns if they don't exist
        social_media_columns = [
            ('social_media_generation_status', 'TEXT DEFAULT "pending"'),
            ('social_media_generation_started_at', 'DATETIME'),
            ('social_media_generation_completed_at', 'DATETIME'),
            ('social_media_content_generated', 'INTEGER DEFAULT 0'),
            ('youtube_shorts_generated', 'INTEGER DEFAULT 0'),
            ('instagram_reels_generated', 'INTEGER DEFAULT 0'),
            ('tiktok_videos_generated', 'INTEGER DEFAULT 0'),
            ('social_media_claude_corrections', 'INTEGER DEFAULT 0'),
            ('social_media_api_calls', 'INTEGER DEFAULT 0')
        ]

        for column_name, column_definition in social_media_columns:
            if column_name not in columns:
                print(f"🔧 Adding social media column: {column_name}")
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()
        conn.close()
        print("✅ Social media generation columns verified/added")

    def get_completed_topic_ready_for_social_media(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed scene generation topic that needs SOCIAL MEDIA generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND scene_generation_status = 'completed'
            AND (social_media_generation_status IS NULL OR social_media_generation_status = 'pending')
            ORDER BY scene_generation_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_social_media_generation_started(self, topic_id: int):
        """Mark social media generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET social_media_generation_status = 'in_progress', 
                social_media_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_social_media_generation_completed(self, topic_id: int, results: Dict):
        """Mark social media generation as completed with detailed stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        youtube_count = results.get("platform_results", {}).get("youtube_shorts", {}).get("completed", 0)
        instagram_count = results.get("platform_results", {}).get("instagram_reels", {}).get("completed", 0)
        tiktok_count = results.get("platform_results", {}).get("tiktok_videos", {}).get("completed", 0)
        total_content = results.get("total_completed", 0)
        claude_corrections = results.get("claude_corrections_used", 0)
        api_calls = results.get("api_calls_made", 0)

        cursor.execute('''
            UPDATE topics 
            SET social_media_generation_status = 'completed',
                social_media_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                social_media_content_generated = ?,
                youtube_shorts_generated = ?,
                instagram_reels_generated = ?,
                tiktok_videos_generated = ?,
                social_media_claude_corrections = ?,
                social_media_api_calls = ?
            WHERE id = ?
        ''', (total_content, youtube_count, instagram_count, tiktok_count, claude_corrections, api_calls, topic_id))

        conn.commit()
        conn.close()

    def get_topic_historical_period(self, topic_id: int) -> Optional[str]:
        """Get historical period for topic"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT historical_period FROM topics WHERE id = ?', (topic_id,))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] else "ancient times"

class SocialMediaVisualGenerator:
    """Social media content visual generator with Claude AI correction and Database integration"""

    def __init__(self, config: SocialMediaServerConfig):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.visual_config["api_base_url"]

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
        self.api_calls_made = 0
        self.successful_downloads = 0
        self.failed_content = {}

        # Initialize Claude corrector
        self.claude_corrector = ClaudeSocialMediaPromptCorrector()

        # Database manager
        db_path = Path(config.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseSocialMediaManager(str(db_path))

        print("🚀 Social Media Visual Generator with Database Integration")
        print(f"🔑 Midjourney API Key: {self.api_key[:8]}...")
        print(f"🧠 Claude AI Correction: {'✅ Enabled' if self.claude_corrector.enabled else '❌ Disabled'}")
        print(f"🗄️ Database: {db_path}")

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps"""
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "api_calls": self.api_calls_made,
            "successful_downloads": self.successful_downloads,
            "metadata": metadata or {}
        }
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step} [Calls: {self.api_calls_made}] [Downloads: {self.successful_downloads}]")
        self.config.logger.info(f"{step} - Status: {status}")

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed scene project that needs SOCIAL MEDIA generation"""
        self.log_step("🔍 Finding completed scene project for social media generation")

        result = self.db_manager.get_completed_topic_ready_for_social_media()

        if not result:
            self.log_step("✅ No completed scene projects ready for social media generation", "INFO")
            return False, None

        topic_id, topic, description, output_path = result

        # Setup project paths
        self.current_topic_id = topic_id
        self.current_output_dir = output_path
        self.current_topic = topic
        self.current_description = description

        # Get historical period from database
        self.current_historical_period = self.db_manager.get_topic_historical_period(topic_id)

        project_info = {
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "output_dir": output_path,
            "historical_period": self.current_historical_period
        }

        # Mark as started in database
        self.db_manager.mark_social_media_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def load_social_media_content_from_project(self) -> Dict:
        """Load social media content from project's social_media_content.json"""
        self.log_step("📂 Loading social media content from project")

        output_dir = Path(self.current_output_dir)
        social_media_json_path = output_dir / "social_media_content.json"

        if not social_media_json_path.exists():
            raise FileNotFoundError(f"Social media content not found: {social_media_json_path}")

        with open(social_media_json_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)

        youtube_shorts = content_data.get("youtube_shorts", [])
        instagram_reels = content_data.get("instagram_reels", [])
        tiktok_videos = content_data.get("tiktok_videos", [])

        total_content = len(youtube_shorts) + len(instagram_reels) + len(tiktok_videos)

        self.log_step("✅ Social media content loaded from project", "SUCCESS", {
            "youtube_shorts": len(youtube_shorts),
            "instagram_reels": len(instagram_reels),
            "tiktok_videos": len(tiktok_videos),
            "total_content": total_content,
            "source_file": str(social_media_json_path)
        })

        return content_data

    def setup_output_directories_in_project(self) -> Path:
        """Create social media directories within the project's output directory"""

        # Create social media directory within project
        project_dir = Path(self.current_output_dir)
        social_media_dir = project_dir / "social_media"
        social_media_dir.mkdir(exist_ok=True)

        # Create platform-specific directories
        self.youtube_dir = social_media_dir / "youtube_shorts"
        self.instagram_dir = social_media_dir / "instagram_reels"
        self.tiktok_dir = social_media_dir / "tiktok_videos"

        self.youtube_dir.mkdir(exist_ok=True)
        self.instagram_dir.mkdir(exist_ok=True)
        self.tiktok_dir.mkdir(exist_ok=True)

        self.social_media_dir = social_media_dir

        print(f"📁 Social media directories created in project:")
        print(f"   📁 Base: {social_media_dir}")
        print(f"   📺 YouTube Shorts: {self.youtube_dir}")
        print(f"   📸 Instagram Reels: {self.instagram_dir}")
        print(f"   🎵 TikTok Videos: {self.tiktok_dir}")

        return social_media_dir

    def sanitize_filename(self, name: str) -> str:
        """Sanitize filename for cross-platform compatibility"""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('._')
        return sanitized[:100]  # Limit length

    def extract_banned_word_from_error(self, error_response: Dict) -> Optional[str]:
        """Extract banned word from Midjourney error response"""
        try:
            error_data = error_response.get("error", {})
            raw_message = error_data.get("raw_message", "")

            if "Banned Prompt:" in raw_message:
                banned_word = raw_message.split("Banned Prompt:")[-1].strip()
                return banned_word

            message = error_response.get("message", "")
            if "Banned Prompt:" in message:
                banned_word = message.split("Banned Prompt:")[-1].strip()
                return banned_word

            return None
        except:
            return None

    def clean_prompt_for_piapi(self, prompt: str) -> str:
        """Clean prompt for PiAPI submission"""
        import re

        # Remove all --ar parameters
        prompt = re.sub(r'--ar\s+\d+:\d+', '', prompt)
        # Remove all --v parameters
        prompt = re.sub(r'--v\s+[\d.]+', '', prompt)
        # Remove any other -- parameters
        prompt = re.sub(r'--\w+(?:\s+[\w:.]+)?', '', prompt)

        # Clean up problematic characters
        prompt = prompt.replace(' - ', ' ')
        prompt = prompt.replace('-', ' ')

        # Normalize spaces
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip()

        return prompt

    def submit_social_media_task_with_correction(self, content_data: Dict, aspect_ratio: str = "9:16") -> Optional[str]:
        """Submit social media task with Claude correction"""

        content_type = content_data.get("content_type", "unknown")
        content_id = content_data.get("content_id", 0)
        original_prompt = content_data.get("midjourney_prompt", "")

        max_attempts = self.claude_corrector.max_attempts
        current_prompt = original_prompt

        # Track correction attempts
        content_key = f"{content_type}_{content_id}"
        if content_key not in self.claude_corrector.correction_attempts:
            self.claude_corrector.correction_attempts[content_key] = 0

        for attempt in range(max_attempts + 1):

            if attempt == 0:
                print(f"📱 {content_type} {content_id}: Submitting original prompt")
            else:
                print(f"🔄 {content_type} {content_id}: Claude correction attempt {attempt}/{max_attempts}")

            # Clean prompt for API
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

            try:
                self.api_calls_made += 1
                response = requests.post(f"{self.base_url}/task", headers=self.headers, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("code") == 200:
                        task_data = result.get("data", {})
                        task_id = task_data.get("task_id")

                        if attempt > 0:
                            print(f"✅ {content_type} {content_id}: Submitted after Claude correction #{attempt}")

                        return task_id
                    else:
                        # Check for banned prompt
                        error_message = result.get('message', '')

                        if "failed to check prompt" in error_message or result.get("error", {}).get("code") == 10000:
                            banned_word = self.extract_banned_word_from_error(result)

                            if banned_word and self.claude_corrector.enabled and attempt < max_attempts:
                                print(f"🛡️ {content_type} {content_id}: Banned word: '{banned_word}'")

                                # Use Claude to correct
                                corrected_prompt = self.claude_corrector.correct_social_media_prompt(
                                    content_data, banned_word, attempt + 1
                                )

                                if corrected_prompt:
                                    current_prompt = corrected_prompt
                                    self.claude_corrector.correction_attempts[content_key] += 1
                                    print(f"🧠 {content_type} {content_id}: Trying Claude-corrected prompt")
                                    time.sleep(2)
                                    continue
                                else:
                                    print(f"❌ {content_type} {content_id}: Claude correction failed")
                                    return None
                            else:
                                print(f"❌ {content_type} {content_id}: Banned prompt - no more attempts")
                                return None
                        else:
                            print(f"❌ {content_type} {content_id}: API Error: {result.get('message', 'Unknown')}")
                            return None

                elif response.status_code == 500:
                    # Check HTTP 500 for banned prompt
                    print(f"⚠️ {content_type} {content_id}: HTTP 500 - checking for banned prompt")
                    try:
                        error_response = response.json()
                        data = error_response.get("data", {})
                        if data:
                            error_data = data.get("error", {})
                            raw_message = error_data.get("raw_message", "")
                        else:
                            error_data = error_response.get("error", {})
                            raw_message = error_data.get("raw_message", "")

                        if "Banned Prompt:" in raw_message:
                            banned_word = raw_message.split("Banned Prompt:")[-1].strip()

                            if banned_word and self.claude_corrector.enabled and attempt < max_attempts:
                                print(f"🛡️ {content_type} {content_id}: HTTP 500 banned word: '{banned_word}'")

                                corrected_prompt = self.claude_corrector.correct_social_media_prompt(
                                    content_data, banned_word, attempt + 1
                                )

                                if corrected_prompt:
                                    current_prompt = corrected_prompt
                                    self.claude_corrector.correction_attempts[content_key] += 1
                                    print(f"🧠 {content_type} {content_id}: Trying Claude correction")
                                    time.sleep(2)
                                    continue
                                else:
                                    return None
                            else:
                                return None
                        else:
                            print(f"❌ {content_type} {content_id}: HTTP 500 - server error")
                            if self.claude_corrector.enabled and attempt < max_attempts:
                                # Try generic correction
                                corrected_prompt = self.claude_corrector.correct_social_media_prompt(
                                    content_data, "content policy violation", attempt + 1
                                )

                                if corrected_prompt:
                                    current_prompt = corrected_prompt
                                    self.claude_corrector.correction_attempts[content_key] += 1
                                    print(f"🧠 {content_type} {content_id}: Trying generic Claude correction")
                                    time.sleep(2)
                                    continue

                            return None

                    except Exception as e:
                        print(f"❌ {content_type} {content_id}: HTTP 500 parse error: {e}")
                        return None
                else:
                    print(f"❌ {content_type} {content_id}: HTTP Error: {response.status_code}")
                    return None

            except Exception as e:
                print(f"❌ {content_type} {content_id}: Request failed: {e}")
                return None

        print(f"❌ {content_type} {content_id}: Failed after {max_attempts} Claude attempts")
        return None

    def check_task_status(self, task_id: str, content_type: str, content_id: int) -> Optional[Dict]:
        """Check task status"""
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
                            print(f"⚠️ {content_type} {content_id}: Completed but no URLs")
                            return False

                    elif status == "failed":
                        error_info = task_data.get("error", {})
                        error_msg = error_info.get("message", "Unknown error")
                        print(f"❌ {content_type} {content_id}: Task failed - {error_msg}")
                        return False
                    else:
                        return None

            return None

        except Exception as e:
            print(f"⚠️ {content_type} {content_id}: Status check error - {e}")
            return None

    def download_image(self, result_data: Dict, save_path: str, content_type: str, content_id: int) -> bool:
        """Download generated image"""
        image_url = result_data["url"]

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://discord.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }

            print(f"📥 {content_type} {content_id}: Downloading...")

            response = requests.get(image_url, headers=headers, timeout=30, stream=True)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = os.path.getsize(save_path)
                print(f"✅ {content_type} {content_id}: Downloaded ({file_size} bytes)")
                self.successful_downloads += 1
                return True
            else:
                print(f"❌ {content_type} {content_id}: Download failed HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ {content_type} {content_id}: Download error - {e}")
            return False

    def generate_platform_content(self, content_list: List[Dict], platform_name: str, output_dir: Path) -> Dict:
        """Generate visuals for a specific platform"""

        print(f"\n📱 GENERATING {platform_name.upper()} CONTENT")
        print(f"🎬 Total content: {len(content_list)}")

        results = {
            "total": len(content_list),
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "tasks": {}
        }

        if not content_list:
            return results

        # Submit tasks
        for content in content_list:
            content_id = content.get(f"{platform_name.split('_')[0]}_id", content.get("reel_id", content.get("tiktok_id", 0)))

            # Add content type and ID for Claude correction
            content["content_type"] = platform_name
            content["content_id"] = content_id

            print(f"\n🎬 Processing {platform_name} {content_id}")
            print(f"   Title: {content.get('title', 'No title')}")

            # Check if already exists
            filename = f"{platform_name}_{content_id:02d}.png"
            save_path = output_dir / filename

            if save_path.exists():
                print(f"⏭️  {platform_name} {content_id}: Already exists, skipping")
                results["completed"] += 1
                continue

            # Submit task with Claude correction
            task_id = self.submit_social_media_task_with_correction(content, aspect_ratio="9:16")

            if task_id:
                results["tasks"][content_id] = {
                    "task_id": task_id,
                    "content_data": content,
                    "save_path": str(save_path)
                }
                results["submitted"] += 1
                print(f"✅ {platform_name} {content_id}: Task submitted")
            else:
                print(f"❌ {platform_name} {content_id}: Submission failed")
                results["failed"] += 1

            # Rate limiting
            time.sleep(5)

        # Monitor tasks
        if results["tasks"]:
            print(f"\n⏳ Monitoring {len(results['tasks'])} {platform_name} tasks...")

            max_cycles = 40
            for cycle in range(max_cycles):
                if not results["tasks"]:
                    break

                completed_count = results["completed"]
                total_count = results["total"]
                print(f"📊 {platform_name} Cycle {cycle + 1}: {completed_count}/{total_count} completed")

                tasks_to_remove = []

                for content_id, task_data in results["tasks"].items():
                    task_id = task_data["task_id"]

                    result_data = self.check_task_status(task_id, platform_name, content_id)

                    if result_data and isinstance(result_data, dict):
                        print(f"✅ {platform_name} {content_id}: Task completed!")

                        # Download image
                        if self.download_image(result_data, task_data["save_path"], platform_name, content_id):
                            results["completed"] += 1

                            # Save metadata
                            metadata = {
                                "content_id": content_id,
                                "platform": platform_name,
                                "title": task_data["content_data"].get("title", ""),
                                "generated_at": datetime.now().isoformat(),
                                "image_url": result_data["url"],
                                "local_path": task_data["save_path"],
                                "claude_corrections": self.claude_corrector.correction_attempts.get(f"{platform_name}_{content_id}", 0)
                            }

                            json_path = Path(task_data["save_path"]).with_suffix('.json')
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)

                        tasks_to_remove.append(content_id)

                    elif result_data is False:
                        print(f"❌ {platform_name} {content_id}: Task failed")
                        results["failed"] += 1
                        tasks_to_remove.append(content_id)

                for content_id in tasks_to_remove:
                    del results["tasks"][content_id]

                if not results["tasks"]:
                    break

                time.sleep(30)

        return results

    def run_database_integrated_generation(self) -> bool:
        """Run social media generation integrated with database"""
        print("🚀" * 50)
        print("DATABASE INTEGRATED SOCIAL MEDIA GENERATOR v1.0")
        print("🧠 CLAUDE SONNET 4 PROMPT CORRECTION")
        print("🗄️ DATABASE INTEGRATED WORKFLOW")
        print("📱 MULTI-PLATFORM CONTENT GENERATION")
        print("🎬 YouTube Shorts + Instagram Reels + TikTok")
        print("🚀" * 50)

        # Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🏛️ Historical period: {project_info['historical_period']}")

        try:
            # Load social media content from project
            content_data = self.load_social_media_content_from_project()

            # Setup directories within project
            social_media_dir = self.setup_output_directories_in_project()

            # Generate platform content
            youtube_results = self.generate_platform_content(
                content_data.get("youtube_shorts", []),
                "youtube_shorts",
                self.youtube_dir
            )

            instagram_results = self.generate_platform_content(
                content_data.get("instagram_reels", []),
                "instagram_reels",
                self.instagram_dir
            )

            tiktok_results = self.generate_platform_content(
                content_data.get("tiktok_videos", []),
                "tiktok_videos",
                self.tiktok_dir
            )

            # Generate summary report
            total_content = youtube_results["total"] + instagram_results["total"] + tiktok_results["total"]
            total_completed = youtube_results["completed"] + instagram_results["completed"] + tiktok_results["completed"]
            total_failed = youtube_results["failed"] + instagram_results["failed"] + tiktok_results["failed"]

            report = {
                "topic_id": self.current_topic_id,
                "topic": self.current_topic,
                "description": self.current_description,
                "historical_period": self.current_historical_period,
                "generation_completed": datetime.now().isoformat(),
                "total_content": total_content,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "success_rate": total_completed / total_content if total_content > 0 else 0,
                "api_calls_made": self.api_calls_made,
                "claude_corrections_used": sum(self.claude_corrector.correction_attempts.values()),
                "platform_results": {
                    "youtube_shorts": youtube_results,
                    "instagram_reels": instagram_results,
                    "tiktok_videos": tiktok_results
                },
                "claude_correction_attempts": dict(self.claude_corrector.correction_attempts),
                "output_directories": {
                    "social_media_dir": str(social_media_dir),
                    "youtube_dir": str(self.youtube_dir),
                    "instagram_dir": str(self.instagram_dir),
                    "tiktok_dir": str(self.tiktok_dir)
                }
            }

            # Save report
            report_path = social_media_dir / "social_media_generation_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Update database with results
            self.db_manager.mark_social_media_generation_completed(self.current_topic_id, report)

            # Print final summary
            print(f"\n🎉 SOCIAL MEDIA GENERATION COMPLETED!")
            print(f"📊 Results Summary:")
            print(f"   ✅ Total completed: {total_completed}/{total_content}")
            print(f"   📺 YouTube Shorts: {youtube_results['completed']}/{youtube_results['total']}")
            print(f"   📸 Instagram Reels: {instagram_results['completed']}/{instagram_results['total']}")
            print(f"   🎵 TikTok Videos: {tiktok_results['completed']}/{tiktok_results['total']}")
            print(f"   🧠 Claude corrections: {sum(self.claude_corrector.correction_attempts.values())}")
            print(f"   📁 Output: {social_media_dir}")
            print(f"   🗄️ Database updated for Topic ID: {self.current_topic_id}")

            success_rate = total_completed / total_content if total_content > 0 else 0
            return success_rate >= 0.7

        except Exception as e:
            print(f"💥 Social media generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run database integrated social media generation"""

    print("🚀 DATABASE INTEGRATED SOCIAL MEDIA VISUAL GENERATOR")
    print("🗄️ Automatically processes completed scene generation topics")
    print("📱 Multi-platform content generation with Claude AI correction")
    print("🎬 No manual JSON file needed - reads from project directories")

    try:
        # Initialize configuration
        config = SocialMediaServerConfig()

        # Initialize generator
        generator = SocialMediaVisualGenerator(config)

        # Run database integrated generation
        success = generator.run_database_integrated_generation()

        if success:
            print("🎊 Social media visual generation completed successfully!")
            sys.exit(0)
        else:
            print("⚠️ No topics ready for social media generation or generation failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Social media generation stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Social media generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()