"""
Sleepy Dull Stories - AUTONOMOUS TTS Audio Generator
YouTube Hook + Subscribe + Voice Directions + Enceladus Voice + AUTONOMOUS MODE
Production-optimized with complete automation and error recovery
"""

import os
import re
import time
import json
import pandas as pd
from google.cloud import texttospeech
from pydub import AudioSegment
from dotenv import load_dotenv
from datetime import datetime

# Server-specific imports
import sqlite3
from pathlib import Path
import sys
import signal
import logging
from typing import Dict, List, Optional, Tuple

# Load environment first
load_dotenv()

# VOICE CONFIGURATION - Enceladus Voice
VOICE_NAME = "en-US-Chirp3-HD-Enceladus"
LANGUAGE_CODE = "en-US"


# Server Configuration Class (from visual generator)
class ServerConfig:
    """Server-friendly configuration management for TTS Generator"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_audio_config()
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

        print(f"✅ TTS Generator server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_audio_config(self):
        """Setup TTS audio generation configuration"""
        self.audio_config = {
            "voice_name": VOICE_NAME,
            "language_code": LANGUAGE_CODE,
            "max_retry_rounds": 5,
            "max_retries_per_chunk": 5,
            "chunk_max_length": 3000,
            "auto_split_enabled": True,
            "quality_profiles": {
                "youtube": {"bitrate": "192k", "description": "YouTube High Quality", "sample_rate": 44100},
                "podcast": {"bitrate": "128k", "description": "Podcast Quality", "sample_rate": 44100},
                "balanced": {"bitrate": "96k", "description": "Balanced Quality", "sample_rate": 22050}
            },
            "cost_per_million_chars": 16.0,  # Chirp3-HD pricing
            "budget_controls": {
                "max_cost_per_story_usd": 5.0,  # Max $5 per story
                "max_cost_per_session_usd": 25.0,  # Max $25 per session
                "emergency_stop_cost_usd": 50.0,  # Emergency stop at $50
                "warn_threshold_usd": 2.5  # Warn at $2.5 per story
            },
            "server_mode": True,
            "production_ready": True
        }

        print("✅ TTS audio configuration loaded")
        print(
            f"💰 Budget controls: ${self.audio_config['budget_controls']['max_cost_per_story_usd']}/story, ${self.audio_config['budget_controls']['max_cost_per_session_usd']}/session")

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"audio_gen_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("AudioGenerator")
        self.logger.info(f"✅ Audio generator logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All audio generator directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Audio Generator server configuration loaded successfully")
except Exception as e:
    print(f"❌ Audio Generator server configuration failed: {e}")
    sys.exit(1)


# Database Audio Management Integration
class DatabaseAudioManager:
    """Professional audio management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_scene_topic_ready_for_audio(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed scene topic that needs AUDIO generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if audio generation columns exist, if not create them
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add columns individually if they don't exist
        columns_to_add = [
            ('audio_generation_status', 'TEXT DEFAULT "pending"'),
            ('audio_generation_started_at', 'DATETIME'),
            ('audio_generation_completed_at', 'DATETIME'),
            ('audio_chunks_generated', 'INTEGER DEFAULT 0'),
            ('audio_duration_seconds', 'REAL DEFAULT 0.0'),
            ('audio_cost_usd', 'REAL DEFAULT 0.0')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                print(f"🔧 Adding column: {column_name}")
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()
        print("✅ Audio generation columns verified/added")

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND scene_generation_status = 'completed'
            AND (audio_generation_status IS NULL OR audio_generation_status = 'pending')
            ORDER BY scene_generation_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_audio_generation_started(self, topic_id: int):
        """Mark audio generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET audio_generation_status = 'in_progress', 
                audio_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_audio_generation_completed(self, topic_id: int, chunks_count: int,
                                        duration_seconds: float, cost_usd: float):
        """Mark audio generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET audio_generation_status = 'completed',
                audio_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                audio_chunks_generated = ?,
                audio_duration_seconds = ?,
                audio_cost_usd = ?
            WHERE id = ?
        ''', (chunks_count, duration_seconds, cost_usd, topic_id))

        conn.commit()
        conn.close()

    def mark_audio_generation_failed(self, topic_id: int, error_message: str):
        """Mark audio generation as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET audio_generation_status = 'failed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def get_pipeline_status(self) -> Dict:
        """Get overall pipeline status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Add columns if they don't exist
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        columns_to_add = [
            ('audio_generation_status', 'TEXT DEFAULT "pending"'),
            ('audio_generation_started_at', 'DATETIME'),
            ('audio_generation_completed_at', 'DATETIME'),
            ('audio_chunks_generated', 'INTEGER DEFAULT 0'),
            ('audio_duration_seconds', 'REAL DEFAULT 0.0'),
            ('audio_cost_usd', 'REAL DEFAULT 0.0')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()

        # Count audio generation queue
        cursor.execute('''
            SELECT COUNT(*) FROM topics 
            WHERE status = 'completed' 
            AND scene_generation_status = 'completed'
            AND (audio_generation_status IS NULL OR audio_generation_status = 'pending')
        ''')
        audio_queue = cursor.fetchone()[0]

        # Count active audio generation
        cursor.execute('''
            SELECT COUNT(*) FROM topics 
            WHERE audio_generation_status = 'in_progress'
        ''')
        audio_active = cursor.fetchone()[0]

        conn.close()

        return {
            'audio_generation_queue': audio_queue,
            'audio_generation_active': audio_active
        }


class ProgressTracker:
    """Scene processing progress tracking ve resume functionality"""

    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.audio_parts_dir = os.path.join(self.output_dir, "audio_parts")
        self.progress_file = os.path.join(self.output_dir, "audio_progress.json")

        # Ensure directories exist
        os.makedirs(self.audio_parts_dir, exist_ok=True)

        # Load existing progress
        self.progress_data = self.load_progress()

    def load_progress(self):
        """Existing progress'i yükle"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 Progress loaded: {len(data.get('completed_chunks', []))} chunks completed")
                return data
            except Exception as e:
                print(f"⚠️  Progress file corrupted, starting fresh: {e}")

        return {
            "story_id": self.story_id,
            "completed_chunks": [],
            "failed_chunks": [],
            "blacklisted_chunks": [],  # New: chunks that failed too many times
            "chunk_attempt_count": {},  # New: track attempts per chunk
            "total_cost_so_far": 0.0,
            "total_characters_so_far": 0,
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat()
        }

    def save_progress(self):
        """Progress'i kaydet"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Progress save warning: {e}")

    def is_chunk_completed(self, chunk_name: str) -> bool:
        """Chunk tamamlanmış mı kontrol et"""
        return chunk_name in self.progress_data.get("completed_chunks", [])

    def is_chunk_blacklisted(self, chunk_name: str) -> bool:
        """Chunk blacklist'te mi kontrol et"""
        return chunk_name in self.progress_data.get("blacklisted_chunks", [])

    def get_chunk_audio_path(self, chunk_name: str) -> str:
        """Chunk için audio dosya path'i"""
        return os.path.join(self.audio_parts_dir, f"{chunk_name}_audio.mp3")

    def increment_chunk_attempts(self, chunk_name: str) -> int:
        """Chunk attempt sayısını artır ve döndür"""
        if "chunk_attempt_count" not in self.progress_data:
            self.progress_data["chunk_attempt_count"] = {}

        current_attempts = self.progress_data["chunk_attempt_count"].get(chunk_name, 0)
        self.progress_data["chunk_attempt_count"][chunk_name] = current_attempts + 1
        self.save_progress()
        return current_attempts + 1

    def blacklist_chunk(self, chunk_name: str, reason: str):
        """Chunk'ı blacklist'e ekle"""
        if "blacklisted_chunks" not in self.progress_data:
            self.progress_data["blacklisted_chunks"] = []

        if chunk_name not in self.progress_data["blacklisted_chunks"]:
            self.progress_data["blacklisted_chunks"].append(chunk_name)
            attempts = self.progress_data.get("chunk_attempt_count", {}).get(chunk_name, 0)
            print(f"⚫ {chunk_name}: Blacklisted after {attempts} failed attempts - {reason}")
            self.save_progress()

    def mark_chunk_completed(self, chunk_name: str, char_count: int, cost: float):
        """Chunk'ı tamamlandı olarak işaretle"""
        if chunk_name not in self.progress_data["completed_chunks"]:
            self.progress_data["completed_chunks"].append(chunk_name)
            self.progress_data["total_cost_so_far"] += cost
            self.progress_data["total_characters_so_far"] += char_count
            self.save_progress()
            print(f"      💾 Progress saved: {chunk_name} completed")

    def mark_chunk_failed(self, chunk_name: str, error: str):
        """Chunk'ı failed olarak işaretle"""
        if "failed_chunks" not in self.progress_data:
            self.progress_data["failed_chunks"] = []

        # Remove existing entry for this chunk if exists
        self.progress_data["failed_chunks"] = [
            f for f in self.progress_data["failed_chunks"]
            if not (isinstance(f, dict) and f.get("chunk_name") == chunk_name)
        ]

        # Add new failure record
        self.progress_data["failed_chunks"].append({
            "chunk_name": chunk_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.save_progress()

    def get_resume_summary(self):
        """Resume özeti"""
        completed = len(self.progress_data.get("completed_chunks", []))
        blacklisted = len(self.progress_data.get("blacklisted_chunks", []))
        cost = self.progress_data.get("total_cost_so_far", 0.0)
        chars = self.progress_data.get("total_characters_so_far", 0)

        return {
            "completed_chunks": completed,
            "blacklisted_chunks": blacklisted,
            "total_cost_so_far": cost,
            "total_characters_so_far": chars,
            "can_resume": completed > 0
        }

    def cleanup_on_success(self):
        """Başarılı tamamlandığında progress dosyasını temizle"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print(f"🗑️  Progress file cleaned up (successful completion)")
        except Exception as e:
            print(f"⚠️  Progress cleanup warning: {e}")


class UsageTracker:
    """Google Cloud TTS API usage ve cost tracking with budget controls"""

    def __init__(self):
        self.session_start = datetime.now()
        self.total_characters = 0
        self.total_requests = 0
        self.requests_log = []
        # Chirp3-HD pricing (premium voice)
        self.cost_per_million_chars = CONFIG.audio_config["cost_per_million_chars"]
        self.session_cost = 0.0
        self.story_cost = 0.0

        # Budget controls
        self.budget_controls = CONFIG.audio_config.get("budget_controls", {})
        self.budget_warnings_shown = set()

    def check_budget_limits(self, story_id: int = None) -> Tuple[bool, str]:
        """Check if budget limits are exceeded - returns (can_continue, reason)"""
        total_cost = self.get_total_cost()

        # Emergency stop check
        emergency_limit = self.budget_controls.get("emergency_stop_cost_usd", 50.0)
        if total_cost >= emergency_limit:
            return False, f"EMERGENCY STOP: Session cost ${total_cost:.2f} >= ${emergency_limit}"

        # Session limit check
        session_limit = self.budget_controls.get("max_cost_per_session_usd", 25.0)
        if total_cost >= session_limit:
            return False, f"SESSION LIMIT: Cost ${total_cost:.2f} >= ${session_limit}"

        # Story limit check (if we're tracking story cost)
        story_limit = self.budget_controls.get("max_cost_per_story_usd", 5.0)
        if self.story_cost >= story_limit:
            return False, f"STORY LIMIT: Story cost ${self.story_cost:.2f} >= ${story_limit}"

        # Warning thresholds
        warn_threshold = self.budget_controls.get("warn_threshold_usd", 2.5)
        if self.story_cost >= warn_threshold and "story_warn" not in self.budget_warnings_shown:
            print(f"⚠️  BUDGET WARNING: Story cost ${self.story_cost:.2f} approaching limit ${story_limit}")
            self.budget_warnings_shown.add("story_warn")

        return True, "OK"

    def reset_story_cost(self):
        """Reset story cost counter for new story"""
        self.story_cost = 0.0
        if "story_warn" in self.budget_warnings_shown:
            self.budget_warnings_shown.remove("story_warn")

    def add_request(self, char_count, chunk_name, duration_seconds=0):
        """TTS request'i log'a ekle with budget tracking"""
        self.total_characters += char_count
        self.total_requests += 1

        request_cost = (char_count / 1_000_000) * self.cost_per_million_chars
        self.session_cost += request_cost
        self.story_cost += request_cost

        request_log = {
            "request_number": self.total_requests,
            "chunk_name": chunk_name,
            "characters": char_count,
            "cost": request_cost,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        }

        self.requests_log.append(request_log)

        # Progress update with budget info
        total_cost = self.get_total_cost()
        print(
            f"      💰 Cost: ${request_cost:.6f} | Story: ${self.story_cost:.4f} | Session: ${total_cost:.4f} | Chars: {self.total_characters:,}")

        # Check budget after each request
        can_continue, reason = self.check_budget_limits()
        if not can_continue:
            print(f"🚨 BUDGET LIMIT EXCEEDED: {reason}")

        return request_cost  # Return individual cost for progress tracking

    def get_total_cost(self):
        """Toplam maliyeti hesapla"""
        return (self.total_characters / 1_000_000) * self.cost_per_million_chars

    def get_session_duration(self):
        """Session süresini hesapla"""
        return (datetime.now() - self.session_start).total_seconds()

    def print_progress_summary(self, current_step, total_steps):
        """İlerleme özeti yazdır"""
        progress_percent = (current_step / total_steps) * 100
        total_cost = self.get_total_cost()
        session_duration = self.get_session_duration()

        print(f"\n📊 PROGRESS UPDATE ({current_step}/{total_steps} - {progress_percent:.1f}%)")
        print(f"   📝 Characters processed: {self.total_characters:,}")
        print(f"   🔄 Requests made: {self.total_requests}")
        print(f"   💰 Session cost: ${total_cost:.4f}")
        print(f"   ⏱️  Session time: {session_duration / 60:.1f} minutes")
        if current_step > 0:
            avg_cost_per_request = total_cost / self.total_requests
            print(f"   📈 Avg cost/request: ${avg_cost_per_request:.6f}")

    def print_final_summary(self):
        """Final session özeti"""
        total_cost = self.get_total_cost()
        session_duration = self.get_session_duration()
        free_tier_usage = (self.total_characters / 1_000_000) * 100

        print(f"\n💰 FINAL USAGE SUMMARY")
        print(f"=" * 50)
        print(f"📝 Total characters: {self.total_characters:,}")
        print(f"🔄 Total requests: {self.total_requests}")
        print(f"💰 Total cost: ${total_cost:.4f}")
        print(f"🆓 Free tier usage: {free_tier_usage:.2f}%")
        print(f"⏱️  Session duration: {session_duration / 60:.1f} minutes")

        if self.total_requests > 0:
            avg_chars_per_request = self.total_characters / self.total_requests
            avg_cost_per_request = total_cost / self.total_requests
            chars_per_second = self.total_characters / session_duration if session_duration > 0 else 0

            print(f"📊 Averages:")
            print(f"   📝 {avg_chars_per_request:.0f} chars/request")
            print(f"   💰 ${avg_cost_per_request:.6f} cost/request")
            print(f"   🚀 {chars_per_second:.1f} chars/second")

        # Free tier remaining
        free_tier_limit = 1_000_000  # 1M characters free per month
        remaining_chars = free_tier_limit - self.total_characters
        print(f"🎁 Free tier remaining: {remaining_chars:,} characters")

        return {
            "total_characters": self.total_characters,
            "total_requests": self.total_requests,
            "total_cost": total_cost,
            "session_duration_minutes": session_duration / 60,
            "free_tier_usage_percent": free_tier_usage
        }


class ServerTTSGenerator:
    """Server-ready TTS generator with database integration"""

    def __init__(self):
        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseAudioManager(str(db_path))

        print("🚀 Server TTS Generator v1.0 Initialized")
        print(f"🎙️ Voice: {VOICE_NAME}")

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps with production logging"""
        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step}")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed scene project that needs AUDIO generation"""
        self.log_step("🔍 Finding completed scene project for audio generation")

        result = self.db_manager.get_completed_scene_topic_ready_for_audio()

        if not result:
            self.log_step("✅ No completed scene projects ready for audio generation", "INFO")
            return False, None

        topic_id, topic, description, output_path = result

        # Setup project paths
        self.current_topic_id = topic_id
        self.current_output_dir = output_path
        self.current_topic = topic
        self.current_description = description

        project_info = {
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "output_dir": output_path
        }

        # Mark as started in database
        self.db_manager.mark_audio_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def extract_hook_and_subscribe(self, complete_text):
        """Complete story'den Hook ve Subscribe kısımlarını çıkar"""
        hook_pattern = r'=== GOLDEN HOOK \(0-30 seconds\) ===(.*?)=== SUBSCRIBE REQUEST'
        subscribe_pattern = r'=== SUBSCRIBE REQUEST \(30-60 seconds\) ===(.*?)=== MAIN STORY'

        hook_match = re.search(hook_pattern, complete_text, re.DOTALL)
        subscribe_match = re.search(subscribe_pattern, complete_text, re.DOTALL)

        hook_text = hook_match.group(1).strip() if hook_match else None
        subscribe_text = subscribe_match.group(1).strip() if subscribe_match else None

        if hook_text and subscribe_text:
            return {
                "hook": hook_text,
                "subscribe": subscribe_text
            }
        return None

    def load_stories_and_directions(self, story_id: int):
        """All stories, complete story ve voice directions dosyalarını yükle"""
        story_dir = Path(self.current_output_dir)

        # all_stories.json dosyasını yükle
        stories_path = story_dir / "all_stories.json"
        if not stories_path.exists():
            return None, None, None, f"all_stories.json bulunamadı: {stories_path}"

        try:
            with open(stories_path, 'r', encoding='utf-8') as f:
                stories_data = json.load(f)
        except Exception as e:
            return None, None, None, f"all_stories.json okuma hatası: {e}"

        # complete_story.txt dosyasını yükle (Hook ve Subscribe için)
        complete_story_path = story_dir / "complete_story.txt"
        hook_and_subscribe = None
        if complete_story_path.exists():
            try:
                with open(complete_story_path, 'r', encoding='utf-8') as f:
                    complete_text = f.read()
                hook_and_subscribe = self.extract_hook_and_subscribe(complete_text)
            except Exception as e:
                print(f"⚠️  complete_story.txt okuma hatası: {e}")

        # voice_directions.json dosyasını yükle
        voice_directions_path = story_dir / "voice_directions.json"
        if not voice_directions_path.exists():
            return stories_data, hook_and_subscribe, None, f"voice_directions.json bulunamadı: {voice_directions_path}"

        try:
            with open(voice_directions_path, 'r', encoding='utf-8') as f:
                voice_directions = json.load(f)
        except Exception as e:
            return stories_data, hook_and_subscribe, None, f"voice_directions.json okuma hatası: {e}"

        return stories_data, hook_and_subscribe, voice_directions, None

    def check_scene_images(self, story_id: int, scene_count: int, has_hook_subscribe: bool = False):
        """Scene image'larının varlığını kontrol et - Hook/Subscribe scene_01.png kullanır"""
        story_dir = Path(self.current_output_dir)
        scenes_dir = story_dir / "scenes"

        if not scenes_dir.exists():
            return [], f"Scenes klasörü bulunamadı: {scenes_dir}"

        available_scenes = []
        missing_scenes = []

        # Story scene'leri için direct mapping:
        # Hook + Subscribe + Story Scene 1 → scene_01.png
        # Story Scene 2 → scene_02.png
        # Story Scene 3 → scene_03.png
        # ...
        # Story Scene 40 → scene_40.png
        for story_scene_num in range(1, scene_count + 1):
            scene_filename = f"scene_{story_scene_num:02d}.png"
            scene_path = scenes_dir / scene_filename

            if scene_path.exists():
                available_scenes.append(story_scene_num)
            else:
                missing_scenes.append(story_scene_num)

        print(f"🖼️  Scene Images Status:")
        if has_hook_subscribe:
            scene_01_exists = (scenes_dir / "scene_01.png").exists()
            print(f"   🎬 Hook + Subscribe + Scene 1: scene_01.png {'✅' if scene_01_exists else '❌'}")
        print(f"   ✅ Available story scenes: {len(available_scenes)}")
        print(f"   ❌ Missing story scenes: {len(missing_scenes)}")

        if missing_scenes:
            print(f"   📝 Missing story scenes: {missing_scenes}")
            print(f"   📝 Missing image files: {[f'scene_{i:02d}.png' for i in missing_scenes]}")

        return available_scenes, None

    def get_youtube_intro_directions(self):
        """YouTube Hook ve Subscribe için özel voice directions"""
        return {
            "hook": {
                "scene_number": 0,
                "title": "Golden Hook - Channel Introduction",
                "direction": "Captivating, mysterious, cinematic buildup. Start with intrigue and wonder. Build anticipation like a movie trailer. Dramatic but engaging pace. Create that 'I must keep watching' feeling.",
                "template": "youtube_hook",
                "style": "cinematic_trailer",
                "speaking_rate": 0.8,
                "pitch": 0.1
            },
            "subscribe": {
                "scene_number": -1,
                "title": "Subscribe Request - Community Building",
                "direction": "Warm, personal, community-focused like MrBeast. Genuine connection with audience. Not salesy but genuinely inviting. Create feeling of joining a special community of dreamers and history lovers. Friendly but passionate.",
                "template": "youtube_subscribe",
                "style": "community_building",
                "speaking_rate": 0.8,
                "pitch": 0.0
            }
        }

    def get_voice_direction_for_scene(self, voice_directions, scene_num):
        """Belirli bir scene için voice direction'ı bul"""
        if not voice_directions:
            return {}

        for direction in voice_directions:
            if direction.get('scene_number') == scene_num:
                return direction

        return {}

    def format_time_ms(self, ms):
        """Millisecond'i MM:SS.mmm formatına çevir"""
        total_seconds = ms / 1000
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int(ms % 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def is_retryable_error(self, error_str: str) -> bool:
        """API error'ının retry edilebilir olup olmadığını kontrol et"""
        retryable_patterns = [
            "503",  # Service unavailable
            "502",  # Bad gateway
            "500",  # Internal server error
            "429",  # Too many requests
            "timeout",
            "connection",
            "network",
            "unavailable",
            "bad gateway",
            "service unavailable"
        ]

        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)

    def apply_scene_content_filter(self, scene_text, scene_num):
        """Apply content filtering for specific problematic scenes"""

        # Scene-specific filters
        if scene_num == 21:
            # Oracle scene filtering
            replacements = {
                "breathes in the mystical fumes": "receives sacred visions",
                "breathing in the sacred vapors": "receiving sacred visions",
                "sacred vapors": "mystical energy",
                "mystical fumes": "divine energy",
                "pupils dilated from": "eyes affected by",
                "consciousness drifting": "awareness expanding"
            }
            for old, new in replacements.items():
                scene_text = scene_text.replace(old, new)

        elif scene_num == 23:
            # Mother-child scene filtering
            replacements = {
                "rocks her youngest child": "holds her youngest child",
                "traces gentle circles on her child's back": "gently comforts her child",
                "baby stirs slightly in her arms": "baby rests peacefully",
                "tiny fist uncurling against Juno's chest": "tiny hand relaxing peacefully",
                "baby's eyelids flutter": "baby grows sleepy",
                "Mother Juno rocks": "Mother Juno holds",
                "chest as trust": "heart as trust"
            }
            for old, new in replacements.items():
                scene_text = scene_text.replace(old, new)

            # Add safety qualifier
            if "historical family scene" not in scene_text.lower():
                scene_text = "Historical family scene in ancient Rome: " + scene_text

        # Universal content safety measures
        general_replacements = {
            "children playing": "young people enjoying activities",
            "intimate": "peaceful",
            "tender": "gentle",
            "embrace": "peaceful moment"
        }

        for old, new in general_replacements.items():
            scene_text = scene_text.replace(old, new)

        return scene_text

    def create_single_audio_chunk(self, scene_text, voice_direction, chunk_name, progress_tracker, tracker=None):
        """Create single audio chunk without retry - for internal use"""
        try:
            client = texttospeech.TextToSpeechClient()
            voice_name = VOICE_NAME
            language_code = LANGUAGE_CODE

            # Process text
            if "Chirp3-HD" in voice_name:
                processed_text = scene_text.replace("[PAUSE]", "... ")
                synthesis_input = texttospeech.SynthesisInput(text=processed_text)
            else:
                processed_text = scene_text.replace("[PAUSE]", '<break time="2s"/>')
                ssml_text = f'<speak>{processed_text}</speak>'
                synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

            # Voice settings
            direction_text = voice_direction.get('direction', '')
            speaking_rate = voice_direction.get('speaking_rate', 0.8)

            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            audio_config_params = {
                "audio_encoding": texttospeech.AudioEncoding.MP3,
                "speaking_rate": speaking_rate,
                "sample_rate_hertz": 44100
            }

            if "Chirp3-HD" not in voice_name:
                audio_config_params["pitch"] = voice_direction.get('pitch', 0.0)

            audio_config = texttospeech.AudioConfig(**audio_config_params)

            # Make request
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Save file
            file_path = progress_tracker.get_chunk_audio_path(chunk_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as out:
                out.write(response.audio_content)

            return True, None, file_path

        except Exception as e:
            return False, str(e), None

    def create_scene_audio_with_robust_retry(self, scene_text, voice_direction, chunk_name, progress_tracker,
                                             tracker=None, max_retries=5):
        """TTS generation with robust retry mechanism AND BUDGET CONTROL"""

        # Extract scene number for content filtering
        scene_num = None
        if chunk_name.startswith("scene_"):
            try:
                scene_num = int(chunk_name.split("_")[1])
            except:
                pass

        # Apply content filtering for problematic scenes
        if scene_num:
            original_length = len(scene_text)
            scene_text = self.apply_scene_content_filter(scene_text, scene_num)
            if len(scene_text) != original_length:
                print(
                    f"      🛡️  Content filter applied to scene {scene_num} ({original_length} → {len(scene_text)} chars)")

        # CHECK BUDGET BEFORE PROCESSING CHUNK
        if tracker:
            can_continue, budget_reason = tracker.check_budget_limits()
            if not can_continue:
                print(f"      🚨 BUDGET LIMIT EXCEEDED before chunk: {budget_reason}")
                return False, f"Budget limit exceeded: {budget_reason}", None

        # Handle overly long scenes (AUTO-SPLIT if >3000 chars)
        if len(scene_text) > 3000:
            print(f"      ⚠️  Scene too long ({len(scene_text)} chars) - AUTO-SPLITTING")

            # Find good split point
            split_point = len(scene_text) // 2
            search_start = max(0, split_point - 200)
            search_end = min(len(scene_text), split_point + 200)
            search_text = scene_text[search_start:search_end]

            sentence_ends = [i for i, char in enumerate(search_text) if char in '.!?']
            if sentence_ends:
                middle_offset = len(search_text) // 2
                best_split = min(sentence_ends, key=lambda x: abs(x - middle_offset))
                actual_split = search_start + best_split + 1
            else:
                space_pos = scene_text.rfind(' ', split_point - 100, split_point + 100)
                actual_split = space_pos if space_pos != -1 else split_point

            part1 = scene_text[:actual_split].strip()
            part2 = scene_text[actual_split:].strip()

            print(f"      ✂️  Split: Part 1 ({len(part1)} chars) + Part 2 ({len(part2)} chars)")

            # Process parts with simple retry
            success1, error1, file1 = None, None, None
            success2, error2, file2 = None, None, None

            for attempt in range(3):  # 3 attempts for each part
                # Budget check before each part
                if tracker:
                    can_continue, budget_reason = tracker.check_budget_limits()
                    if not can_continue:
                        return False, f"Budget limit exceeded during split: {budget_reason}", None

                if not success1:
                    success1, error1, file1 = self.create_single_audio_chunk(
                        part1, voice_direction, f"{chunk_name}_part1", progress_tracker, tracker
                    )
                    if success1:
                        print(f"      ✅ Part 1 completed")
                    else:
                        print(f"      🔄 Part 1 retry {attempt + 1}/3: {error1}")
                        time.sleep(5)

                if not success2:
                    success2, error2, file2 = self.create_single_audio_chunk(
                        part2, voice_direction, f"{chunk_name}_part2", progress_tracker, tracker
                    )
                    if success2:
                        print(f"      ✅ Part 2 completed")
                    else:
                        print(f"      🔄 Part 2 retry {attempt + 1}/3: {error2}")
                        time.sleep(5)

                if success1 and success2:
                    break

            if not (success1 and success2):
                return False, f"Split parts failed: Part1={error1}, Part2={error2}", None

            # Combine parts
            try:
                from pydub import AudioSegment
                audio1 = AudioSegment.from_mp3(file1)
                audio2 = AudioSegment.from_mp3(file2)
                pause = AudioSegment.silent(duration=1000)  # 1 second pause
                combined = audio1 + pause + audio2

                combined_file = progress_tracker.get_chunk_audio_path(chunk_name)
                combined.export(combined_file, format="mp3")

                print(f"      🔗 Combined: {os.path.getsize(combined_file) / 1024:.1f} KB")

                # Mark as completed
                total_chars = len(part1) + len(part2)
                estimated_cost = (total_chars / 1_000_000) * CONFIG.audio_config["cost_per_million_chars"]
                progress_tracker.mark_chunk_completed(chunk_name, total_chars, estimated_cost)

                return True, None, combined_file

            except Exception as combine_error:
                return False, f"Combine failed: {combine_error}", None

        # NORMAL LENGTH PROCESSING (≤3000 chars)
        if len(scene_text) > 2500:
            print(f"      ⚠️  Scene quite long ({len(scene_text)} chars), processing as single chunk")

        # Check if already completed
        if progress_tracker.is_chunk_completed(chunk_name):
            permanent_file = progress_tracker.get_chunk_audio_path(chunk_name)
            if os.path.exists(permanent_file):
                print(f"      ⏭️  Skipping {chunk_name} (already completed)")
                file_size = os.path.getsize(permanent_file) / 1024
                print(f"      ✅ Restored: {file_size:.1f} KB (from progress)")
                return True, None, permanent_file
            else:
                print(f"      ⚠️  Progress shows completed but file missing, regenerating: {chunk_name}")

        # Check if blacklisted
        if progress_tracker.is_chunk_blacklisted(chunk_name):
            print(f"      ⚫ Skipping {chunk_name} (blacklisted after too many failures)")
            return False, "Blacklisted due to repeated failures", None

        # Retry delays: 10s, 20s, 30s, 60s, 120s
        retry_delays = [10, 20, 30, 60, 120]

        for attempt in range(max_retries + 1):  # 0-5 (6 total attempts)
            try:
                # CHECK BUDGET BEFORE EACH ATTEMPT
                if tracker:
                    can_continue, budget_reason = tracker.check_budget_limits()
                    if not can_continue:
                        print(f"      🚨 BUDGET LIMIT EXCEEDED during attempt {attempt + 1}: {budget_reason}")
                        return False, f"Budget limit exceeded: {budget_reason}", None

                # Increment attempt count
                attempt_count = progress_tracker.increment_chunk_attempts(chunk_name)

                print(
                    f"      🎯 {chunk_name} - Attempt {attempt + 1}/{max_retries + 1} (Total attempts: {attempt_count})")

                # Blacklist if too many attempts
                if attempt_count > 8:  # After 8 total attempts across all sessions
                    progress_tracker.blacklist_chunk(chunk_name, f"Exceeded {attempt_count} total attempts")
                    return False, f"Blacklisted after {attempt_count} attempts", None

                client = texttospeech.TextToSpeechClient()

                # User specified Enceladus voice
                voice_name = VOICE_NAME
                language_code = LANGUAGE_CODE

                # Text'i hazırla - Chirp3-HD voices SSML desteklemiyor
                if "Chirp3-HD" in voice_name:
                    processed_text = scene_text.replace("[PAUSE]", "... ")
                    synthesis_input = texttospeech.SynthesisInput(text=processed_text)
                    if attempt == 0:  # Only print once
                        print(f"      ⚠️  Chirp3-HD voice detected: using plain text (no SSML)")
                else:
                    processed_text = scene_text.replace("[PAUSE]", '<break time="2s"/>')
                    ssml_text = f'<speak>{processed_text}</speak>'
                    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

                # Direction'dan ses özelliklerini belirle
                direction_text = voice_direction.get('direction', '')
                speaking_rate = voice_direction.get('speaking_rate', 0.8)

                # Direction'a göre speaking rate ayarla
                if 'speaking_rate' not in voice_direction:
                    if 'slow' in direction_text.lower() or 'meditative' in direction_text.lower():
                        speaking_rate = 0.7
                    elif 'rhythmic' in direction_text.lower() or 'flowing' in direction_text.lower():
                        speaking_rate = 0.85
                    elif 'gentle' in direction_text.lower() or 'tender' in direction_text.lower():
                        speaking_rate = 0.75
                    elif 'alert' in direction_text.lower() or 'business' in direction_text.lower():
                        speaking_rate = 0.9

                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name
                )

                # Audio config - Chirp3-HD doesn't support pitch
                audio_config_params = {
                    "audio_encoding": texttospeech.AudioEncoding.MP3,
                    "speaking_rate": speaking_rate,
                    "sample_rate_hertz": 44100
                }

                if "Chirp3-HD" not in voice_name:
                    audio_config_params["pitch"] = voice_direction.get('pitch', 0.0)
                elif attempt == 0:
                    print(f"      ⚠️  Chirp3-HD voice detected: pitch adjustment disabled")

                audio_config = texttospeech.AudioConfig(**audio_config_params)

                if attempt == 0:  # Only print details on first attempt
                    print(f"      📤 TTS request ({len(scene_text)} chars)")
                    print(f"      🎭 Direction: {direction_text[:50]}...")
                    print(f"      🎙️ Voice: {voice_name}")
                    print(f"      ⚡ Speaking rate: {speaking_rate}")

                # TTS request timing
                start_time = time.time()

                try:
                    response = client.synthesize_speech(
                        input=synthesis_input,
                        voice=voice,
                        audio_config=audio_config
                    )
                    generation_time = time.time() - start_time

                except Exception as tts_error:
                    generation_time = time.time() - start_time
                    print(f"      🚨 TTS API ERROR: {str(tts_error)} ({generation_time:.2f}s)")
                    raise tts_error

                # Save to permanent location immediately
                permanent_file = progress_tracker.get_chunk_audio_path(chunk_name)
                os.makedirs(os.path.dirname(permanent_file), exist_ok=True)

                with open(permanent_file, "wb") as out:
                    out.write(response.audio_content)

                file_size = os.path.getsize(permanent_file) / 1024

                if attempt > 0:
                    print(f"      🎉 Success on attempt {attempt + 1}: {file_size:.1f} KB in {generation_time:.2f}s")
                else:
                    print(f"      ✅ Success: {file_size:.1f} KB in {generation_time:.2f}s")

                # Usage tracking WITH BUDGET UPDATE
                cost = 0
                if tracker:
                    cost = tracker.add_request(len(scene_text), chunk_name, generation_time)

                    # Check budget after cost update
                    can_continue, budget_reason = tracker.check_budget_limits()
                    if not can_continue:
                        print(f"      🚨 BUDGET LIMIT EXCEEDED after TTS request: {budget_reason}")
                        # Note: We still mark as completed since TTS succeeded, but return budget error
                        progress_tracker.mark_chunk_completed(chunk_name, len(scene_text), cost)
                        return False, f"Budget limit exceeded after TTS: {budget_reason}", None
                else:
                    cost = (len(scene_text) / 1_000_000) * CONFIG.audio_config["cost_per_million_chars"]

                # Mark as completed
                progress_tracker.mark_chunk_completed(chunk_name, len(scene_text), cost)

                return True, None, permanent_file

            except Exception as e:
                error_str = str(e)
                print(f"      🐛 ERROR: {error_str}")

                # Mark as failed
                progress_tracker.mark_chunk_failed(chunk_name, error_str)

                if attempt < max_retries and self.is_retryable_error(error_str):
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"      🔄 Retry {attempt + 1}/{max_retries} in {delay}s")
                    time.sleep(delay)
                    continue
                else:
                    print(f"      ❌ Final failure after {attempt + 1} attempts: {error_str}")

                    # Consider blacklisting if too many failures
                    if attempt_count >= 5:
                        progress_tracker.blacklist_chunk(chunk_name, f"Failed {attempt_count} times: {error_str[:100]}")

                    return False, error_str, None

        return False, "Max retries exceeded", None

    def process_scene_based_audio_generation_with_retry(self, stories_data, hook_and_subscribe, voice_directions,
                                                        available_scenes, output_file, story_id, quality="youtube",
                                                        max_retry_rounds=5):
        """Scene-based audio generation with robust retry system AND BUDGET CONTROL"""
        print("🎵 SCENE-BASED AUDIO GENERATION WITH ROBUST RETRY + BUDGET CONTROL")
        print("🎬 YouTube Production Quality + Hook & Subscribe")
        print(f"🎙️ Voice: {VOICE_NAME}")
        print(f"🔄 Retry system: {max_retry_rounds} rounds with exponential backoff")
        print(f"💰 Budget monitoring: Active")
        print("=" * 70)

        # Initialize trackers
        progress_tracker = ProgressTracker(story_id, CONFIG.paths['OUTPUT_DIR'])
        tracker = UsageTracker()

        # Reset story cost for new story
        tracker.reset_story_cost()

        # Show resume info
        resume_info = progress_tracker.get_resume_summary()
        if resume_info["can_resume"]:
            print(f"📂 RESUMING: {resume_info['completed_chunks']} chunks already completed")
            print(f"   💰 Previous cost: ${resume_info['total_cost_so_far']:.4f}")
            print(f"   📝 Previous chars: {resume_info['total_characters_so_far']:,}")
            if resume_info['blacklisted_chunks'] > 0:
                print(f"   ⚫ Blacklisted: {resume_info['blacklisted_chunks']} chunks")

        # Quality settings
        settings = CONFIG.audio_config["quality_profiles"].get(quality,
                                                               CONFIG.audio_config["quality_profiles"]["youtube"])
        print(f"🎛️  Quality: {quality} ({settings['description']}) - {settings['bitrate']}")

        # YouTube intro directions
        intro_directions = self.get_youtube_intro_directions()

        # Determine all chunks to process
        all_chunks = []

        # Add hook if available
        if hook_and_subscribe and hook_and_subscribe.get('hook'):
            all_chunks.append(("hook", hook_and_subscribe['hook'], intro_directions['hook']))

        # Add subscribe if available
        if hook_and_subscribe and hook_and_subscribe.get('subscribe'):
            all_chunks.append(("subscribe", hook_and_subscribe['subscribe'], intro_directions['subscribe']))

        # Add story scenes
        for scene_num in available_scenes:
            scene_key = str(scene_num)
            if scene_key in stories_data:
                scene_text = stories_data[scene_key]
                voice_direction = self.get_voice_direction_for_scene(voice_directions, scene_num)
                all_chunks.append((f"scene_{scene_num}", scene_text, voice_direction))

        print(f"📊 Total chunks to process: {len(all_chunks)}")

        # Process chunks with retry rounds
        for retry_round in range(max_retry_rounds):
            print(f"\n{'🔄 RETRY ROUND ' + str(retry_round) if retry_round > 0 else '🚀 INITIAL ROUND'}")
            print("=" * 50)

            # CHECK BUDGET BEFORE EACH ROUND
            can_continue, budget_reason = tracker.check_budget_limits(story_id)
            if not can_continue:
                print(f"🚨 BUDGET LIMIT EXCEEDED: {budget_reason}")
                return False, f"Budget limit exceeded: {budget_reason}", None

            # Get missing chunks (not completed and not blacklisted)
            missing_chunks = []
            for chunk_name, chunk_text, chunk_direction in all_chunks:
                if not progress_tracker.is_chunk_completed(chunk_name) and not progress_tracker.is_chunk_blacklisted(
                        chunk_name):
                    missing_chunks.append((chunk_name, chunk_text, chunk_direction))

            if not missing_chunks:
                print("✅ All chunks completed!")
                break

            print(f"📝 Missing chunks: {len(missing_chunks)}")
            print(f"💰 Current story cost: ${tracker.story_cost:.4f}")

            # Show blacklisted chunks if any
            blacklisted_count = len(progress_tracker.progress_data.get("blacklisted_chunks", []))
            if blacklisted_count > 0:
                print(f"⚫ Blacklisted chunks: {blacklisted_count}")

            # Wait between retry rounds (except first)
            if retry_round > 0:
                wait_time = 30 + (retry_round * 15)  # 45s, 60s, 75s, 90s
                print(f"⏳ Waiting {wait_time}s before retry round...")
                time.sleep(wait_time)

            # Process missing chunks
            successful_in_round = 0
            failed_in_round = 0

            for i, (chunk_name, chunk_text, chunk_direction) in enumerate(missing_chunks):
                print(f"\n📄 Processing {chunk_name} ({i + 1}/{len(missing_chunks)})")

                # CHECK BUDGET BEFORE EACH CHUNK
                can_continue, budget_reason = tracker.check_budget_limits(story_id)
                if not can_continue:
                    print(f"🚨 BUDGET LIMIT EXCEEDED during processing: {budget_reason}")
                    return False, f"Budget limit exceeded: {budget_reason}", None

                success, error, file_path = self.create_scene_audio_with_robust_retry(
                    chunk_text, chunk_direction, chunk_name, progress_tracker, tracker
                )

                if success:
                    successful_in_round += 1
                    print(f"      ✅ {chunk_name} completed and saved to audio_parts/")
                else:
                    failed_in_round += 1
                    print(f"      ❌ {chunk_name} failed: {error}")

                    # Check if failure was budget-related
                    if "BUDGET" in str(error).upper():
                        return False, f"Budget limit exceeded: {error}", None

                # Rate limiting between chunks
                if i < len(missing_chunks) - 1:
                    base_wait = 3 if retry_round == 0 else 5 + retry_round
                    time.sleep(base_wait)

            print(f"\n📊 Round {retry_round + 1} Results:")
            print(f"   ✅ Successful: {successful_in_round}")
            print(f"   ❌ Failed: {failed_in_round}")
            print(f"   💰 Story cost: ${tracker.story_cost:.4f}")

            # Print progress summary
            if successful_in_round > 0:
                completed_total = len(progress_tracker.progress_data.get("completed_chunks", []))
                tracker.print_progress_summary(completed_total, len(all_chunks))

        # Final budget check
        can_continue, budget_reason = tracker.check_budget_limits(story_id)
        if not can_continue:
            print(f"🚨 FINAL BUDGET CHECK FAILED: {budget_reason}")
            return False, f"Budget limit exceeded: {budget_reason}", None

        # Final check and combine
        print(f"\n🔍 FINAL COMBINATION CHECK:")
        print("=" * 50)

        final_missing = []
        completed_chunks = []

        for chunk_name, chunk_text, chunk_direction in all_chunks:
            if progress_tracker.is_chunk_completed(chunk_name):
                file_path = progress_tracker.get_chunk_audio_path(chunk_name)
                if os.path.exists(file_path):
                    completed_chunks.append((chunk_name, file_path, chunk_direction))
                    print(f"   ✅ {chunk_name}: {os.path.getsize(file_path) / 1024:.1f} KB")
                else:
                    final_missing.append(chunk_name)
                    print(f"   ❌ {chunk_name}: marked complete but file missing")
            else:
                final_missing.append(chunk_name)
                print(f"   ⏳ {chunk_name}: not completed")

        if final_missing:
            blacklisted_chunks = progress_tracker.progress_data.get("blacklisted_chunks", [])
            missing_not_blacklisted = [c for c in final_missing if c not in blacklisted_chunks]

            if missing_not_blacklisted:
                print(f"\n❌ Still missing after {max_retry_rounds} rounds: {missing_not_blacklisted}")
                return False, f"Missing chunks: {missing_not_blacklisted}", None
            else:
                print(f"\n⚠️  Some chunks blacklisted but continuing with {len(completed_chunks)} completed chunks")

        if not completed_chunks:
            return False, "No chunks were successfully generated", None

        # Combine completed chunks
        print(f"\n🔗 Combining {len(completed_chunks)} audio chunks...")
        print(f"💰 Final story cost: ${tracker.story_cost:.4f}")

        try:
            combined = AudioSegment.empty()
            current_time_ms = 0
            timeline_data = {
                "total_scenes": len(completed_chunks),
                "pause_between_scenes_ms": 2000,
                "scenes": [],
                "total_duration_ms": 0,
                "total_duration_formatted": "",
                "created_at": datetime.now().isoformat(),
                "youtube_optimized": True,
                "voice_used": VOICE_NAME,
                "story_id": story_id,
                "retry_rounds_used": max_retry_rounds,
                "final_missing_count": len(final_missing),
                "blacklisted_count": len(progress_tracker.progress_data.get("blacklisted_chunks", [])),
                "budget_controls_used": True,
                "story_cost_usd": tracker.story_cost
            }

            for i, (chunk_name, file_path, voice_direction) in enumerate(completed_chunks):
                print(f"   🔗 Adding {chunk_name}")

                audio = AudioSegment.from_mp3(file_path)
                audio_duration_ms = len(audio)

                # Timeline data
                chunk_start_ms = current_time_ms
                chunk_end_ms = current_time_ms + audio_duration_ms

                # Determine chunk type
                if chunk_name == "hook":
                    chunk_type = "youtube_hook"
                    scene_num = 0
                    image_file = "scene_01.png"
                elif chunk_name == "subscribe":
                    chunk_type = "youtube_subscribe"
                    scene_num = -1
                    image_file = "scene_01.png"
                else:
                    chunk_type = "story_scene"
                    scene_num = int(chunk_name.split('_')[1])
                    image_file = f"scene_{scene_num:02d}.png"

                chunk_timeline = {
                    "type": chunk_type,
                    "scene_number": scene_num,
                    "title": voice_direction.get('title', chunk_name),
                    "direction": voice_direction.get('direction', ''),
                    "start_time_ms": chunk_start_ms,
                    "end_time_ms": chunk_end_ms,
                    "duration_ms": audio_duration_ms,
                    "start_time_formatted": self.format_time_ms(chunk_start_ms),
                    "end_time_formatted": self.format_time_ms(chunk_end_ms),
                    "duration_formatted": self.format_time_ms(audio_duration_ms),
                    "audio_file": f"{chunk_name}_audio.mp3",
                    "image_file": image_file
                }

                timeline_data["scenes"].append(chunk_timeline)

                # Add audio
                combined += audio
                current_time_ms += audio_duration_ms

                # Add pause (except last)
                if i < len(completed_chunks) - 1:
                    pause = AudioSegment.silent(duration=2000)
                    combined += pause
                    current_time_ms += 2000

            # Finalize timeline
            timeline_data["total_duration_ms"] = current_time_ms
            timeline_data["total_duration_formatted"] = self.format_time_ms(current_time_ms)

            # Add usage summary WITH BUDGET INFO
            usage_summary = tracker.print_final_summary()
            usage_summary["budget_controls"] = CONFIG.audio_config.get("budget_controls", {})
            usage_summary["story_cost_usd"] = tracker.story_cost
            timeline_data["usage_summary"] = usage_summary

            # Export final audio
            print(f"\n💾 Exporting final YouTube audio...")
            combined.export(
                output_file,
                format="mp3",
                bitrate=settings['bitrate'],
                parameters=[
                    "-ac", "2",  # Stereo
                    "-ar", str(settings['sample_rate']),  # Sample rate
                    "-q:a", "0"  # Highest quality
                ]
            )

            # Save timeline
            timeline_file = output_file.replace('.mp3', '_timeline.json')
            with open(timeline_file, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, indent=2, ensure_ascii=False)

            # Stats
            duration_min = len(combined) / 60000
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

            print(f"\n🎉 SUCCESS!")
            print(f"   📁 Audio: {output_file}")
            print(f"   📋 Timeline: {timeline_file}")
            print(f"   ⏱️  Duration: {duration_min:.1f} minutes")
            print(f"   📦 Size: {file_size_mb:.1f} MB")
            print(f"   🎭 Chunks used: {len(completed_chunks)}/{len(all_chunks)}")
            print(f"   💰 Final cost: ${tracker.story_cost:.4f}")

            # Cleanup progress on success
            progress_tracker.cleanup_on_success()

            return True, None, timeline_data

        except Exception as e:
            return False, f"Audio combining failed: {e}", None

    def create_audio_summary(self, story_id: int, topic: str, success: bool, scenes_processed: int = 0,
                             error: str = None, usage_data: dict = None):
        """Audio oluşturma özeti - Usage tracking ile"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "story_id": story_id,
            "topic": topic,
            "audio_generation_success": success,
            "scenes_processed": scenes_processed,
            "voice_used": VOICE_NAME,
            "error": error,
            "approach": "robust_retry_system_with_progress_tracking",
            "quality": "youtube_production",
            "retry_system": "enabled_with_blacklisting",
            "server_mode": True
        }

        # Usage data ekle
        if usage_data:
            summary["usage_data"] = usage_data

        output_dir = Path(self.current_output_dir)
        summary_path = output_dir / "audio_summary.json"

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"✅ Audio summary saved: {summary_path}")
        except Exception as e:
            print(f"⚠️  Summary save warning: {e}")

        return summary

    def run_audio_generation(self) -> bool:
        """Run AUDIO generation process for server environment"""
        print("🚀" * 50)
        print("SERVER TTS AUDIO GENERATOR v1.0")
        print("🔗 Database integrated")
        print("🎵 YouTube Production Audio Generation")
        print("🎭 Hook + Subscribe + Voice Directions")
        print("💰 Budget controls enabled")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # Print budget limits
        budget = CONFIG.audio_config.get("budget_controls", {})
        print(f"💰 BUDGET LIMITS:")
        print(f"   📊 Max per story: ${budget.get('max_cost_per_story_usd', 5.0)}")
        print(f"   📊 Max per session: ${budget.get('max_cost_per_session_usd', 25.0)}")
        print(f"   🚨 Emergency stop: ${budget.get('emergency_stop_cost_usd', 50.0)}")
        print(f"   ⚠️  Warning threshold: ${budget.get('warn_threshold_usd', 2.5)}")

        # Initialize success tracking
        overall_success = False

        # Step 1: Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🆔 Topic ID: {project_info['topic_id']}")

        try:
            story_id = self.current_topic_id
            topic = self.current_topic

            # Step 2: Load stories, hook/subscribe and voice directions
            stories_data, hook_and_subscribe, voice_directions, load_error = self.load_stories_and_directions(story_id)

            if load_error:
                print(f"❌ {load_error}")
                self.create_audio_summary(story_id, topic, False, 0, load_error, None)
                return False

            scene_count = len(stories_data)
            has_hook = bool(hook_and_subscribe and hook_and_subscribe.get('hook'))
            has_subscribe = bool(hook_and_subscribe and hook_and_subscribe.get('subscribe'))

            print(f"📊 Total scenes in stories: {scene_count}")
            print(f"🎬 YouTube Hook: {'✅' if has_hook else '❌'}")
            print(f"📢 Subscribe Request: {'✅' if has_subscribe else '❌'}")
            print(f"🎭 Voice directions loaded: {'Yes' if voice_directions else 'No'}")

            # Step 3: Check scene images
            available_scenes, image_error = self.check_scene_images(story_id, scene_count, has_hook or has_subscribe)

            if image_error:
                print(f"❌ {image_error}")
                self.create_audio_summary(story_id, topic, False, 0, image_error, None)
                return False

            if not available_scenes:
                error_msg = "Scene image'ları bulunamadı. Audio oluşturma atlanıyor."
                print(f"❌ {error_msg}")
                self.create_audio_summary(story_id, topic, False, 0, error_msg, None)
                return False

            # Step 4: Generate audio with robust retry AND BUDGET CONTROL
            audio_output = Path(self.current_output_dir) / "story_audio_youtube.mp3"

            start_time = time.time()

            print(f"🔧 Starting robust retry audio generation for story_id={story_id}")
            print(f"🔧 Audio output path: {audio_output}")
            print(f"💰 Budget monitoring active")

            success, error, timeline_data = self.process_scene_based_audio_generation_with_retry(
                stories_data=stories_data,
                hook_and_subscribe=hook_and_subscribe,
                voice_directions=voice_directions,
                available_scenes=available_scenes,
                output_file=str(audio_output),
                story_id=story_id,
                quality="youtube",
                max_retry_rounds=5
            )

            end_time = time.time()
            processing_time = int(end_time - start_time)

            if success:
                print(f"✅ Robust retry audio generation başarılı!")
                print(f"⚡ İşlem süresi: {processing_time // 60}m {processing_time % 60}s")

                if timeline_data:
                    duration_seconds = timeline_data['total_duration_ms'] / 1000
                    chunks_count = timeline_data['total_scenes']
                    cost_usd = timeline_data.get('usage_summary', {}).get('total_cost', 0.0)

                    print(f"💰 Final cost for this story: ${cost_usd:.4f}")

                    # Step 5: Update database with cost tracking
                    self.db_manager.mark_audio_generation_completed(
                        self.current_topic_id, chunks_count, duration_seconds, cost_usd
                    )

                    print(f"📋 Timeline: {chunks_count} total chunks")
                    print(f"⏱️  Total duration: {timeline_data['total_duration_formatted']}")
                    print(f"💾 Database updated with cost: ${cost_usd:.4f}")

                # Create summary
                usage_data = timeline_data.get('usage_summary') if timeline_data else None
                self.create_audio_summary(story_id, topic, True, len(available_scenes), None, usage_data)

                print("\n" + "🎉" * 50)
                print("AUDIO GENERATION SUCCESSFUL!")
                print("✅ YouTube-optimized audio with Hook & Subscribe")
                print("✅ Voice directions applied")
                print("✅ Robust retry system completed")
                print("✅ Budget controls respected")
                print("✅ Database updated with cost tracking")
                print("🎉" * 50)
                overall_success = True

            else:
                print(f"❌ Robust retry audio generation başarısız: {error}")

                # Check if failure was due to budget
                if "BUDGET" in str(error).upper() or "COST" in str(error).upper():
                    print(f"💰 Failure due to budget limits: {error}")
                    error = f"Budget limit exceeded: {error}"

                self.db_manager.mark_audio_generation_failed(
                    self.current_topic_id, error
                )

                self.create_audio_summary(story_id, topic, False, len(available_scenes), error, None)
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ Audio generation failed: {e}", "ERROR")
            self.db_manager.mark_audio_generation_failed(
                self.current_topic_id, str(e)
            )
            import traceback
            traceback.print_exc()
            return False


def run_autonomous_mode():
    """Run autonomous mode - continuously process completed scene topics for audio generation"""
    print("🤖 AUTONOMOUS AUDIO GENERATION MODE STARTED")
    print("🔄 Will process all completed scene topics continuously")
    print("⏹️ Press Ctrl+C to stop gracefully")

    # Initialize database manager for pipeline status
    db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
    db_manager = DatabaseAudioManager(str(db_path))

    # Setup graceful shutdown
    running = True
    processed_count = 0
    start_time = time.time()

    def signal_handler(signum, frame):
        nonlocal running
        print(f"\n⏹️ Received shutdown signal ({signum})")
        print("🔄 Finishing current audio generation and shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            # Check pipeline status
            status = db_manager.get_pipeline_status()

            if status['audio_generation_queue'] > 0:
                print(f"\n🎵 Found {status['audio_generation_queue']} completed scene topics ready for audio generation")

                # Initialize generator
                generator = ServerTTSGenerator()

                # Process one topic
                success = generator.run_audio_generation()

                if success:
                    processed_count += 1
                    print(f"\n✅ Audio generation completed!")
                    print(f"📊 Progress: {processed_count} topics processed")
                else:
                    print(f"\n⚠️ Audio generation failed or no projects ready")

                # Short pause between topics
                if running:
                    time.sleep(5)

            else:
                # No topics ready, wait
                print("😴 No completed scene topics ready for audio generation. Waiting 60s...")
                for i in range(60):
                    if not running:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️ Keyboard interrupt received")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("⏳ Waiting 30 seconds before retry...")
            time.sleep(30)

    # Shutdown summary
    runtime = time.time() - start_time
    print(f"\n🏁 AUTONOMOUS AUDIO GENERATION SHUTDOWN")
    print(f"⏱️ Total runtime: {runtime / 3600:.1f} hours")
    print(f"✅ Topics processed: {processed_count}")
    print("👋 Goodbye!")


if __name__ == "__main__":
    # Check for autonomous mode
    if len(sys.argv) > 1 and sys.argv[1] == '--autonomous':
        run_autonomous_mode()
    else:
        # Original single topic mode
        try:
            print("🚀 SERVER TTS AUDIO GENERATOR v1.0")
            print("🔗 Database integration with robust retry")
            print("🎵 YouTube Production Audio Generation")
            print("🎭 Hook + Subscribe + Voice Directions + Enceladus Voice")
            print("🛡️ ROBUST RETRY SYSTEM - Scene Generator Inspired")
            print("💾 IMMEDIATE SAVE - Each chunk saved immediately")
            print("⚫ BLACKLIST SYSTEM - Auto-blacklist failed chunks")
            print("🔄 RESUME SYSTEM - Continue from any interruption")
            print("🖥️ Production-ready automation")
            print("=" * 60)

            generator = ServerTTSGenerator()
            success = generator.run_audio_generation()

            if success:
                print("🎊 Audio generation completed successfully!")
                print("📁 Audio saved: story_audio_youtube.mp3")
                print("📋 Timeline saved: story_audio_youtube_timeline.json")
                print("🎬 YouTube Hook & Subscribe included")
                print("🎭 Voice directions applied")
                print("💾 Audio parts preserved in audio_parts/")
                print("🔄 Resume capability enabled")
                print("💰 Real-time cost tracking")
                print("🛡️ ROBUST API error protection")
            else:
                print("⚠️ Audio generation failed or no projects ready")

        except KeyboardInterrupt:
            print("\n⏹️ Audio generation stopped by user")
            print("🛡️ Progress saved! Restart to resume from last completed chunk.")
            print("💾 All completed chunks saved in audio_parts/ directory")
        except Exception as e:
            print(f"💥 Audio generation failed: {e}")
            print("🛡️ Progress saved! Check audio_progress.json for resume info.")
            print("🔄 Robust retry system will attempt recovery on restart.")
            print("💾 Completed chunks preserved in audio_parts/ directory")
            CONFIG.logger.error(f"Audio generation failed: {e}")
            import traceback

            traceback.print_exc()