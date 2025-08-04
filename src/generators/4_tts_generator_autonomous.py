"""
Sleepy Dull Stories - ENHANCED TTS Audio Generator with File Existence Check
SKIP existing audio files + Enhanced Progress Tracking + Resume from any point
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
            "production_ready": True,
            "skip_existing_files": True,  # NEW: Skip existing audio files
            "check_file_validity": True,   # NEW: Validate existing files
            "min_file_size_kb": 5         # NEW: Minimum valid file size
        }

        print("✅ TTS audio configuration loaded")
        print(f"⏭️  Skip existing files: {self.audio_config['skip_existing_files']}")
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
            ('audio_cost_usd', 'REAL DEFAULT 0.0'),
            ('audio_chunks_skipped', 'INTEGER DEFAULT 0'),    # NEW: Count of skipped chunks
            ('audio_existing_files_count', 'INTEGER DEFAULT 0')  # NEW: Count of existing files
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
                                        duration_seconds: float, cost_usd: float,
                                        chunks_skipped: int = 0, existing_files_count: int = 0):
        """Mark audio generation as completed with skip statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET audio_generation_status = 'completed',
                audio_generation_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                audio_chunks_generated = ?,
                audio_duration_seconds = ?,
                audio_cost_usd = ?,
                audio_chunks_skipped = ?,
                audio_existing_files_count = ?
            WHERE id = ?
        ''', (chunks_count, duration_seconds, cost_usd, chunks_skipped, existing_files_count, topic_id))

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
            ('audio_cost_usd', 'REAL DEFAULT 0.0'),
            ('audio_chunks_skipped', 'INTEGER DEFAULT 0'),
            ('audio_existing_files_count', 'INTEGER DEFAULT 0')
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


class EnhancedProgressTracker:
    """Enhanced progress tracking with audio file existence checking"""

    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.audio_parts_dir = os.path.join(self.output_dir, "audio_parts")
        self.progress_file = os.path.join(self.output_dir, "audio_progress.json")

        # Ensure directories exist
        os.makedirs(self.audio_parts_dir, exist_ok=True)

        # Load existing progress
        self.progress_data = self.load_progress()

        # NEW: Track skipped files
        self.skipped_chunks = []
        self.existing_files_found = []

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
            "blacklisted_chunks": [],
            "chunk_attempt_count": {},
            "skipped_chunks": [],      # NEW: Track skipped chunks
            "existing_files": [],      # NEW: Track existing files found
            "total_cost_so_far": 0.0,
            "total_characters_so_far": 0,
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat()
        }

    def save_progress(self):
        """Progress'i kaydet"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        # Update skipped chunks info
        self.progress_data["skipped_chunks"] = list(set(self.progress_data.get("skipped_chunks", []) + self.skipped_chunks))
        self.progress_data["existing_files"] = list(set(self.progress_data.get("existing_files", []) + self.existing_files_found))

        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Progress save warning: {e}")

    def is_audio_file_valid(self, file_path: str) -> bool:
        """Check if audio file exists and is valid"""
        if not os.path.exists(file_path):
            return False

        # Check minimum file size
        try:
            file_size_kb = os.path.getsize(file_path) / 1024
            min_size = CONFIG.audio_config.get("min_file_size_kb", 5)

            if file_size_kb < min_size:
                print(f"      ⚠️  File too small: {file_size_kb:.1f}KB < {min_size}KB")
                return False

            # Try to load with pydub to verify it's a valid audio file
            if CONFIG.audio_config.get("check_file_validity", True):
                try:
                    audio = AudioSegment.from_mp3(file_path)
                    duration_ms = len(audio)
                    if duration_ms < 100:  # Less than 100ms is suspicious
                        print(f"      ⚠️  Audio too short: {duration_ms}ms")
                        return False
                    return True
                except Exception as e:
                    print(f"      ⚠️  Invalid audio file: {e}")
                    return False
            else:
                # Just check file size
                return True

        except Exception as e:
            print(f"      ⚠️  File validation error: {e}")
            return False

    def check_existing_audio_file(self, chunk_name: str) -> Tuple[bool, str]:
        """Check if audio file already exists and is valid"""
        file_path = self.get_chunk_audio_path(chunk_name)

        if self.is_audio_file_valid(file_path):
            file_size = os.path.getsize(file_path) / 1024
            return True, f"Valid audio file ({file_size:.1f}KB)"
        else:
            return False, "File missing or invalid"

    def is_chunk_completed(self, chunk_name: str) -> bool:
        """Enhanced: Check both progress and actual file existence"""
        # First check progress data
        if chunk_name in self.progress_data.get("completed_chunks", []):
            # Verify file still exists and is valid
            file_valid, reason = self.check_existing_audio_file(chunk_name)
            if file_valid:
                return True
            else:
                print(f"      ⚠️  {chunk_name}: Marked complete but {reason.lower()}, regenerating")
                # Remove from completed list since file is invalid
                if chunk_name in self.progress_data["completed_chunks"]:
                    self.progress_data["completed_chunks"].remove(chunk_name)
                return False

        # Check if file exists even if not marked as completed (resume scenario)
        file_valid, reason = self.check_existing_audio_file(chunk_name)
        if file_valid:
            print(f"      📁 {chunk_name}: Found existing valid file, marking as completed")
            # Add to completed list
            if chunk_name not in self.progress_data["completed_chunks"]:
                self.progress_data["completed_chunks"].append(chunk_name)
                self.existing_files_found.append(chunk_name)
            self.save_progress()
            return True

        return False

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

    def mark_chunk_skipped(self, chunk_name: str, reason: str):
        """Mark chunk as skipped with reason"""
        if chunk_name not in self.skipped_chunks:
            self.skipped_chunks.append(chunk_name)
            print(f"      ⏭️  Skipped: {chunk_name} - {reason}")
            self.save_progress()

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
        """Enhanced resume summary with skip statistics"""
        completed = len(self.progress_data.get("completed_chunks", []))
        blacklisted = len(self.progress_data.get("blacklisted_chunks", []))
        skipped = len(self.progress_data.get("skipped_chunks", []))
        existing = len(self.progress_data.get("existing_files", []))
        cost = self.progress_data.get("total_cost_so_far", 0.0)
        chars = self.progress_data.get("total_characters_so_far", 0)

        return {
            "completed_chunks": completed,
            "blacklisted_chunks": blacklisted,
            "skipped_chunks": skipped,
            "existing_files_found": existing,
            "total_cost_so_far": cost,
            "total_characters_so_far": chars,
            "can_resume": completed > 0 or existing > 0
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
    """Enhanced server-ready TTS generator with file existence checking"""

    def __init__(self):
        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseAudioManager(str(db_path))

        print("🚀 Enhanced Server TTS Generator v2.0 Initialized")
        print(f"🎙️ Voice: {VOICE_NAME}")
        print(f"⏭️  Skip existing files: {CONFIG.audio_config['skip_existing_files']}")

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

    def check_final_audio_file(self) -> Tuple[bool, str]:
        """Check if final audio file already exists and is valid"""
        final_audio_path = Path(self.current_output_dir) / "story_audio_youtube.mp3"
        timeline_path = Path(self.current_output_dir) / "story_audio_youtube_timeline.json"

        if not final_audio_path.exists():
            return False, "Final audio file does not exist"

        # Check file size
        try:
            file_size_mb = final_audio_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 1:  # Less than 1MB is suspicious for a full story
                return False, f"Final audio file too small ({file_size_mb:.1f}MB)"

            # Check if timeline exists
            if not timeline_path.exists():
                return False, "Timeline file missing"

            # Try to validate audio file
            if CONFIG.audio_config.get("check_file_validity", True):
                try:
                    audio = AudioSegment.from_mp3(str(final_audio_path))
                    duration_minutes = len(audio) / 60000
                    if duration_minutes < 5:  # Less than 5 minutes is suspicious
                        return False, f"Final audio too short ({duration_minutes:.1f} minutes)"
                    return True, f"Valid final audio file ({file_size_mb:.1f}MB, {duration_minutes:.1f} minutes)"
                except Exception as e:
                    return False, f"Invalid audio file: {e}"
            else:
                return True, f"Final audio file exists ({file_size_mb:.1f}MB)"

        except Exception as e:
            return False, f"Error checking final file: {e}"

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
        """Enhanced YouTube Hook ve Subscribe için özel voice directions"""
        return {
            "hook": {
                "scene_number": 0,
                "title": "Golden Hook - Channel Introduction",
                "direction": "DYNAMIC, mysterious, cinematic trailer style. Start with ENERGY and intrigue. Build dramatic tension. Create 'I MUST keep watching' feeling. Fast-paced storytelling with strategic pauses.",
                "speaking_rate": 1.0,  # ✅ Normal hız (hızlı başlangıç)
                "pitch": 0.3  # ✅ Daha yüksek enerji
            },
            "subscribe": {
                "scene_number": -1,
                "title": "Subscribe Request - Community Building",
                "direction": "Energetic but warming transition. MrBeast-style genuine excitement. Create FOMO for the community. Passionate but welcoming.",
                "speaking_rate": 0.9,  # ✅ Hâlâ canlı ama yavaşlıyor
                "pitch": 0.1  # ✅ Orta enerji
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
            speaking_rate = voice_direction.get('speaking_rate', 0.9)

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
        """Enhanced TTS generation with file existence checking and robust retry"""

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

        # ENHANCED: Check if chunk is already completed with valid file
        if progress_tracker.is_chunk_completed(chunk_name):
            permanent_file = progress_tracker.get_chunk_audio_path(chunk_name)
            if os.path.exists(permanent_file):
                file_size = os.path.getsize(permanent_file) / 1024
                print(f"      ⏭️  Skipping {chunk_name} (already completed, {file_size:.1f}KB)")
                progress_tracker.mark_chunk_skipped(chunk_name, f"Already completed ({file_size:.1f}KB)")
                return True, None, permanent_file

        # Check if blacklisted
        if progress_tracker.is_chunk_blacklisted(chunk_name):
            print(f"      ⚫ Skipping {chunk_name} (blacklisted after too many failures)")
            return False, "Blacklisted due to repeated failures", None

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
                speaking_rate = voice_direction.get('speaking_rate', 0.9)

                # Direction'a göre speaking rate ayarla
                if 'speaking_rate' not in voice_direction:
                    if 'slow' in direction_text.lower() or 'meditative' in direction_text.lower():
                        speaking_rate = 0.83
                    elif 'rhythmic' in direction_text.lower() or 'flowing' in direction_text.lower():
                        speaking_rate = 0.95
                    elif 'gentle' in direction_text.lower() or 'tender' in direction_text.lower():
                        speaking_rate = 0.9
                    elif 'alert' in direction_text.lower() or 'business' in direction_text.lower():
                        speaking_rate = 1.0

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

    def process_scene_based_audio_generation_with_skip_existing(self, stories_data, hook_and_subscribe, voice_directions,
                                                                available_scenes, output_file, story_id, quality="youtube",
                                                                max_retry_rounds=5):
        """Enhanced scene-based audio generation with existing file skipping"""
        print("🎵 ENHANCED SCENE-BASED AUDIO GENERATION")
        print("⏭️  SKIP EXISTING FILES + Smart Resume + Budget Control")
        print("🎬 YouTube Production Quality + Hook & Subscribe")
        print(f"🎙️ Voice: {VOICE_NAME}")
        print(f"🔄 Retry system: {max_retry_rounds} rounds with exponential backoff")
        print(f"💰 Budget monitoring: Active")
        print("=" * 70)

        # Initialize enhanced trackers
        progress_tracker = EnhancedProgressTracker(story_id, CONFIG.paths['OUTPUT_DIR'])
        tracker = UsageTracker()

        # Reset story cost for new story
        tracker.reset_story_cost()

        # Show enhanced resume info
        resume_info = progress_tracker.get_resume_summary()
        if resume_info["can_resume"]:
            print(f"📂 ENHANCED RESUMING:")
            print(f"   ✅ Completed chunks: {resume_info['completed_chunks']}")
            print(f"   📁 Existing files found: {resume_info['existing_files_found']}")
            print(f"   ⏭️  Chunks skipped: {resume_info['skipped_chunks']}")
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

        # ENHANCED: Pre-scan for existing files
        print(f"\n🔍 PRE-SCANNING FOR EXISTING AUDIO FILES:")
        print("=" * 50)

        existing_count = 0
        valid_existing = []
        invalid_existing = []

        for chunk_name, chunk_text, chunk_direction in all_chunks:
            file_exists, status = progress_tracker.check_existing_audio_file(chunk_name)
            if file_exists:
                existing_count += 1
                valid_existing.append(chunk_name)
                file_size = os.path.getsize(progress_tracker.get_chunk_audio_path(chunk_name)) / 1024
                print(f"   ✅ {chunk_name}: {status}")
                # Mark as completed if not already marked
                if not progress_tracker.is_chunk_completed(chunk_name):
                    progress_tracker.mark_chunk_completed(chunk_name, len(chunk_text), 0.0)  # No cost for existing
            else:
                print(f"   ❌ {chunk_name}: {status}")
                invalid_existing.append(chunk_name)

        print(f"\n📊 PRE-SCAN RESULTS:")
        print(f"   ✅ Valid existing files: {len(valid_existing)}")
        print(f"   ❌ Missing/invalid files: {len(invalid_existing)}")
        print(f"   💰 Cost savings from existing files: Estimated ${len(valid_existing) * 0.1:.2f}")

        if len(valid_existing) == len(all_chunks):
            print(f"\n🎉 ALL AUDIO FILES ALREADY EXIST AND ARE VALID!")
            print(f"   ⏭️  Skipping to final audio combination...")
            # Jump directly to combination
            completed_chunks = []
            for chunk_name, chunk_text, chunk_direction in all_chunks:
                file_path = progress_tracker.get_chunk_audio_path(chunk_name)
                completed_chunks.append((chunk_name, file_path, chunk_direction))
                progress_tracker.mark_chunk_skipped(chunk_name, "Pre-existing valid file")

            # Skip to combination
            return self.combine_audio_chunks(completed_chunks, output_file, story_id, settings, progress_tracker, tracker)

        # Process chunks with retry rounds (only for missing/invalid chunks)
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
            print(f"⏭️  Already completed/skipped: {len(all_chunks) - len(missing_chunks)}")
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
            skipped_in_round = 0

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
                    if "skipping" in str(error).lower():
                        skipped_in_round += 1
                        print(f"      ⏭️  {chunk_name} skipped (existing file found)")
                    else:
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
            print(f"   ✅ Generated: {successful_in_round}")
            print(f"   ⏭️  Skipped: {skipped_in_round}")
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
        total_skipped = len(progress_tracker.progress_data.get("skipped_chunks", []))

        for chunk_name, chunk_text, chunk_direction in all_chunks:
            if progress_tracker.is_chunk_completed(chunk_name):
                file_path = progress_tracker.get_chunk_audio_path(chunk_name)
                if os.path.exists(file_path):
                    completed_chunks.append((chunk_name, file_path, chunk_direction))
                    file_size = os.path.getsize(file_path) / 1024
                    was_existing = chunk_name in progress_tracker.progress_data.get("existing_files", [])
                    status = "📁 Pre-existing" if was_existing else "🆕 Generated"
                    print(f"   ✅ {chunk_name}: {file_size:.1f}KB ({status})")
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
        return self.combine_audio_chunks(completed_chunks, output_file, story_id, settings, progress_tracker, tracker)

    def combine_audio_chunks(self, completed_chunks, output_file, story_id, settings, progress_tracker, tracker):
        """Combine audio chunks into final file with enhanced statistics"""

        print(f"\n🔗 Combining {len(completed_chunks)} audio chunks...")
        print(f"💰 Final story cost: ${tracker.story_cost:.4f}")

        # Count pre-existing vs newly generated
        existing_files = progress_tracker.progress_data.get("existing_files", [])
        pre_existing_count = len([chunk for chunk_name, _, _ in completed_chunks if chunk_name in existing_files])
        newly_generated_count = len(completed_chunks) - pre_existing_count

        print(f"📊 Chunk breakdown:")
        print(f"   📁 Pre-existing files: {pre_existing_count}")
        print(f"   🆕 Newly generated: {newly_generated_count}")

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
                "retry_rounds_used": max_retry_rounds if hasattr(self, 'max_retry_rounds') else 5,
                "final_missing_count": 0,
                "blacklisted_count": len(progress_tracker.progress_data.get("blacklisted_chunks", [])),
                "budget_controls_used": True,
                "story_cost_usd": tracker.story_cost,
                "pre_existing_files_count": pre_existing_count,
                "newly_generated_files_count": newly_generated_count,
                "existing_files_used": existing_files,
                "skip_existing_enabled": True
            }

            for i, (chunk_name, file_path, voice_direction) in enumerate(completed_chunks):
                print(f"   🔗 Adding {chunk_name}")

                audio = AudioSegment.from_mp3(file_path)
                audio_duration_ms = len(audio)

                # Timeline data
                chunk_start_ms = current_time_ms
                chunk_end_ms = current_time_ms + audio_duration_ms

                # Determine chunk type and status
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

                # Determine if file was pre-existing or newly generated
                file_status = "pre_existing" if chunk_name in existing_files else "newly_generated"

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
                    "image_file": image_file,
                    "file_status": file_status,  # NEW: Track if pre-existing or generated
                    "file_size_kb": round(os.path.getsize(file_path) / 1024, 1)
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

            # Add enhanced usage summary
            usage_summary = tracker.print_final_summary()
            usage_summary["budget_controls"] = CONFIG.audio_config.get("budget_controls", {})
            usage_summary["story_cost_usd"] = tracker.story_cost
            usage_summary["cost_savings_from_existing"] = pre_existing_count * 0.1  # Estimated savings
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

            # Enhanced stats
            duration_min = len(combined) / 60000
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            total_skipped = len(progress_tracker.progress_data.get("skipped_chunks", []))

            print(f"\n🎉 SUCCESS!")
            print(f"   📁 Audio: {output_file}")
            print(f"   📋 Timeline: {timeline_file}")
            print(f"   ⏱️  Duration: {duration_min:.1f} minutes")
            print(f"   📦 Size: {file_size_mb:.1f} MB")
            print(f"   🎭 Chunks used: {len(completed_chunks)}")
            print(f"   📁 Pre-existing files: {pre_existing_count}")
            print(f"   🆕 Newly generated: {newly_generated_count}")
            print(f"   💰 Final cost: ${tracker.story_cost:.4f}")
            if pre_existing_count > 0:
                print(f"   💸 Estimated savings: ${pre_existing_count * 0.1:.2f}")

            # Cleanup progress on success
            progress_tracker.cleanup_on_success()

            return True, None, timeline_data

        except Exception as e:
            return False, f"Audio combining failed: {e}", None

    def create_audio_summary(self, story_id: int, topic: str, success: bool, scenes_processed: int = 0,
                             error: str = None, usage_data: dict = None, chunks_skipped: int = 0,
                             existing_files_count: int = 0):
        """Enhanced audio summary with skip statistics"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "story_id": story_id,
            "topic": topic,
            "audio_generation_success": success,
            "scenes_processed": scenes_processed,
            "chunks_skipped": chunks_skipped,
            "existing_files_reused": existing_files_count,
            "voice_used": VOICE_NAME,
            "error": error,
            "approach": "enhanced_skip_existing_with_progress_tracking",
            "quality": "youtube_production",
            "retry_system": "enabled_with_blacklisting",
            "skip_existing_files": True,
            "file_validation_enabled": CONFIG.audio_config.get("check_file_validity", True),
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
            print(f"✅ Enhanced audio summary saved: {summary_path}")
        except Exception as e:
            print(f"⚠️  Summary save warning: {e}")

        return summary

    def run_enhanced_audio_generation(self) -> bool:
        """Run ENHANCED AUDIO generation with existing file detection"""
        print("🚀" * 50)
        print("ENHANCED SERVER TTS AUDIO GENERATOR v2.0")
        print("⏭️  SKIP EXISTING FILES + Smart Resume")
        print("🔗 Database integrated")
        print("🎵 YouTube Production Audio Generation")
        print("🎭 Hook + Subscribe + Voice Directions")
        print("💰 Budget controls enabled")
        print("📁 File existence validation enabled")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # Print budget limits
        budget = CONFIG.audio_config.get("budget_controls", {})
        print(f"💰 BUDGET LIMITS:")
        print(f"   📊 Max per story: ${budget.get('max_cost_per_story_usd', 5.0)}")
        print(f"   📊 Max per session: ${budget.get('max_cost_per_session_usd', 25.0)}")
        print(f"   🚨 Emergency stop: ${budget.get('emergency_stop_cost_usd', 50.0)}")
        print(f"   ⚠️  Warning threshold: ${budget.get('warn_threshold_usd', 2.5)}")

        # Print skip settings
        print(f"⏭️  SKIP SETTINGS:")
        print(f"   📁 Skip existing files: {CONFIG.audio_config['skip_existing_files']}")
        print(f"   🔍 Validate existing files: {CONFIG.audio_config['check_file_validity']}")
        print(f"   📏 Min file size: {CONFIG.audio_config['min_file_size_kb']}KB")

        # Initialize success tracking
        overall_success = False

        # Step 1: Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🆔 Topic ID: {project_info['topic_id']}")

        # ENHANCED: Check if final audio already exists
        final_exists, final_status = self.check_final_audio_file()
        if final_exists:
            print(f"\n🎉 FINAL AUDIO ALREADY EXISTS!")
            print(f"   ✅ Status: {final_status}")
            print(f"   ⏭️  Skipping entire audio generation process")

            # Still mark as completed in database
            try:
                # Get some basic info for database
                timeline_path = Path(self.current_output_dir) / "story_audio_youtube_timeline.json"
                chunks_count = 0
                duration_seconds = 0.0

                if timeline_path.exists():
                    with open(timeline_path, 'r', encoding='utf-8') as f:
                        timeline_data = json.load(f)
                        chunks_count = timeline_data.get('total_scenes', 0)
                        duration_seconds = timeline_data.get('total_duration_ms', 0) / 1000

                self.db_manager.mark_audio_generation_completed(
                    self.current_topic_id, chunks_count, duration_seconds, 0.0,
                    chunks_count, chunks_count  # All chunks were pre-existing
                )
                print(f"📋 Database updated: marked as completed (pre-existing final file)")

            except Exception as e:
                print(f"⚠️  Database update warning: {e}")

            return True

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

            # Step 4: Generate audio with enhanced skip existing functionality
            audio_output = Path(self.current_output_dir) / "story_audio_youtube.mp3"

            start_time = time.time()

            print(f"🔧 Starting enhanced audio generation with existing file detection for story_id={story_id}")
            print(f"🔧 Audio output path: {audio_output}")
            print(f"⏭️  Skip existing files: Enabled")
            print(f"💰 Budget monitoring: Active")

            success, error, timeline_data = self.process_scene_based_audio_generation_with_skip_existing(
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
                print(f"✅ Enhanced audio generation successful!")
                print(f"⚡ Processing time: {processing_time // 60}m {processing_time % 60}s")

                if timeline_data:
                    duration_seconds = timeline_data['total_duration_ms'] / 1000
                    chunks_count = timeline_data['total_scenes']
                    cost_usd = timeline_data.get('usage_summary', {}).get('total_cost', 0.0)
                    pre_existing_count = timeline_data.get('pre_existing_files_count', 0)
                    newly_generated_count = timeline_data.get('newly_generated_files_count', 0)

                    print(f"💰 Final cost for this story: ${cost_usd:.4f}")
                    print(f"📁 Files reused: {pre_existing_count}")
                    print(f"🆕 Files generated: {newly_generated_count}")

                    # Step 5: Update database with enhanced statistics
                    self.db_manager.mark_audio_generation_completed(
                        self.current_topic_id, chunks_count, duration_seconds, cost_usd,
                        pre_existing_count, pre_existing_count  # chunks_skipped and existing_files_count
                    )

                    print(f"📋 Timeline: {chunks_count} total chunks")
                    print(f"⏱️  Total duration: {timeline_data['total_duration_formatted']}")
                    print(f"💾 Database updated with enhanced stats")

                # Create enhanced summary
                usage_data = timeline_data.get('usage_summary') if timeline_data else None
                chunks_skipped = timeline_data.get('pre_existing_files_count', 0) if timeline_data else 0
                existing_files_count = chunks_skipped

                self.create_audio_summary(story_id, topic, True, len(available_scenes), None, usage_data,
                                        chunks_skipped, existing_files_count)

                print("\n" + "🎉" * 50)
                print("ENHANCED AUDIO GENERATION SUCCESSFUL!")
                print("✅ YouTube-optimized audio with Hook & Subscribe")
                print("✅ Voice directions applied")
                print("✅ Existing files detected and reused")
                print("✅ Budget controls respected")
                print("✅ Database updated with skip statistics")
                if chunks_skipped > 0:
                    print(f"✅ Cost savings: {chunks_skipped} files reused")
                print("🎉" * 50)
                overall_success = True

            else:
                print(f"❌ Enhanced audio generation failed: {error}")

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
            self.log_step(f"❌ Enhanced audio generation failed: {e}", "ERROR")
            self.db_manager.mark_audio_generation_failed(
                self.current_topic_id, str(e)
            )
            import traceback
            traceback.print_exc()
            return False


def run_autonomous_mode():
    """Run autonomous mode - continuously process completed scene topics for audio generation"""
    print("🤖 ENHANCED AUTONOMOUS AUDIO GENERATION MODE STARTED")
    print("⏭️  Skip existing files + Smart resume enabled")
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

                # Initialize enhanced generator
                generator = ServerTTSGenerator()

                # Process one topic with enhanced features
                success = generator.run_enhanced_audio_generation()

                if success:
                    processed_count += 1
                    print(f"\n✅ Enhanced audio generation completed!")
                    print(f"📊 Progress: {processed_count} topics processed")
                else:
                    print(f"\n⚠️ Enhanced audio generation failed or no projects ready")

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
    print(f"\n🏁 ENHANCED AUTONOMOUS AUDIO GENERATION SHUTDOWN")
    print(f"⏱️ Total runtime: {runtime / 3600:.1f} hours")
    print(f"✅ Topics processed: {processed_count}")
    print("👋 Goodbye!")


if __name__ == "__main__":
    # Check for autonomous mode
    if len(sys.argv) > 1 and sys.argv[1] == '--autonomous':
        run_autonomous_mode()
    else:
        # Enhanced single topic mode
        try:
            print("🚀 ENHANCED SERVER TTS AUDIO GENERATOR v2.0")
            print("⏭️  SKIP EXISTING FILES + Smart Resume")
            print("🔗 Database integration with robust retry")
            print("🎵 YouTube Production Audio Generation")
            print("🎭 Hook + Subscribe + Voice Directions + Enceladus Voice")
            print("🛡️ ROBUST RETRY SYSTEM - Scene Generator Inspired")
            print("💾 IMMEDIATE SAVE - Each chunk saved immediately")
            print("⚫ BLACKLIST SYSTEM - Auto-blacklist failed chunks")
            print("🔄 RESUME SYSTEM - Continue from any interruption")
            print("📁 FILE VALIDATION - Skip existing valid files")
            print("💰 BUDGET CONTROLS - Real-time cost monitoring")
            print("🖥️ Production-ready automation")
            print("=" * 60)

            generator = ServerTTSGenerator()
            success = generator.run_enhanced_audio_generation()

            if success:
                print("🎊 Enhanced audio generation completed successfully!")
                print("📁 Audio saved: story_audio_youtube.mp3")
                print("📋 Timeline saved: story_audio_youtube_timeline.json")
                print("🎬 YouTube Hook & Subscribe included")
                print("🎭 Voice directions applied")
                print("💾 Audio parts preserved in audio_parts/")
                print("🔄 Resume capability enabled")
                print("⏭️  Existing files automatically detected and reused")
                print("💰 Real-time cost tracking with budget controls")
                print("🛡️ ROBUST API error protection")
            else:
                print("⚠️ Enhanced audio generation failed or no projects ready")

        except KeyboardInterrupt:
            print("\n⏹️ Enhanced audio generation stopped by user")
            print("🛡️ Progress saved! Restart to resume from last completed chunk.")
            print("💾 All completed chunks saved in audio_parts/ directory")
            print("⏭️  Existing files will be detected and skipped on restart")
        except Exception as e:
            print(f"💥 Enhanced audio generation failed: {e}")
            print("🛡️ Progress saved! Check audio_progress.json for resume info.")
            print("🔄 Enhanced retry system will attempt recovery on restart.")
            print("💾 Completed chunks preserved in audio_parts/ directory")
            print("⏭️  Existing files will be automatically detected")
            CONFIG.logger.error(f"Enhanced audio generation failed: {e}")
            import traceback
            traceback.print_exc()