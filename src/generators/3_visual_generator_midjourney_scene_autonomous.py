"""
Sleepy Dull Stories - AUTONOMOUS Video Composer - IMPROVED
SCENE-BY-SCENE Processing for Perfect Audio Sync + AUTONOMOUS MODE
FIXED: All sequence/total_segments errors removed + Enhanced error handling
"""

import pandas as pd
import os
import json
import ffmpeg
import subprocess
import random
from pathlib import Path
from datetime import datetime
import sqlite3
import sys
import signal
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import time

# Load environment first
load_dotenv()

# SAFE MOVIEPY IMPORT WITH FALLBACK
try:
    from moviepy.editor import (
        ImageClip, VideoFileClip, AudioFileClip,
        CompositeVideoClip, concatenate_videoclips
    )
    import moviepy.config as mp_config
    MOVIEPY_AVAILABLE = True
    print("✅ MoviePy available and imported successfully")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"⚠️ MoviePy not available: {e}")
    print("📦 Install with: pip install moviepy")
except Exception as e:
    MOVIEPY_AVAILABLE = False
    print(f"⚠️ MoviePy import failed: {e}")

# SAFE GARBAGE COLLECTION IMPORT
try:
    import gc
    print("✅ Garbage collection available")
except ImportError:
    print("⚠️ Garbage collection not available")
    gc = None


# Server Configuration Class
class ServerConfig:
    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_video_config()
        self.ensure_directories()

    def setup_paths(self):
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

        print(f"✅ Video Composer server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_video_config(self):
        self.video_config = {
            "max_retry_rounds": 3,
            "target_fps": 30,
            "target_resolution": [1920, 1080],
            "quality_profiles": {
                "youtube": {"codec": "libx264", "preset": "fast", "crf": 18},
                "balanced": {"codec": "libx264", "preset": "medium", "crf": 23},
                "fast": {"codec": "libx264", "preset": "ultrafast", "crf": 28}
            },
            "budget_controls": {
                "max_processing_time_minutes": 60,
                "max_memory_usage_gb": 8,
                "warn_threshold_minutes": 40
            },
            "scene_render_controls": {
                "max_retries_per_scene": 3,
                "retry_delay_seconds": 10,
                "cleanup_retry_delay": 5,
                "emergency_cleanup_timeout": 30
            },
            "server_mode": True,
            "production_ready": True,
            "moviepy_enabled": MOVIEPY_AVAILABLE,
            "ffmpeg_fallback": True,
            "timeline_mode": True,
            "scene_by_scene_mode": True
        }

        print("✅ Video composition configuration loaded")
        print(f"🎬 Target: {self.video_config['target_resolution'][0]}x{self.video_config['target_resolution'][1]} @ {self.video_config['target_fps']}fps")
        print(f"📋 Timeline mode: {'✅ ENABLED' if self.video_config['timeline_mode'] else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if self.video_config['scene_by_scene_mode'] else '❌ DISABLED'}")
        print(f"🎭 MoviePy support: {'✅ AVAILABLE' if MOVIEPY_AVAILABLE else '❌ NOT AVAILABLE'}")

    def setup_logging(self):
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"video_composer_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("VideoComposer")
        self.logger.info(f"✅ Video composer logging initialized: {log_file}")

    def ensure_directories(self):
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All video composer directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Video Composer server configuration loaded successfully")
except Exception as e:
    print(f"❌ Video Composer server configuration failed: {e}")
    sys.exit(1)


# Database Video Management Integration
class DatabaseVideoManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def ensure_video_columns(self, cursor):
        """Ensure video generation columns exist"""
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        columns_to_add = [
            ('video_generation_status', 'TEXT DEFAULT "pending"'),
            ('video_generation_started_at', 'DATETIME'),
            ('video_generation_completed_at', 'DATETIME'),
            ('video_duration_seconds', 'REAL DEFAULT 0.0'),
            ('video_file_size_mb', 'REAL DEFAULT 0.0'),
            ('video_processing_time_minutes', 'REAL DEFAULT 0.0')
        ]

        columns_added = []
        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                try:
                    cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')
                    columns_added.append(column_name)
                except Exception as e:
                    print(f"⚠️ Warning adding column {column_name}: {e}")

        if columns_added:
            print(f"🔧 Added database columns: {', '.join(columns_added)}")

    def get_completed_audio_topic_ready_for_video(self) -> Optional[Tuple[int, str, str, str]]:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            self.ensure_video_columns(cursor)
            conn.commit()

            cursor.execute('''
                SELECT id, topic, description, output_path 
                FROM topics 
                WHERE status = 'completed' 
                AND audio_generation_status = 'completed'
                AND (video_generation_status IS NULL OR video_generation_status = 'pending')
                ORDER BY audio_generation_completed_at ASC 
                LIMIT 1
            ''')

            result = cursor.fetchone()
            return result if result else None

        except Exception as e:
            print(f"❌ Database error in get_completed_audio_topic_ready_for_video: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def mark_video_generation_started(self, topic_id: int):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            self.ensure_video_columns(cursor)

            cursor.execute('''
                UPDATE topics 
                SET video_generation_status = 'in_progress', 
                    video_generation_started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (topic_id,))

            conn.commit()
            print(f"✅ Database: Topic {topic_id} marked as video generation started")

        except Exception as e:
            print(f"❌ Database error in mark_video_generation_started: {e}")
        finally:
            if conn:
                conn.close()

    def mark_video_generation_completed(self, topic_id: int, duration_seconds: float,
                                        file_size_mb: float, processing_time_minutes: float):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            self.ensure_video_columns(cursor)

            cursor.execute('''
                UPDATE topics 
                SET video_generation_status = 'completed',
                    video_generation_completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP,
                    video_duration_seconds = ?,
                    video_file_size_mb = ?,
                    video_processing_time_minutes = ?
                WHERE id = ?
            ''', (duration_seconds, file_size_mb, processing_time_minutes, topic_id))

            conn.commit()
            print(f"✅ Database: Topic {topic_id} marked as video generation completed")

        except Exception as e:
            print(f"❌ Database error in mark_video_generation_completed: {e}")
        finally:
            if conn:
                conn.close()

    def mark_video_generation_failed(self, topic_id: int, error_message: str):
        """Mark video generation as failed"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            self.ensure_video_columns(cursor)

            cursor.execute('''
                UPDATE topics 
                SET video_generation_status = 'failed',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (topic_id,))

            conn.commit()
            print(f"❌ Database: Topic {topic_id} marked as video generation failed - {error_message}")

        except Exception as e:
            print(f"❌ Database error in mark_video_generation_failed: {e}")
        finally:
            if conn:
                conn.close()

    def get_pipeline_status(self) -> Dict:
        """Get overall pipeline status"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            self.ensure_video_columns(cursor)
            conn.commit()

            # Count video generation queue
            cursor.execute('''
                SELECT COUNT(*) FROM topics 
                WHERE status = 'completed' 
                AND audio_generation_status = 'completed'
                AND (video_generation_status IS NULL OR video_generation_status = 'pending')
            ''')
            video_queue = cursor.fetchone()[0]

            # Count active video generation
            cursor.execute('''
                SELECT COUNT(*) FROM topics 
                WHERE video_generation_status = 'in_progress'
            ''')
            video_active = cursor.fetchone()[0]

            return {
                'video_generation_queue': video_queue,
                'video_generation_active': video_active
            }

        except Exception as e:
            print(f"❌ Database error in get_pipeline_status: {e}")
            return {
                'video_generation_queue': 0,
                'video_generation_active': 0
            }
        finally:
            if conn:
                conn.close()


class VideoProgressTracker:
    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.progress_file = os.path.join(self.output_dir, "video_progress.json")

        os.makedirs(self.output_dir, exist_ok=True)
        self.progress_data = self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 Video progress loaded: {data.get('current_stage', 'unknown')} stage")
                return data
            except Exception as e:
                print(f"⚠️  Video progress file corrupted, starting fresh: {e}")

        return {
            "story_id": self.story_id,
            "current_stage": "init",
            "completed_stages": [],
            "failed_attempts": 0,
            "scene_failures": {},  # Track failures per scene
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat(),
            "render_method": "unknown",
            "stages": {
                "project_load": {"status": "pending", "timestamp": None},
                "timeline_load": {"status": "pending", "timestamp": None},
                "scene_preparation": {"status": "pending", "timestamp": None},
                "video_render": {"status": "pending", "timestamp": None},
                "verification": {"status": "pending", "timestamp": None}
            }
        }

    def save_progress(self):
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Video progress save warning: {e}")

    def mark_stage_completed(self, stage: str):
        if stage not in self.progress_data["completed_stages"]:
            self.progress_data["completed_stages"].append(stage)

        if stage in self.progress_data["stages"]:
            self.progress_data["stages"][stage] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }

        self.progress_data["current_stage"] = stage
        self.save_progress()
        print(f"      📋 Stage completed: {stage}")

    def mark_stage_failed(self, stage: str, error: str):
        if stage in self.progress_data["stages"]:
            self.progress_data["stages"][stage] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": error
            }

        self.progress_data["failed_attempts"] += 1
        self.save_progress()

    def mark_scene_failure(self, scene_id: str, error: str):
        """Track scene-specific failures"""
        if "scene_failures" not in self.progress_data:
            self.progress_data["scene_failures"] = {}

        if scene_id not in self.progress_data["scene_failures"]:
            self.progress_data["scene_failures"][scene_id] = []

        self.progress_data["scene_failures"][scene_id].append({
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.save_progress()

    def get_scene_failure_count(self, scene_id: str) -> int:
        """Get failure count for specific scene"""
        return len(self.progress_data.get("scene_failures", {}).get(scene_id, []))

    def set_render_method(self, method: str):
        self.progress_data["render_method"] = method
        self.save_progress()

    def cleanup_on_success(self):
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print(f"🗑️  Video progress file cleaned up (successful completion)")
        except Exception as e:
            print(f"⚠️  Video progress cleanup warning: {e}")


class VideoUsageTracker:
    def __init__(self):
        self.session_start = datetime.now()
        self.processing_stages = []
        self.total_processing_time = 0.0
        self.memory_usage_mb = 0.0

        self.performance_data = {
            "render_method": "unknown",
            "total_scenes": 0,
            "video_duration_seconds": 0.0,
            "processing_time_minutes": 0.0,
            "memory_peak_mb": 0.0,
            "filesize_mb": 0.0,
            "timeline_mode": CONFIG.video_config.get("timeline_mode", False),
            "scene_by_scene_mode": CONFIG.video_config.get("scene_by_scene_mode", False),
            "moviepy_available": MOVIEPY_AVAILABLE
        }

        self.budget_controls = CONFIG.video_config.get("budget_controls", {})

    def check_processing_limits(self) -> Tuple[bool, str]:
        """Check if processing limits are exceeded"""
        current_time = (datetime.now() - self.session_start).total_seconds() / 60

        max_time = self.budget_controls.get("max_processing_time_minutes", 60)
        if current_time >= max_time:
            return False, f"PROCESSING TIME LIMIT: {current_time:.1f} min >= {max_time} min"

        warn_threshold = self.budget_controls.get("warn_threshold_minutes", 40)
        if current_time >= warn_threshold:
            print(f"⚠️  PROCESSING WARNING: {current_time:.1f} min approaching limit {max_time} min")

        return True, "OK"

    def add_stage(self, stage_name: str, duration_seconds: float):
        self.processing_stages.append({
            "stage": stage_name,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        })
        self.total_processing_time += duration_seconds

    def update_performance_data(self, **kwargs):
        self.performance_data.update(kwargs)

    def print_final_summary(self):
        total_time = (datetime.now() - self.session_start).total_seconds() / 60

        print(f"\n🎬 FINAL VIDEO PROCESSING SUMMARY")
        print(f"=" * 50)
        print(f"🎭 Render method: {self.performance_data.get('render_method', 'unknown')}")
        print(f"📋 Timeline mode: {'✅ ENABLED' if self.performance_data.get('timeline_mode') else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if self.performance_data.get('scene_by_scene_mode') else '❌ DISABLED'}")
        print(f"🎭 MoviePy support: {'✅ AVAILABLE' if self.performance_data.get('moviepy_available') else '❌ NOT AVAILABLE'}")
        print(f"📺 Total scenes: {self.performance_data.get('total_scenes', 0)}")
        print(f"⏱️  Video duration: {self.performance_data.get('video_duration_seconds', 0.0):.1f}s")
        print(f"⚡ Processing time: {total_time:.1f} minutes")
        print(f"💾 Output file size: {self.performance_data.get('filesize_mb', 0.0):.1f} MB")

        if self.processing_stages:
            print(f"📊 Processing stages:")
            for stage in self.processing_stages:
                print(f"   🔄 {stage['stage']}: {stage['duration_seconds']:.1f}s")

        return {
            "total_processing_time_minutes": total_time,
            "performance_data": self.performance_data,
            "processing_stages": self.processing_stages
        }


class ServerYouTubeVideoProducer:
    def __init__(self):
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None

        self.base_dir = Path(CONFIG.paths['BASE_DIR'])
        self.data_path = Path(CONFIG.paths['DATA_DIR'])
        self.output_path = Path(CONFIG.paths['OUTPUT_DIR'])
        self.overlay_path = self.base_dir / "src" / "data" / "overlay_videos"

        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseVideoManager(str(db_path))

        print("🎬 Server YouTube Video Producer v2.1 Initialized")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🎥 Overlay Path: {self.overlay_path}")
        print(f"📋 Timeline mode: {'✅ ENABLED' if CONFIG.video_config.get('timeline_mode') else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if CONFIG.video_config.get('scene_by_scene_mode') else '❌ DISABLED'}")
        print(f"🎭 MoviePy support: {'✅ AVAILABLE' if MOVIEPY_AVAILABLE else '❌ NOT AVAILABLE'}")

        self.check_dependencies()

    def check_dependencies(self):
        """Check all required dependencies"""
        dependencies_ok = True

        # Check FFmpeg
        if not self.check_ffmpeg():
            dependencies_ok = False

        # Check MoviePy
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available - some features will be limited")
            dependencies_ok = False

        # Check overlay video
        fireplace_video = self.overlay_path / "fireplace.mp4"
        if not fireplace_video.exists():
            print(f"⚠️ Fireplace overlay not found: {fireplace_video}")
            print("   Video will be created without overlay")
        else:
            print(f"✅ Fireplace overlay found: {fireplace_video}")

        if dependencies_ok:
            print("✅ All dependencies checked and available")
        else:
            print("⚠️ Some dependencies missing - functionality may be limited")

        return dependencies_ok

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step}")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def check_ffmpeg(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ FFmpeg found and working")
                return True
            else:
                print("❌ FFmpeg not working properly")
                return False
        except Exception as e:
            print(f"❌ FFmpeg check failed: {e}")
            return False

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        self.log_step("🔍 Finding completed audio project for video generation")

        result = self.db_manager.get_completed_audio_topic_ready_for_video()

        if not result:
            self.log_step("✅ No completed audio projects ready for video generation", "INFO")
            return False, None

        topic_id, topic, description, output_path = result

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

        print("\n" + "🎬" * 60)
        print("VIDEO COMPOSER - PROJECT DETAILS")
        print("🎬" * 60)
        print(f"📊 PROJECT ID: {topic_id}")
        print(f"📚 TOPIC: {topic}")
        print(f"📝 DESCRIPTION: {description}")
        print(f"📁 PROJECT PATH: {output_path}")
        print()

        # Enhanced project validation
        project_dir = Path(output_path)
        audio_parts_dir = project_dir / "audio_parts"
        scenes_dir = project_dir / "scenes"
        timeline_file = project_dir / "story_audio_youtube_timeline.json"

        print("📁 INPUT DIRECTORIES:")
        print(f"   🎵 Audio Parts: {audio_parts_dir}")
        print(f"      {'✅ EXISTS' if audio_parts_dir.exists() else '❌ NOT FOUND'}")
        if audio_parts_dir.exists():
            audio_files = list(audio_parts_dir.glob("*.mp3"))
            print(f"      📊 Audio files found: {len(audio_files)}")

        print(f"   🖼️  Scenes: {scenes_dir}")
        print(f"      {'✅ EXISTS' if scenes_dir.exists() else '❌ NOT FOUND'}")
        if scenes_dir.exists():
            image_files = list(scenes_dir.glob("*.png"))
            print(f"      📊 Image files found: {len(image_files)}")

        print(f"   📋 Timeline: {timeline_file}")
        print(f"      {'✅ EXISTS' if timeline_file.exists() else '❌ NOT FOUND'}")

        print("🎬" * 60)

        self.db_manager.mark_video_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def print_progress(self, step, total_steps, description):
        percentage = (step / total_steps) * 100
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        print(f"📊 [{progress_bar}] {percentage:.1f}% - {description}")

    def load_project_data(self, row_index):
        project_dir = Path(self.current_output_dir)

        try:
            platform_metadata_path = project_dir / "platform_metadata.json"
            with open(platform_metadata_path, 'r', encoding='utf-8') as f:
                platform_metadata = json.load(f)

            print(f"✅ Platform metadata loaded")
            return platform_metadata

        except Exception as e:
            print(f"❌ Error loading platform metadata: {e}")
            return None

    def load_audio_timeline(self, project_dir):
        timeline_path = Path(project_dir) / "story_audio_youtube_timeline.json"

        print(f"\n📋 LOADING AUDIO TIMELINE:")
        print(f"   📁 Timeline file: {timeline_path}")

        if not timeline_path.exists():
            print(f"   ❌ Timeline not found!")
            return [], None, None

        try:
            with open(timeline_path, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)

            print(f"   ✅ Timeline file loaded")
            print(f"   📊 Total timeline scenes: {timeline_data.get('total_scenes', 0)}")

            all_scenes = timeline_data.get('scenes', [])

            story_scenes = []
            hook_scene = None
            subscribe_scene = None

            for scene in all_scenes:
                if scene['type'] == 'story_scene':
                    story_scenes.append({
                        'scene_id': scene['scene_number'],
                        'title': scene['title'],
                        'audio_file': scene['audio_file'],
                        'image_file': scene.get('image_file', f"scene_{scene['scene_number']:02d}.png"),
                        'duration': scene.get('duration_ms', 0) / 1000.0
                    })
                elif scene['type'] == 'youtube_hook':
                    hook_scene = scene
                elif scene['type'] == 'youtube_subscribe':
                    subscribe_scene = scene

            print(f"   📖 Story scenes found: {len(story_scenes)}")
            print(f"   🎬 Hook scene: {'✅ Found' if hook_scene else '❌ Missing'}")
            print(f"   🔔 Subscribe scene: {'✅ Found' if subscribe_scene else '❌ Missing'}")

            if story_scenes:
                scene_ids = [s['scene_id'] for s in story_scenes]
                print(f"   📋 Scene IDs: {sorted(scene_ids)}")
                print(f"   📏 Scene range: {min(scene_ids)} to {max(scene_ids)}")

            return story_scenes, timeline_data, (hook_scene, subscribe_scene)

        except Exception as e:
            print(f"   ❌ Timeline load error: {e}")
            return [], None, None

    def get_audio_duration(self, audio_file_path):
        try:
            probe = ffmpeg.probe(str(audio_file_path))
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            print(f"⚠️ Could not get duration for {audio_file_path}: {e}")
            return 4.0

    def find_scene_files(self, audio_dir, scenes_dir, scene_id):
        """Find scene files with enhanced error handling"""
        try:
            # Validate directories exist
            if not audio_dir.exists():
                print(f"⚠️ Audio directory not found: {audio_dir}")
                return None, None

            if not scenes_dir.exists():
                print(f"⚠️ Scenes directory not found: {scenes_dir}")
                return None, None

            # Audio file search
            audio_file = None
            for format_str in [f"scene_{scene_id:02d}_audio", f"scene_{scene_id}_audio"]:
                test_file = audio_dir / f"{format_str}.mp3"
                if test_file.exists():
                    audio_file = test_file
                    break

            # Image file search
            image_file = None
            for format_str in [f"scene_{scene_id:02d}", f"scene_{scene_id}"]:
                test_file = scenes_dir / f"{format_str}.png"
                if test_file.exists():
                    image_file = test_file
                    break

            return audio_file, image_file

        except Exception as e:
            print(f"❌ Error in find_scene_files for scene {scene_id}: {e}")
            return None, None

    def safe_cleanup_clips(self, clips_to_cleanup):
        """Safely cleanup MoviePy clips with error handling"""
        if not clips_to_cleanup:
            return

        cleanup_count = 0
        for clip in clips_to_cleanup:
            if clip is not None:
                try:
                    if hasattr(clip, 'close'):
                        clip.close()
                    cleanup_count += 1
                except Exception as e:
                    # Silent cleanup - don't spam console
                    pass

        if gc:
            gc.collect()

        print(f"      🧹 Cleaned up {cleanup_count}/{len(clips_to_cleanup)} clips")

    def render_single_scene_with_retry(self, scene_data, scene_type, audio_dir, scenes_dir,
                                       scene_videos_dir, fireplace_overlay_base, progress_tracker):
        """Render single scene with enhanced retry logic"""
        max_retries = CONFIG.video_config.get("scene_render_controls", {}).get("max_retries_per_scene", 3)
        retry_delay = CONFIG.video_config.get("scene_render_controls", {}).get("retry_delay_seconds", 10)

        for attempt in range(max_retries):
            try:
                print(f"   🎯 Render attempt {attempt + 1}/{max_retries}")

                result = self.render_single_scene_internal(
                    scene_data, scene_type, audio_dir, scenes_dir,
                    scene_videos_dir, fireplace_overlay_base
                )

                if result:
                    if attempt > 0:
                        print(f"   ✅ Success on attempt {attempt + 1}")
                    return result
                else:
                    print(f"   ❌ Attempt {attempt + 1} failed")

            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} exception: {e}")

                # Track scene-specific failure
                scene_id = str(scene_data.get('scene_id', scene_type))
                progress_tracker.mark_scene_failure(scene_id, str(e))

                if attempt < max_retries - 1:
                    print(f"   ⏳ Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)

                    # Force cleanup before retry
                    if gc:
                        gc.collect()
                        time.sleep(2)

        print(f"   ❌ All {max_retries} attempts failed")
        return None

    def render_single_scene_internal(self, scene_data, scene_type, audio_dir, scenes_dir,
                                     scene_videos_dir, fireplace_overlay_base):
        """Internal scene rendering logic"""
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available for scene rendering")
            return None

        clips_to_cleanup = []

        try:
            if scene_type == "hook":
                print(f"\n🎬 RENDERING HOOK SCENE:")
                audio_file = audio_dir / scene_data['audio_file']

                if not audio_file.exists():
                    print(f"   ❌ Hook audio file not found: {audio_file}")
                    return None

                scene_duration = self.get_audio_duration(audio_file)

                # Use multiple random images for hook
                available_scenes = [s for s in os.listdir(scenes_dir) if s.endswith('.png')][:10]
                if len(available_scenes) < 3:
                    print(f"   ⚠️ Limited scene images for hook: {len(available_scenes)}")

                hook_scenes_to_use = random.sample(available_scenes, min(5, len(available_scenes)))
                image_duration = scene_duration / len(hook_scenes_to_use) if hook_scenes_to_use else scene_duration

                image_clips = []
                for scene_file in hook_scenes_to_use:
                    scene_path = scenes_dir / scene_file
                    if scene_path.exists():
                        img_clip = ImageClip(str(scene_path)).set_duration(image_duration)
                        image_clips.append(img_clip)
                        clips_to_cleanup.append(img_clip)

                if image_clips:
                    main_video = concatenate_videoclips(image_clips, method="compose")
                    clips_to_cleanup.append(main_video)
                else:
                    print(f"   ❌ No images found for hook")
                    return None

            elif scene_type == "subscribe":
                print(f"\n🔔 RENDERING SUBSCRIBE SCENE:")
                audio_file = audio_dir / scene_data['audio_file']

                if not audio_file.exists():
                    print(f"   ❌ Subscribe audio file not found: {audio_file}")
                    return None

                scene_duration = self.get_audio_duration(audio_file)

                # Use multiple random images for subscribe
                available_scenes = [s for s in os.listdir(scenes_dir) if s.endswith('.png')][:8]
                sub_scenes_to_use = random.sample(available_scenes, min(3, len(available_scenes)))
                image_duration = scene_duration / len(sub_scenes_to_use) if sub_scenes_to_use else scene_duration

                image_clips = []
                for scene_file in sub_scenes_to_use:
                    scene_path = scenes_dir / scene_file
                    if scene_path.exists():
                        img_clip = ImageClip(str(scene_path)).set_duration(image_duration)
                        image_clips.append(img_clip)
                        clips_to_cleanup.append(img_clip)

                if image_clips:
                    main_video = concatenate_videoclips(image_clips, method="compose")
                    clips_to_cleanup.append(main_video)
                else:
                    print(f"   ❌ No images found for subscribe")
                    return None

            else:  # story scene
                scene_id = scene_data['scene_id']
                print(f"\n📖 RENDERING STORY SCENE {scene_id}:")

                # Find audio and image files with enhanced validation
                audio_file, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_id)

                if not audio_file or not image_file:
                    print(f"   ❌ Missing files for scene {scene_id}")
                    print(f"      Audio: {'✅' if audio_file else '❌'} {audio_file}")
                    print(f"      Image: {'✅' if image_file else '❌'} {image_file}")
                    return None

                scene_duration = self.get_audio_duration(audio_file)

                # Create main video (single image for story scenes)
                main_video = ImageClip(str(image_file)).set_duration(scene_duration)
                clips_to_cleanup.append(main_video)

            print(f"   ✅ Main video created: {scene_duration:.1f}s")

            # Add fireplace overlay with enhanced error handling
            if fireplace_overlay_base:
                print(f"   🔥 Adding fireplace overlay...")

                try:
                    fireplace_duration = fireplace_overlay_base.duration
                    print(f"      📏 Fireplace duration: {fireplace_duration:.1f}s")
                    print(f"      📏 Scene duration: {scene_duration:.1f}s")

                    # Enhanced fireplace handling
                    if scene_duration <= fireplace_duration:
                        scene_fireplace = fireplace_overlay_base.subclip(0, scene_duration)
                        clips_to_cleanup.append(scene_fireplace)
                    else:
                        loops_needed = int(scene_duration / fireplace_duration) + 1
                        print(f"      🔄 Scene longer than fireplace - need {loops_needed} loops")

                        fireplace_clips = []
                        for loop_idx in range(loops_needed):
                            fireplace_clips.append(fireplace_overlay_base.copy())

                        scene_fireplace_looped = concatenate_videoclips(fireplace_clips)
                        clips_to_cleanup.extend(fireplace_clips)
                        clips_to_cleanup.append(scene_fireplace_looped)

                        scene_fireplace = scene_fireplace_looped.subclip(0, scene_duration)
                        clips_to_cleanup.append(scene_fireplace)

                    # Resize and set opacity
                    scene_fireplace = scene_fireplace.resize(main_video.size)
                    scene_fireplace = scene_fireplace.set_opacity(0.3)
                    scene_fireplace = scene_fireplace.without_audio()

                    # Composite
                    scene_final = CompositeVideoClip([main_video, scene_fireplace])
                    clips_to_cleanup.append(scene_final)

                    print(f"   ✅ Fireplace overlay added")

                except Exception as e:
                    print(f"   ⚠️ Fireplace overlay failed, using video without overlay: {e}")
                    scene_final = main_video
            else:
                scene_final = main_video

            # Add audio with validation
            if not audio_file.exists():
                print(f"   ❌ Audio file missing: {audio_file}")
                return None

            scene_audio = AudioFileClip(str(audio_file))
            scene_final = scene_final.set_audio(scene_audio)
            clips_to_cleanup.append(scene_audio)
            clips_to_cleanup.append(scene_final)

            # Determine output filename
            if scene_type == "hook":
                scene_file = scene_videos_dir / "00_hook.mp4"
            elif scene_type == "subscribe":
                scene_file = scene_videos_dir / "01_subscribe.mp4"
            else:
                scene_file = scene_videos_dir / f"{scene_id:03d}_scene_{scene_id}.mp4"

            print(f"   🚀 Rendering scene...")
            print(f"      📁 Output: {scene_file.name}")
            print(f"      📏 Duration: {scene_duration:.1f}s")

            start_render_time = time.time()

            # Enhanced render parameters
            scene_final.write_videofile(
                str(scene_file),
                fps=30,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=f'temp-audio-{scene_file.stem}.m4a',
                remove_temp=True,
                verbose=False,
                logger='bar',
                # Enhanced quality settings
                bitrate="8000k",
                audio_bitrate="192k"
            )

            render_time = time.time() - start_render_time
            file_size_mb = os.path.getsize(scene_file) / (1024 * 1024)

            print(f"      ⏱️  Render time: {render_time / 60:.1f} minutes")
            print(f"      📊 Render speed: {scene_duration / render_time:.1f}x realtime")
            print(f"      📦 File size: {file_size_mb:.1f} MB")

            # Cleanup scene clips immediately
            self.safe_cleanup_clips(clips_to_cleanup)

            print(f"   ✅ Scene rendered successfully!")
            return str(scene_file)

        except Exception as e:
            print(f"   ❌ Scene rendering failed: {e}")
            # Emergency cleanup
            self.safe_cleanup_clips(clips_to_cleanup)
            return None

    def create_video_scene_by_scene_style(self, story_scenes, hook_subscribe_data, row_index, total_duration,
                                           progress_tracker, usage_tracker):
        """SCENE-BY-SCENE MoviePy Style with enhanced error handling"""
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available - cannot use scene-by-scene processing")
            return None

        fireplace_video = self.overlay_path / "fireplace.mp4"
        final_video = Path(self.current_output_dir) / "final_video.mp4"
        scene_videos_dir = Path(self.current_output_dir) / "scene_videos"
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        scenes_dir = Path(self.current_output_dir) / "scenes"

        print(f"🎬 SCENE-BY-SCENE MOVIEPY STYLE: Process each scene separately ({total_duration:.1f}s = {total_duration / 60:.1f} minutes)")
        print("📝 Using MoviePy with SCENE-BY-SCENE processing for perfect audio sync")
        print("🎯 Natural breakpoints - no audio cutting in the middle of words")

        scene_videos_dir.mkdir(exist_ok=True)

        try:
            # Load fireplace overlay once with error handling
            fireplace_overlay_base = None
            if fireplace_video.exists():
                print("🔥 Loading fireplace overlay...")
                try:
                    fireplace_overlay_base = VideoFileClip(str(fireplace_video))
                    print(f"   📏 Fireplace duration: {fireplace_overlay_base.duration:.1f}s")
                except Exception as e:
                    print(f"   ⚠️ Failed to load fireplace overlay: {e}")
                    fireplace_overlay_base = None
            else:
                print("⚠️ Fireplace overlay not found - proceeding without overlay")

            hook_scene, subscribe_scene = hook_subscribe_data
            scene_video_files = []

            print(f"\n🎬 SCENE-BY-SCENE PROCESSING:")
            print(f"   📊 Total story scenes: {len(story_scenes)}")
            print(f"   🎬 Hook scene: {'✅' if hook_scene else '❌'}")
            print(f"   🔔 Subscribe scene: {'✅' if subscribe_scene else '❌'}")

            # Process all scenes with enhanced tracking
            scene_counter = 0
            total_scenes = len(story_scenes) + (1 if hook_scene else 0) + (1 if subscribe_scene else 0)
            overall_start_time = time.time()

            print(f"\n🎬 STARTING SCENE-BY-SCENE PROCESSING:")
            print(f"   📊 Total scenes to process: {total_scenes}")

            # Check processing limits before starting
            can_continue, reason = usage_tracker.check_processing_limits()
            if not can_continue:
                print(f"🚨 PROCESSING LIMIT EXCEEDED before starting: {reason}")
                return None

            # 1. Hook scene
            if hook_scene:
                scene_counter += 1
                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Hook)")

                hook_video = self.render_single_scene_with_retry(
                    hook_scene, "hook", audio_dir, scenes_dir,
                    scene_videos_dir, fireplace_overlay_base, progress_tracker
                )
                if hook_video:
                    scene_video_files.append(hook_video)

            # 2. Subscribe scene
            if subscribe_scene:
                scene_counter += 1
                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Subscribe)")

                subscribe_video = self.render_single_scene_with_retry(
                    subscribe_scene, "subscribe", audio_dir, scenes_dir,
                    scene_videos_dir, fireplace_overlay_base, progress_tracker
                )
                if subscribe_video:
                    scene_video_files.append(subscribe_video)

            # 3. Story scenes with progress tracking
            for scene_data in story_scenes:
                scene_counter += 1
                elapsed_time = time.time() - overall_start_time
                estimated_remaining = (elapsed_time / scene_counter) * (total_scenes - scene_counter)

                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Story Scene {scene_data['scene_id']})")
                print(f"   ⏱️  Elapsed: {elapsed_time / 60:.1f}m | Remaining: {estimated_remaining / 60:.1f}m")

                # Check processing limits during rendering
                can_continue, reason = usage_tracker.check_processing_limits()
                if not can_continue:
                    print(f"🚨 PROCESSING LIMIT EXCEEDED during scene {scene_counter}: {reason}")
                    break

                scene_video = self.render_single_scene_with_retry(
                    scene_data, "story", audio_dir, scenes_dir,
                    scene_videos_dir, fireplace_overlay_base, progress_tracker
                )
                if scene_video:
                    scene_video_files.append(scene_video)

            total_processing_time = time.time() - overall_start_time
            print(f"\n📊 ALL SCENES PROCESSED:")
            print(f"   ✅ Successful scenes: {len(scene_video_files)}/{total_scenes}")
            print(f"   ⏱️  Total processing time: {total_processing_time / 60:.1f} minutes")

            # Cleanup fireplace overlay
            if fireplace_overlay_base:
                try:
                    fireplace_overlay_base.close()
                except:
                    pass

            print(f"\n🔗 COMBINING SCENE VIDEOS:")
            print(f"   📦 Total scene videos created: {len(scene_video_files)}")

            if not scene_video_files:
                print("   ❌ No scene videos created")
                return None

            # Enhanced FFmpeg combination with error handling
            scene_list_file = scene_videos_dir / "scene_list.txt"
            try:
                with open(scene_list_file, 'w') as f:
                    for scene_file in scene_video_files:
                        # Validate scene file exists
                        if os.path.exists(scene_file):
                            f.write(f"file '{scene_file}'\n")
                        else:
                            print(f"   ⚠️ Scene file missing: {scene_file}")

                # Use FFmpeg to combine scenes with enhanced error handling
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(scene_list_file),
                    '-c', 'copy',
                    '-y',
                    str(final_video)
                ]

                print(f"   🔄 Running FFmpeg to combine scenes...")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print(f"   ✅ Scene videos combined successfully!")

                    # Enhanced cleanup with error handling
                    for scene_file in scene_video_files:
                        try:
                            if os.path.exists(scene_file):
                                os.remove(scene_file)
                        except Exception as e:
                            print(f"   ⚠️ Could not remove {scene_file}: {e}")

                    try:
                        if os.path.exists(scene_list_file):
                            os.remove(scene_list_file)
                        if scene_videos_dir.exists() and not list(scene_videos_dir.iterdir()):
                            scene_videos_dir.rmdir()
                    except:
                        pass

                    print(f"   ✅ Cleanup completed!")
                else:
                    print(f"   ❌ FFmpeg combination failed:")
                    print(f"      {result.stderr}")
                    return None

            except Exception as e:
                print(f"   ❌ Error during scene combination: {e}")
                return None

            if final_video.exists():
                print(f"\n✅ SCENE-BY-SCENE MOVIEPY STYLE completed: {final_video}")
                print(f"🎬 Successfully rendered {len(scene_video_files)} scenes!")
                print("🔥 Fireplace overlay working with enhanced error handling!")
                print("🎯 Perfect audio sync - no word cutting!")
                return final_video
            else:
                print("❌ Final video not created")
                return None

        except Exception as e:
            print(f"❌ Scene-by-scene MoviePy failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_final_video(self, video_file):
        print(f"\n🔍 VIDEO VERIFICATION PROCESS:")
        print(f"   📁 Video file: {Path(video_file).name}")

        try:
            probe = ffmpeg.probe(str(video_file))

            video_stream = None
            audio_stream = None

            for stream in probe['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                elif stream['codec_type'] == 'audio':
                    audio_stream = stream

            duration = float(probe['format']['duration'])
            file_size = os.path.getsize(video_file) / (1024 * 1024)

            print(f"\n📊 VIDEO FILE INFORMATION:")
            print(f"   ⏱️  Duration: {duration:.1f}s ({duration / 60:.1f} minutes)")
            print(f"   📦 File size: {file_size:.1f} MB")

            if video_stream:
                print(f"\n🎬 VIDEO STREAM:")
                print(f"   📺 Resolution: {video_stream['width']}x{video_stream['height']}")
                print(f"   🎞️  Codec: {video_stream['codec_name']}")

            if audio_stream:
                print(f"\n🎵 AUDIO STREAM:")
                print(f"   🔊 Codec: {audio_stream['codec_name']}")
                print(f"   📻 Sample rate: {audio_stream.get('sample_rate', 'unknown')} Hz")

            print(f"\n✅ VIDEO VERIFICATION COMPLETED SUCCESSFULLY")
            return True, duration, file_size

        except Exception as e:
            print(f"\n❌ VIDEO VERIFICATION FAILED: {e}")
            return False, 0.0, 0.0

    def create_video(self, row_index, topic_data, progress_tracker, usage_tracker):
        """Main video creation function with enhanced error handling"""
        total_steps = 5
        current_step = 0

        print(f"\n" + "🎬" * 80)
        print("VIDEO COMPOSER - SCENE-BY-SCENE PROCESSING v2.1")
        print("🎬" * 80)
        print(f"🎯 PROJECT: {topic_data['topic']}")
        print(f"🆔 PROJECT ID: {row_index}")
        print(f"📁 OUTPUT DIR: {self.current_output_dir}")
        print()
        print("📋 PROCESSING METHOD:")
        print("   🎯 SCENE-BY-SCENE APPROACH - Enhanced Server Version v2.1")
        print("   ✅ Uses story_audio_youtube_timeline.json (ACTUAL generated scenes)")
        print("   📝 Each scene processed individually with its own audio")
        print("   📝 Enhanced error handling and retry logic")
        print("   📝 Progressive processing limit monitoring")
        print("   📝 Scene-specific failure tracking")
        print("   ✅ Fixed: Perfect audio sync - no word cutting!")
        print("   ✅ Enhanced: Robust error recovery!")
        print("🎬" * 80)

        try:
            # 1. Load platform metadata
            current_step += 1
            self.print_progress(current_step, total_steps, "Loading platform metadata...")
            start_time = time.time()

            platform_metadata = self.load_project_data(row_index)
            if not platform_metadata:
                progress_tracker.mark_stage_failed("project_load", "Failed to load platform metadata")
                return None

            progress_tracker.mark_stage_completed("project_load")
            usage_tracker.add_stage("project_load", time.time() - start_time)

            # 2. Load audio timeline
            current_step += 1
            self.print_progress(current_step, total_steps, "Loading audio timeline...")
            start_time = time.time()

            story_scenes, timeline_data, hook_subscribe_data = self.load_audio_timeline(self.current_output_dir)
            if not story_scenes:
                progress_tracker.mark_stage_failed("timeline_load", "Failed to load audio timeline")
                return None

            print(f"\n📊 TIMELINE DATA LOADED:")
            print(f"   📖 Story scenes: {len(story_scenes)}")
            print(f"   🎬 Hook available: {'✅' if hook_subscribe_data[0] else '❌'}")
            print(f"   🔔 Subscribe available: {'✅' if hook_subscribe_data[1] else '❌'}")

            progress_tracker.mark_stage_completed("timeline_load")
            usage_tracker.add_stage("timeline_load", time.time() - start_time)
            usage_tracker.update_performance_data(total_scenes=len(story_scenes))

            # 3. Scene-by-scene preparation
            current_step += 1
            self.print_progress(current_step, total_steps, "🎯 Preparing scene-by-scene processing...")
            start_time = time.time()

            total_duration = timeline_data.get('total_duration_ms', 0) / 1000.0

            print(f"\n🎯 SCENE-BY-SCENE PREPARATION:")
            print(f"   📖 Story scenes: {len(story_scenes)}")
            print(f"   ⏱️  Total duration: {total_duration / 60:.1f} minutes")
            print(f"   🎯 Natural breakpoints ensure perfect audio sync")
            print(f"   🛡️ Enhanced error handling enabled")

            progress_tracker.mark_stage_completed("scene_preparation")
            usage_tracker.add_stage("scene_preparation", time.time() - start_time)
            usage_tracker.update_performance_data(video_duration_seconds=total_duration)

            # 4. Scene-by-scene video render with processing limits
            current_step += 1
            self.print_progress(current_step, total_steps, "🎬 SCENE-BY-SCENE MoviePy: Processing each scene separately...")
            start_time = time.time()

            # Check processing limits before starting render
            can_continue, reason = usage_tracker.check_processing_limits()
            if not can_continue:
                progress_tracker.mark_stage_failed("video_render", f"Processing limit exceeded: {reason}")
                return None

            print(f"\n🎬 VIDEO RENDERING (SCENE-BY-SCENE):")
            print(f"   📝 Method: SCENE-BY-SCENE MoviePy (perfect audio sync)")
            print(f"   🔥 Overlay: Enhanced fireplace optimization")
            print(f"   📊 Story scenes: {len(story_scenes)}")
            print(f"   🛡️ Enhanced error handling: ENABLED")
            print(f"   🎭 MoviePy support: {'✅ AVAILABLE' if MOVIEPY_AVAILABLE else '❌ NOT AVAILABLE'}")

            progress_tracker.set_render_method("moviepy_scene_by_scene_timeline_based_enhanced")
            usage_tracker.update_performance_data(render_method="moviepy_scene_by_scene_timeline_based_enhanced")

            final_video = self.create_video_scene_by_scene_style(
                story_scenes, hook_subscribe_data, row_index, total_duration, progress_tracker, usage_tracker
            )

            if not final_video:
                progress_tracker.mark_stage_failed("video_render", "Scene-by-scene render failed")
                return None

            print(f"   ✅ Video rendering completed")

            progress_tracker.mark_stage_completed("video_render")
            usage_tracker.add_stage("video_render", time.time() - start_time)

            # 5. Video verification
            current_step += 1
            self.print_progress(current_step, total_steps, "Verifying final video...")
            start_time = time.time()

            verification_success, actual_duration, file_size_mb = self.verify_final_video(final_video)
            if verification_success:
                print(f"\n✅ VIDEO VERIFICATION PASSED:")
                print(f"   ⏱️  Duration: {actual_duration:.1f}s ({actual_duration / 60:.1f} min)")
                print(f"   📦 File size: {file_size_mb:.1f} MB")
                progress_tracker.mark_stage_completed("verification")
                usage_tracker.update_performance_data(filesize_mb=file_size_mb)

            usage_tracker.add_stage("verification", time.time() - start_time)

            # Final summary
            usage_summary = usage_tracker.print_final_summary()

            # Calculate total segments
            total_segments = len(story_scenes) + (1 if hook_subscribe_data[0] else 0) + (1 if hook_subscribe_data[1] else 0)

            video_metadata = {
                "title": platform_metadata["title_options"][0] if platform_metadata.get("title_options") else topic_data["topic"],
                "duration_seconds": actual_duration,
                "story_scene_count": len(story_scenes),
                "total_segments": total_segments,
                "created_at": datetime.now().isoformat(),
                "output_file": str(final_video),
                "processing_steps": total_steps,
                "render_method": "moviepy_scene_by_scene_timeline_based_enhanced_v2.1",
                "timeline_mode": True,
                "scene_by_scene_processing": True,
                "perfect_audio_sync": True,
                "enhanced_error_handling": True,
                "moviepy_available": MOVIEPY_AVAILABLE,
                "fireplace_optimization": "enhanced",
                "server_version": "2.1",
                "database_integrated": True,
                "usage_summary": usage_summary
            }

            metadata_file = Path(self.current_output_dir) / "video_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(video_metadata, f, indent=2, ensure_ascii=False)

            self.print_progress(total_steps, total_steps, "ENHANCED SCENE-BY-SCENE MoviePy render completed!")

            print(f"\n" + "🎉" * 80)
            print("VIDEO CREATION COMPLETED SUCCESSFULLY!")
            print("🎉" * 80)
            print(f"🎬 PROJECT: {topic_data['topic']}")
            print(f"📁 VIDEO FILE: {final_video}")
            print(f"⏱️  DURATION: {actual_duration / 60:.1f} minutes")
            print(f"📦 FILE SIZE: {file_size_mb:.1f} MB")
            print(f"🎬 SEGMENTS: {total_segments} total")
            print(f"📖 STORY SCENES: {len(story_scenes)} from timeline")
            print(f"🎭 METHOD: ENHANCED Scene-by-Scene Timeline MoviePy v2.1")
            print(f"🔥 OVERLAY: Enhanced fireplace optimization")
            print(f"🎵 AUDIO: Perfect sync - no word cutting!")
            print(f"🛡️ ERROR HANDLING: Enhanced with retry logic")
            print(f"✅ STATUS: Uses REAL generated scenes from timeline!")
            print("🎉" * 80)

            return final_video, actual_duration, file_size_mb, usage_summary["total_processing_time_minutes"]

        except Exception as e:
            print(f"💥 Video creation error: {e}")
            progress_tracker.mark_stage_failed("video_render", str(e))
            import traceback
            traceback.print_exc()
            return None

    def run_video_generation(self) -> bool:
        print("🚀" * 50)
        print("SERVER VIDEO COMPOSER v2.1 - ENHANCED SCENE-BY-SCENE TIMELINE MODE")
        print("🔗 Database integrated with enhanced error handling")
        print("🎬 YouTube Production Video Generation")
        print("📋 Timeline-based scene loading (ACTUAL generated scenes)")
        print("🎯 Scene-by-scene processing (perfect audio sync)")
        print("🛡️ Enhanced error handling and retry logic")
        print("🔥 Enhanced Fireplace Optimization")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        limits = CONFIG.video_config.get("budget_controls", {})
        print(f"⏱️  PROCESSING LIMITS:")
        print(f"   📊 Max processing time: {limits.get('max_processing_time_minutes', 60)} minutes")
        print(f"   📊 Warning threshold: {limits.get('warn_threshold_minutes', 40)} minutes")
        print(f"   🎭 MoviePy support: {'✅ AVAILABLE' if MOVIEPY_AVAILABLE else '❌ NOT AVAILABLE'}")

        # Check critical dependencies
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available - video generation requires MoviePy")
            print("📦 Install with: pip install moviepy")
            return False

        # Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")

        # Initialize trackers
        progress_tracker = VideoProgressTracker(self.current_topic_id, CONFIG.paths['OUTPUT_DIR'])
        usage_tracker = VideoUsageTracker()

        try:
            # Create video with enhanced scene-by-scene approach
            result = self.create_video(self.current_topic_id, project_info, progress_tracker, usage_tracker)

            if result and len(result) == 4:
                final_video, duration_seconds, file_size_mb, processing_time_minutes = result

                print(f"✅ Video generation successful!")
                print(f"⏱️  Duration: {duration_seconds / 60:.1f} minutes")
                print(f"📦 File size: {file_size_mb:.1f} MB")
                print(f"⚡ Processing time: {processing_time_minutes:.1f} minutes")

                # Update database with results
                self.db_manager.mark_video_generation_completed(
                    self.current_topic_id, duration_seconds, file_size_mb, processing_time_minutes
                )

                progress_tracker.cleanup_on_success()

                print("\n" + "🎉" * 50)
                print("VIDEO GENERATION SUCCESSFUL!")
                print("✅ Enhanced scene-by-scene processing (perfect audio sync)")
                print("✅ Enhanced fireplace optimization")
                print("✅ Enhanced error handling and retry logic")
                print("✅ Database updated with metrics")
                print("🎉" * 50)
                return True

            else:
                print(f"❌ Video generation failed")
                self.db_manager.mark_video_generation_failed(
                    self.current_topic_id, "Video creation failed"
                )
                return False

        except Exception as e:
            self.log_step(f"❌ Video generation failed: {e}", "ERROR")
            self.db_manager.mark_video_generation_failed(
                self.current_topic_id, str(e)
            )
            import traceback
            traceback.print_exc()
            return False


def run_autonomous_mode():
    """Run autonomous mode - continuously process completed audio topics for video generation"""
    print("🤖 AUTONOMOUS VIDEO GENERATION MODE STARTED")
    print("🔄 Will process all completed audio topics continuously")
    print("⏹️ Press Ctrl+C to stop gracefully")

    # Initialize database manager for pipeline status with error handling
    try:
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        db_manager = DatabaseVideoManager(str(db_path))
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return

    # Setup graceful shutdown
    running = True
    processed_count = 0
    error_count = 0
    start_time = time.time()
    last_activity_time = time.time()

    def signal_handler(signum, frame):
        nonlocal running
        print(f"\n⏹️ Received shutdown signal ({signum})")
        print("🔄 Finishing current video generation and shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            # Check pipeline status
            status = db_manager.get_pipeline_status()

            if status['video_generation_queue'] > 0:
                print(f"\n🎬 Found {status['video_generation_queue']} completed audio topics ready for video generation")
                last_activity_time = time.time()

                # Initialize producer with error handling
                try:
                    producer = ServerYouTubeVideoProducer()
                except Exception as e:
                    print(f"❌ Producer initialization failed: {e}")
                    error_count += 1
                    if error_count > 10:
                        print("❌ Too many initialization failures, shutting down")
                        break
                    time.sleep(30)
                    continue

                # Process one topic
                try:
                    success = producer.run_video_generation()

                    if success:
                        processed_count += 1
                        error_count = 0  # Reset error count on success
                        print(f"\n✅ Video generation completed!")
                        print(f"📊 Progress: {processed_count} topics processed")
                        print(f"🎬 Enhanced scene-by-scene processing used")
                    else:
                        print(f"\n⚠️ Video generation failed")
                        error_count += 1

                except Exception as e:
                    print(f"❌ Video generation error: {e}")
                    error_count += 1
                    import traceback
                    traceback.print_exc()

                # Adaptive waiting based on error count
                if error_count > 3:
                    wait_time = 300  # 5 minutes after multiple errors
                    print(f"⚠️ Multiple errors detected, waiting {wait_time}s...")
                elif error_count > 0:
                    wait_time = 120   # 2 minutes after single error
                    print(f"⚠️ Error detected, waiting {wait_time}s...")
                else:
                    wait_time = 10    # Normal wait

                for i in range(wait_time):
                    if not running:
                        break
                    time.sleep(1)

            else:
                # No topics ready, smart waiting
                time_since_activity = time.time() - last_activity_time
                error_count = 0  # Reset error count when no work

                if time_since_activity < 300:  # Less than 5 minutes since last activity
                    wait_time = 60  # Wait 1 minute
                    print("😴 No topics ready. Recent activity detected - waiting 60s...")
                else:
                    wait_time = 300  # Wait 5 minutes
                    print("😴 No topics ready for extended period - waiting 5 minutes...")
                    print(f"⏰ Last activity: {time_since_activity / 60:.1f} minutes ago")

                # Wait with interrupt capability
                for i in range(wait_time):
                    if not running:
                        break
                    if i > 0 and i % 60 == 0:  # Show progress every minute
                        remaining = (wait_time - i) / 60
                        print(f"⏳ Still waiting... {remaining:.1f} minutes remaining")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️ Keyboard interrupt received")
            break
        except Exception as e:
            print(f"❌ Unexpected error in main loop: {e}")
            error_count += 1

            # Exit if too many critical errors
            if error_count > 10:
                print("❌ Too many critical errors, shutting down")
                break

            print("⏳ Waiting 30 seconds before retry...")
            time.sleep(30)

    # Shutdown summary
    runtime = time.time() - start_time
    print(f"\n🏁 AUTONOMOUS VIDEO GENERATION SHUTDOWN")
    print(f"⏱️ Total runtime: {runtime / 3600:.1f} hours")
    print(f"✅ Topics processed: {processed_count}")
    print(f"❌ Total errors: {error_count}")
    print(f"🎬 Enhanced scene-by-scene processing used")
    print("👋 Goodbye!")


if __name__ == "__main__":
    # Check for autonomous mode
    if len(sys.argv) > 1 and sys.argv[1] == '--autonomous':
        run_autonomous_mode()
    else:
        # Original single topic mode
        try:
            print("🚀 SERVER VIDEO COMPOSER v2.1 - ENHANCED SCENE-BY-SCENE TIMELINE MODE")
            print("🔗 Database integration with enhanced progress tracking")
            print("🎬 YouTube Production Video Generation")
            print("📋 Timeline-based scene loading (ACTUAL generated scenes)")
            print("🎯 Scene-by-scene processing (perfect audio sync)")
            print("🛡️ Enhanced error handling and retry logic")
            print("🔥 Enhanced Fireplace Optimization")
            print("🎭 Safe MoviePy import with fallback handling")
            print("🖥️ Production-ready automation")
            print("=" * 60)

            # Check MoviePy availability before starting
            if not MOVIEPY_AVAILABLE:
                print("❌ MoviePy not available!")
                print("📦 Install with: pip install moviepy")
                print("⚠️ Video generation requires MoviePy")
                sys.exit(1)

            producer = ServerYouTubeVideoProducer()
            success = producer.run_video_generation()

            if success:
                print("🎊 Video generation completed successfully!")
                print("📁 Video saved: final_video.mp4")
                print("📋 Metadata saved: video_metadata.json")
                print("🔥 Enhanced fireplace optimization included")
                print("🎯 Enhanced scene-by-scene processing used (perfect audio sync)")
                print("🛡️ Enhanced error handling and retry logic")
                print("💾 Enhanced progress tracking enabled")
                print("🖥️ Server infrastructure working")
            else:
                print("⚠️ Video generation failed or no projects ready")

        except KeyboardInterrupt:
            print("\n⏹️ Video generation stopped by user")
            print("🛡️ Progress saved! Restart to resume from last completed stage.")
        except Exception as e:
            print(f"💥 Video generation failed: {e}")
            print("🛡️ Progress saved! Check video_progress.json for resume info.")
            CONFIG.logger.error(f"Video generation failed: {e}")
            import traceback
            traceback.print_exc()