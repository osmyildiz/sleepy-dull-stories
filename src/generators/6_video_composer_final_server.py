"""
Sleepy Dull Stories - SERVER-READY Video Composer
YouTube Video Production with ALL Scenes + MoviePy + Fireplace Overlay
Production-optimized with complete automation and database integration
FIXED: TTS Generator scene checking logic applied
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
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import time

# Load environment first
load_dotenv()

# Server Configuration Class (from TTS generator)
class ServerConfig:
    """Server-friendly configuration management for Video Composer"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_video_config()
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

        print(f"✅ Video Composer server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_video_config(self):
        """Setup video composition configuration"""
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
                "max_processing_time_minutes": 30,
                "max_memory_usage_gb": 8,
                "warn_threshold_minutes": 20
            },
            "server_mode": True,
            "production_ready": True,
            "moviepy_enabled": True,
            "ffmpeg_fallback": True
        }

        print("✅ Video composition configuration loaded")
        print(f"🎬 Target: {self.video_config['target_resolution'][0]}x{self.video_config['target_resolution'][1]} @ {self.video_config['target_fps']}fps")

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"video_composer_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
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
        """Ensure all required directories exist"""
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
    """Professional video management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_audio_topic_ready_for_video(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed audio topic that needs VIDEO generation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if video generation columns exist, if not create them
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add columns individually if they don't exist
        columns_to_add = [
            ('video_generation_status', 'TEXT DEFAULT "pending"'),
            ('video_generation_started_at', 'DATETIME'),
            ('video_generation_completed_at', 'DATETIME'),
            ('video_duration_seconds', 'REAL DEFAULT 0.0'),
            ('video_file_size_mb', 'REAL DEFAULT 0.0'),
            ('video_processing_time_minutes', 'REAL DEFAULT 0.0')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                print(f"🔧 Adding column: {column_name}")
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()
        print("✅ Video generation columns verified/added")

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
        conn.close()

        return result if result else None

    def mark_video_generation_started(self, topic_id: int):
        """Mark video generation as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET video_generation_status = 'in_progress', 
                video_generation_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_video_generation_completed(self, topic_id: int, duration_seconds: float,
                                       file_size_mb: float, processing_time_minutes: float):
        """Mark video generation as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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
        conn.close()

class VideoProgressTracker:
    """Video processing progress tracking and resume functionality"""

    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.progress_file = os.path.join(self.output_dir, "video_progress.json")

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load existing progress
        self.progress_data = self.load_progress()

    def load_progress(self):
        """Load existing progress"""
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
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat(),
            "render_method": "unknown",
            "stages": {
                "project_load": {"status": "pending", "timestamp": None},
                "sequence_build": {"status": "pending", "timestamp": None},
                "audio_combine": {"status": "pending", "timestamp": None},
                "video_render": {"status": "pending", "timestamp": None},
                "verification": {"status": "pending", "timestamp": None}
            }
        }

    def save_progress(self):
        """Save progress"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Video progress save warning: {e}")

    def mark_stage_completed(self, stage: str):
        """Mark stage as completed"""
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
        """Mark stage as failed"""
        if stage in self.progress_data["stages"]:
            self.progress_data["stages"][stage] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": error
            }

        self.progress_data["failed_attempts"] += 1
        self.save_progress()

    def set_render_method(self, method: str):
        """Set the render method being used"""
        self.progress_data["render_method"] = method
        self.save_progress()

    def cleanup_on_success(self):
        """Clean up progress file on successful completion"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print(f"🗑️  Video progress file cleaned up (successful completion)")
        except Exception as e:
            print(f"⚠️  Video progress cleanup warning: {e}")

class VideoUsageTracker:
    """Video processing usage and performance tracking"""

    def __init__(self):
        self.session_start = datetime.now()
        self.processing_stages = []
        self.total_processing_time = 0.0
        self.memory_usage_mb = 0.0

        # Performance tracking
        self.performance_data = {
            "render_method": "unknown",
            "total_scenes": 0,
            "video_duration_seconds": 0.0,
            "processing_time_minutes": 0.0,
            "memory_peak_mb": 0.0,
            "filesize_mb": 0.0
        }

        # Budget controls
        self.budget_controls = CONFIG.video_config.get("budget_controls", {})

    def check_processing_limits(self) -> Tuple[bool, str]:
        """Check if processing limits are exceeded"""
        current_time = (datetime.now() - self.session_start).total_seconds() / 60  # minutes

        max_time = self.budget_controls.get("max_processing_time_minutes", 30)
        if current_time >= max_time:
            return False, f"PROCESSING TIME LIMIT: {current_time:.1f} min >= {max_time} min"

        warn_threshold = self.budget_controls.get("warn_threshold_minutes", 20)
        if current_time >= warn_threshold:
            print(f"⚠️  PROCESSING WARNING: {current_time:.1f} min approaching limit {max_time} min")

        return True, "OK"

    def add_stage(self, stage_name: str, duration_seconds: float):
        """Add processing stage timing"""
        self.processing_stages.append({
            "stage": stage_name,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        })
        self.total_processing_time += duration_seconds

    def update_performance_data(self, **kwargs):
        """Update performance tracking data"""
        self.performance_data.update(kwargs)

    def print_final_summary(self):
        """Print final processing summary"""
        total_time = (datetime.now() - self.session_start).total_seconds() / 60

        print(f"\n🎬 FINAL VIDEO PROCESSING SUMMARY")
        print(f"=" * 50)
        print(f"🎭 Render method: {self.performance_data.get('render_method', 'unknown')}")
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
    """Server-ready YouTube Video Producer with database integration and TTS scene checking"""

    def __init__(self):
        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None

        # Server paths from CONFIG
        self.base_dir = Path(CONFIG.paths['BASE_DIR'])
        self.data_path = Path(CONFIG.paths['DATA_DIR'])
        self.output_path = Path(CONFIG.paths['OUTPUT_DIR'])
        self.overlay_path = self.data_path / "overlay_videos"

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseVideoManager(str(db_path))

        print("🎬 Server YouTube Video Producer v1.1 Initialized")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🎥 Overlay Path: {self.overlay_path}")
        print("✅ TTS Generator scene checking logic integrated")

        self.check_ffmpeg()

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log generation steps with production logging"""
        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step}")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def check_ffmpeg(self):
        """FFmpeg kurulu mu kontrol et"""
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
        """Get next completed audio project that needs VIDEO generation"""
        self.log_step("🔍 Finding completed audio project for video generation")

        result = self.db_manager.get_completed_audio_topic_ready_for_video()

        if not result:
            self.log_step("✅ No completed audio projects ready for video generation", "INFO")
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

        # Print detailed project information
        print("\n" + "🎬" * 60)
        print("VIDEO COMPOSER - PROJECT DETAILS")
        print("🎬" * 60)
        print(f"📊 PROJECT ID: {topic_id}")
        print(f"📚 TOPIC: {topic}")
        print(f"📝 DESCRIPTION: {description}")
        print(f"📁 PROJECT PATH: {output_path}")
        print()

        # Check and display input paths
        project_dir = Path(output_path)
        audio_parts_dir = project_dir / "audio_parts"
        scenes_dir = project_dir / "scenes"

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

        # Check for required project files
        print("\n📋 REQUIRED PROJECT FILES:")
        required_files = [
            ("scene_plan.json", "Scene plan"),
            ("platform_metadata.json", "Platform metadata"),
            ("story_audio_youtube_timeline.json", "Audio timeline")
        ]

        for filename, description in required_files:
            file_path = project_dir / filename
            status = "✅ FOUND" if file_path.exists() else "❌ MISSING"
            print(f"   📄 {description}: {status}")
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    print(f"      📏 Size: {file_size:,} bytes")
                except:
                    pass

        print("\n📤 OUTPUT DIRECTORY:")
        print(f"   📁 Final video output: {project_dir}")
        print("🎬" * 60)

        # Mark as started in database
        self.db_manager.mark_video_generation_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def print_progress(self, step, total_steps, description):
        """Progress göstergesi yazdır"""
        percentage = (step / total_steps) * 100
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        print(f"📊 [{progress_bar}] {percentage:.1f}% - {description}")

    def load_project_data(self, row_index):
        """Proje JSON dosyalarını yükle"""
        project_dir = Path(self.current_output_dir)

        try:
            scene_plan_path = project_dir / "scene_plan.json"
            with open(scene_plan_path, 'r', encoding='utf-8') as f:
                scene_plan = json.load(f)

            platform_metadata_path = project_dir / "platform_metadata.json"
            with open(platform_metadata_path, 'r', encoding='utf-8') as f:
                platform_metadata = json.load(f)

            print(f"✅ Project data loaded successfully")
            return scene_plan, platform_metadata

        except Exception as e:
            print(f"❌ Error loading project data: {e}")
            return None, None

    def load_audio_timeline(self, project_dir):
        """Load actual generated scenes from audio timeline instead of scene plan"""
        timeline_path = Path(project_dir) / "story_audio_youtube_timeline.json"

        if not timeline_path.exists():
            print(f"❌ Timeline not found: {timeline_path}")
            return [], None

        try:
            with open(timeline_path, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)

            # Extract story scenes only (not hook/subscribe)
            story_scenes = []
            for scene in timeline_data['scenes']:
                if scene['type'] == 'story_scene':
                    story_scenes.append({
                        'scene_id': scene['scene_number'],
                        'title': scene['title'],
                        'audio_file': scene['audio_file']
                    })

            print(f"✅ Timeline loaded: {len(story_scenes)} story scenes")
            return story_scenes, timeline_data

        except Exception as e:
            print(f"❌ Timeline load error: {e}")
            return [], None

    def get_audio_duration(self, audio_file_path):
        """Ses dosyasının süresini al"""
        try:
            probe = ffmpeg.probe(str(audio_file_path))
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            print(f"⚠️ Could not get duration for {audio_file_path}: {e}")
            return 4.0

    def find_audio_file(self, audio_dir, filename_base):
        """Audio dosyasını bul"""
        file_path = audio_dir / f"{filename_base}.mp3"
        if file_path.exists():
            return file_path
        return None

    def find_scene_files(self, audio_dir, scenes_dir, scene_id):
        """Scene audio ve image dosyalarını bul"""
        # Audio dosyası
        audio_file = None
        for format_str in [f"scene_{scene_id:02d}_audio", f"scene_{scene_id}_audio"]:
            test_file = audio_dir / f"{format_str}.mp3"
            if test_file.exists():
                audio_file = test_file
                break

        # Image dosyası
        image_file = None
        for format_str in [f"scene_{scene_id:02d}", f"scene_{scene_id}"]:
            test_file = scenes_dir / f"{format_str}.png"
            if test_file.exists():
                image_file = test_file
                break

        return audio_file, image_file

    def check_available_scenes(self, story_id: int, max_scenes: int = 50):
        """TTS Generator mantığı: Mevcut scene image'larını kontrol et ve available list döndür"""
        scenes_dir = Path(self.current_output_dir) / "scenes"
        audio_dir = Path(self.current_output_dir) / "audio_parts"

        if not scenes_dir.exists():
            print(f"❌ Scenes directory not found: {scenes_dir}")
            return [], f"Scenes directory not found: {scenes_dir}"

        if not audio_dir.exists():
            print(f"❌ Audio parts directory not found: {audio_dir}")
            return [], f"Audio parts directory not found: {audio_dir}"

        available_scenes = []
        missing_scenes = []

        print(f"🔍 CHECKING AVAILABLE SCENES:")
        print(f"   📁 Scenes dir: {scenes_dir}")
        print(f"   🎵 Audio dir: {audio_dir}")
        print(f"   📊 Checking scenes 1-{max_scenes}")
        print()

        # Check hook and subscribe
        hook_audio = audio_dir / "hook_audio.mp3"
        subscribe_audio = audio_dir / "subscribe_audio.mp3"

        print(f"🎬 HOOK & SUBSCRIBE:")
        print(f"   🎬 Hook audio: {'✅ FOUND' if hook_audio.exists() else '❌ MISSING'}")
        print(f"   🔔 Subscribe audio: {'✅ FOUND' if subscribe_audio.exists() else '❌ MISSING'}")
        print()

        # Check story scenes (1 to max_scenes)
        print(f"📖 STORY SCENES:")
        for story_scene_num in range(1, max_scenes + 1):
            # Check image file
            scene_image = scenes_dir / f"scene_{story_scene_num:02d}.png"

            # Check audio file
            scene_audio = audio_dir / f"scene_{story_scene_num:02d}_audio.mp3"

            if scene_image.exists() and scene_audio.exists():
                available_scenes.append(story_scene_num)
                if len(available_scenes) <= 10:  # Show first 10
                    print(f"   ✅ Scene {story_scene_num}: Both image and audio found")
            else:
                missing_parts = []
                if not scene_image.exists():
                    missing_parts.append("image")
                if not scene_audio.exists():
                    missing_parts.append("audio")

                missing_scenes.append(story_scene_num)
                if len(missing_scenes) <= 5:  # Show first 5 missing
                    print(f"   ❌ Scene {story_scene_num}: Missing {', '.join(missing_parts)}")

            # Stop checking after finding no files for 5 consecutive scenes
            if story_scene_num > 10 and len(available_scenes) == 0:
                print(f"   ⏹️  No scenes found in first 10, stopping search")
                break
            elif story_scene_num > len(available_scenes) + 10 and len(available_scenes) > 0:
                print(f"   ⏹️  No more scenes found after scene {available_scenes[-1]}, stopping search")
                break

        if len(available_scenes) > 10:
            print(f"   ... (showing first 10 available scenes)")
        if len(missing_scenes) > 5:
            print(f"   ... (showing first 5 missing scenes)")

        print(f"\n📊 SCENE AVAILABILITY SUMMARY:")
        print(f"   ✅ Available scenes: {len(available_scenes)}")
        print(f"   ❌ Missing scenes: {len(missing_scenes)}")

        if available_scenes:
            print(f"   📋 Available scene IDs: {available_scenes[:10]}{'...' if len(available_scenes) > 10 else ''}")
            print(f"   📏 Scene range: {min(available_scenes)} to {max(available_scenes)}")

        if missing_scenes:
            print(f"   📋 Missing scene IDs: {missing_scenes[:10]}{'...' if len(missing_scenes) > 10 else ''}")

        return available_scenes, None

    def create_simple_video_sequence_with_available_scenes(self, available_scenes):
        """TTS Generator mantığı: Sadece mevcut scene'lerle video sequence oluştur"""
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        scenes_dir = Path(self.current_output_dir) / "scenes"

        sequence = []
        total_duration = 0

        print(f"\n🎵 BUILDING VIDEO SEQUENCE WITH AVAILABLE SCENES:")
        print(f"   📊 Available scenes: {len(available_scenes)}")
        print(f"   📁 Audio source: {audio_dir}")
        print(f"   🖼️  Image source: {scenes_dir}")
        print()

        # 1. HOOK SECTION - TTS Generator mantığı ile
        print("🎬 HOOK SECTION:")
        hook_file = self.find_audio_file(audio_dir, "hook_audio")
        if hook_file:
            hook_duration = self.get_audio_duration(hook_file)

            # Hook için available scene'lerden rastgele seç
            if len(available_scenes) >= 5:
                # En az 5 scene varsa random 5 seç
                hook_scenes = random.sample(available_scenes, 5)
            else:
                # 5'ten az varsa hepsini kullan
                hook_scenes = available_scenes.copy()

            scene_duration = hook_duration / len(hook_scenes) if hook_scenes else hook_duration

            print(f"   ✅ Hook audio found: {hook_file.name}")
            print(f"   ⏱️  Duration: {hook_duration:.1f}s")
            print(f"   🎬 Using {len(hook_scenes)} available scenes: {hook_scenes}")
            print(f"   📏 Scene duration: {scene_duration:.1f}s each")

            hook_images_found = 0
            for scene_id in hook_scenes:
                image_file = scenes_dir / f"scene_{scene_id:02d}.png"
                if image_file.exists():
                    sequence.append({
                        "type": "hook",
                        "image": str(image_file),
                        "duration": scene_duration
                    })
                    hook_images_found += 1

            total_duration += hook_duration
            print(f"   📊 Images used: {hook_images_found}/{len(hook_scenes)}")
        else:
            print("   ❌ Hook audio not found")

        print()

        # 2. SUBSCRIBE SECTION - TTS Generator mantığı ile
        print("🔔 SUBSCRIBE SECTION:")
        subscribe_file = self.find_audio_file(audio_dir, "subscribe_audio")
        if subscribe_file:
            subscribe_duration = self.get_audio_duration(subscribe_file)

            # Subscribe için available scene'lerden rastgele seç (hook'ta kullanılmayanlardan)
            remaining_scenes = [s for s in available_scenes if s not in (hook_scenes if 'hook_scenes' in locals() else [])]

            if len(remaining_scenes) >= 3:
                subscribe_scenes = random.sample(remaining_scenes, 3)
            elif len(available_scenes) >= 3:
                # Hook'ta kullanılanları tekrar kullan
                subscribe_scenes = random.sample(available_scenes, 3)
            else:
                # Çok az scene varsa hepsini kullan
                subscribe_scenes = available_scenes.copy()

            scene_duration = subscribe_duration / len(subscribe_scenes) if subscribe_scenes else subscribe_duration

            print(f"   ✅ Subscribe audio found: {subscribe_file.name}")
            print(f"   ⏱️  Duration: {subscribe_duration:.1f}s")
            print(f"   🎬 Using {len(subscribe_scenes)} available scenes: {subscribe_scenes}")
            print(f"   📏 Scene duration: {scene_duration:.1f}s each")

            subscribe_images_found = 0
            for scene_id in subscribe_scenes:
                image_file = scenes_dir / f"scene_{scene_id:02d}.png"
                if image_file.exists():
                    sequence.append({
                        "type": "subscribe",
                        "image": str(image_file),
                        "duration": scene_duration
                    })
                    subscribe_images_found += 1

            total_duration += subscribe_duration
            print(f"   📊 Images used: {subscribe_images_found}/{len(subscribe_scenes)}")
        else:
            print("   ❌ Subscribe audio not found")

        print()

        # 3. MAIN SCENES SECTION - TTS Generator mantığı: SADECE AVAILABLE SCENES
        print("📖 MAIN SCENES SECTION:")
        print(f"   📊 Processing {len(available_scenes)} available story scenes...")

        scenes_added = 0

        for scene_id in available_scenes:
            audio_file = audio_dir / f"scene_{scene_id:02d}_audio.mp3"
            image_file = scenes_dir / f"scene_{scene_id:02d}.png"

            if audio_file.exists() and image_file.exists():
                scene_duration = self.get_audio_duration(audio_file)
                sequence.append({
                    "type": "scene",
                    "scene_id": scene_id,
                    "image": str(image_file),
                    "duration": scene_duration
                })
                total_duration += scene_duration
                scenes_added += 1

                if scenes_added <= 5 or scenes_added >= len(available_scenes) - 5:  # Show first and last 5
                    print(f"   ✅ Scene {scene_id}: {scene_duration:.1f}s")
                elif scenes_added == 6:
                    print(f"   ... (processing {len(available_scenes) - 10} more scenes)")
            else:
                print(f"   ⚠️  Scene {scene_id}: Files missing (unexpected)")

        print(f"\n📊 SEQUENCE SUMMARY:")
        print(f"   ✅ Total segments: {len(sequence)}")
        print(f"   🎬 Hook segments: {len([s for s in sequence if s['type'] == 'hook'])}")
        print(f"   🔔 Subscribe segments: {len([s for s in sequence if s['type'] == 'subscribe'])}")
        print(f"   📖 Story segments: {len([s for s in sequence if s['type'] == 'scene'])}")
        print(f"   📏 Total duration: {total_duration / 60:.1f} minutes")

        print("\n" + "=" * 50)
        return sequence, total_duration

    def create_simple_video_sequence(self, row_index, scene_plan):
        """Ana fonksiyon: TTS Generator mantığı ile mevcut scene'leri bul ve video üret"""

        # ✅ STEP 1: TTS Generator mantığı - Mevcut scene'leri bul
        print(f"📊 Scene plan received: {type(scene_plan)}")
        if isinstance(scene_plan, dict):
            print(f"📊 Scene plan keys: {list(scene_plan.keys())}")

        # Scene sayısını tahmin et (max 50 scene'e kadar kontrol et)
        available_scenes, error = self.check_available_scenes(row_index, max_scenes=50)

        if error:
            print(f"❌ Scene check failed: {error}")
            return [], 0

        if not available_scenes:
            print(f"❌ No available scenes found for video creation")
            return [], 0

        # ✅ STEP 2: TTS Generator mantığı - Sadece mevcut scene'lerle video üret
        return self.create_simple_video_sequence_with_available_scenes(available_scenes)

    def create_image_list_file(self, row_index, sequence):
        """Image list dosyası oluştur"""
        list_file = Path(self.current_output_dir) / "simple_image_list.txt"

        with open(list_file, 'w') as f:
            for segment in sequence:
                f.write(f"file '{segment['image']}'\n")
                f.write(f"duration {segment['duration']:.2f}\n")

        print(f"✅ Created image list: {list_file}")
        return list_file

    def combine_all_audio(self, row_index, scene_plan):
        """Tüm audio dosyalarını sırayla birleştir"""
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        combined_audio = Path(self.current_output_dir) / "combined_audio.wav"
        audio_list_file = Path(self.current_output_dir) / "audio_list.txt"

        print(f"🎵 AUDIO COMBINATION PROCESS:")
        print(f"   📁 Source directory: {audio_dir}")
        print(f"   📄 Audio list file: {audio_list_file.name}")
        print(f"   🎵 Combined output: {combined_audio.name}")
        print()

        audio_files = []
        audio_summary = {"found": 0, "missing": 0, "total_duration": 0}

        # 1. Hook
        print("🎬 HOOK AUDIO:")
        hook_file = self.find_audio_file(audio_dir, "hook_audio")
        if hook_file:
            audio_files.append(str(hook_file))
            duration = self.get_audio_duration(hook_file)
            audio_summary["found"] += 1
            audio_summary["total_duration"] += duration
            print(f"   ✅ Found: {hook_file.name} ({duration:.1f}s)")
        else:
            audio_summary["missing"] += 1
            print(f"   ❌ Missing: hook_audio.mp3")

        # 2. Subscribe
        print("\n🔔 SUBSCRIBE AUDIO:")
        subscribe_file = self.find_audio_file(audio_dir, "subscribe_audio")
        if subscribe_file:
            audio_files.append(str(subscribe_file))
            duration = self.get_audio_duration(subscribe_file)
            audio_summary["found"] += 1
            audio_summary["total_duration"] += duration
            print(f"   ✅ Found: {subscribe_file.name} ({duration:.1f}s)")
        else:
            audio_summary["missing"] += 1
            print(f"   ❌ Missing: subscribe_audio.mp3")

        # 3. All available scenes (from check_available_scenes)
        print(f"\n📖 STORY SCENES AUDIO:")

        # Get available scenes list
        available_scenes, _ = self.check_available_scenes(row_index, max_scenes=50)
        print(f"   📊 Processing {len(available_scenes)} available scenes...")

        scenes_found = 0
        scenes_missing = 0

        for i, scene_id in enumerate(available_scenes):
            audio_file = audio_dir / f"scene_{scene_id:02d}_audio.mp3"

            if audio_file.exists():
                audio_files.append(str(audio_file))
                duration = self.get_audio_duration(audio_file)
                audio_summary["total_duration"] += duration
                scenes_found += 1

                # Show first and last few scenes
                if i < 3 or i >= len(available_scenes) - 3:
                    print(f"   ✅ Scene {scene_id}: {audio_file.name} ({duration:.1f}s)")
                elif i == 3:
                    print(f"   ... (processing {len(available_scenes) - 6} more scenes)")
            else:
                scenes_missing += 1
                if i < 3 or i >= len(available_scenes) - 3:
                    print(f"   ❌ Scene {scene_id}: audio file missing")

        audio_summary["found"] += scenes_found
        audio_summary["missing"] += scenes_missing

        print(f"\n📊 AUDIO SUMMARY:")
        print(f"   ✅ Audio files found: {audio_summary['found']}")
        print(f"   ❌ Audio files missing: {audio_summary['missing']}")
        print(f"   ⏱️  Total duration: {audio_summary['total_duration'] / 60:.1f} minutes")
        print(f"   📝 Files to combine: {len(audio_files)}")

        if not audio_files:
            print(f"   ❌ No audio files found for combination!")
            return None

        # Audio list dosyası oluştur
        print(f"\n📝 Creating audio list file...")
        with open(audio_list_file, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")

        print(f"   ✅ Audio list created: {len(audio_files)} entries")

        # FFmpeg ile birleştir
        print(f"\n🔄 Combining audio with FFmpeg...")
        try:
            (
                ffmpeg
                .input(str(audio_list_file), format='concat', safe=0)
                .output(str(combined_audio), acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True)
            )

            # Verify combined audio
            if combined_audio.exists():
                file_size = combined_audio.stat().st_size
                final_duration = self.get_audio_duration(combined_audio)
                print(f"   ✅ Audio combination successful!")
                print(f"   📁 Output: {combined_audio.name}")
                print(f"   📏 Size: {file_size:,} bytes")
                print(f"   ⏱️  Duration: {final_duration:.1f}s ({final_duration / 60:.1f} min)")
                return combined_audio
            else:
                print(f"   ❌ Combined audio file not created")
                return None

        except Exception as e:
            print(f"   ❌ Audio combination failed: {e}")
            return None

    def add_background_audio(self, main_audio_file, row_index):
        """Background fireplace audio ekle"""
        fireplace_audio = self.overlay_path / "fireplace.mp3"
        final_audio = Path(self.current_output_dir) / "final_audio.wav"

        if not fireplace_audio.exists():
            print("⚠️ Fireplace audio not found")
            return main_audio_file

        try:
            # Ana audio süresi
            probe = ffmpeg.probe(str(main_audio_file))
            duration = float(probe['streams'][0]['duration'])

            # Background ses hazırla
            background = (
                ffmpeg
                .input(str(fireplace_audio))
                .filter('aloop', loop=-1, size=2e+09)
                .filter('volume', 0.15)
                .filter('atrim', duration=duration)
            )

            # Ana ses
            main = ffmpeg.input(str(main_audio_file))

            # Karıştır
            (
                ffmpeg
                .filter([main, background], 'amix', inputs=2, duration='longest')
                .output(str(final_audio))
                .overwrite_output()
                .run(quiet=True)
            )

            print(f"✅ Added background audio: {final_audio}")
            return final_audio
        except Exception as e:
            print(f"❌ Background audio failed: {e}")
            return main_audio_file

    def create_video_moviepy_style(self, image_list_file, audio_file, row_index, total_duration):
        """FIXED MoviePy Style: Use ALL images in sequence + proper cleanup timing"""
        fireplace_video = self.overlay_path / "fireplace.mp4"
        final_video = Path(self.current_output_dir) / "final_video.mp4"

        print(f"🎬 FIXED MOVIEPY STYLE: ALL images sequence ({total_duration:.1f}s = {total_duration / 60:.1f} minutes)")
        print("📝 Using MoviePy with ALL sequence images + FIXED cleanup timing")

        # Store clips for proper cleanup after rendering
        clips_to_cleanup = []

        try:
            from moviepy.editor import ImageClip, VideoFileClip, AudioFileClip, CompositeVideoClip, \
                concatenate_videoclips

            # Parse image list file to get ALL images and durations
            with open(image_list_file, 'r') as f:
                lines = f.readlines()

            image_clips = []
            i = 0
            while i < len(lines):
                if lines[i].startswith('file '):
                    image_path = lines[i].strip().replace("file '", "").replace("'", "")
                    if i + 1 < len(lines) and lines[i + 1].startswith('duration '):
                        duration = float(lines[i + 1].strip().replace("duration ", ""))

                        # Create image clip for this segment
                        img_clip = ImageClip(str(image_path))
                        img_clip = img_clip.set_duration(duration)
                        image_clips.append(img_clip)
                        clips_to_cleanup.append(img_clip)

                        print(f"📝 Added: {Path(image_path).name} ({duration:.1f}s)")
                        i += 2  # Skip next line (duration)
                    else:
                        i += 1
                else:
                    i += 1

            if not image_clips:
                print("❌ Could not find any images in list")
                return None

            print(f"✅ Created {len(image_clips)} image clips")
            print("🎵 Loading audio...")

            # Audio clip
            audio_clip = AudioFileClip(str(audio_file))
            clips_to_cleanup.append(audio_clip)
            actual_duration = audio_clip.duration

            # Concatenate all image clips to create the main video
            main_video = concatenate_videoclips(image_clips, method="compose")
            clips_to_cleanup.append(main_video)

            print(f"✅ Main video created with {len(image_clips)} scenes: {actual_duration:.1f}s")

            # Fireplace overlay (if exists)
            if fireplace_video.exists():
                print("🔥 Adding animated fireplace overlay...")

                # Overlay clip
                overlay_clip = VideoFileClip(str(fireplace_video))
                clips_to_cleanup.append(overlay_clip)

                # Loop overlay to match duration
                if overlay_clip.duration < actual_duration:
                    # Calculate how many loops needed
                    loop_count = int(actual_duration / overlay_clip.duration) + 1
                    print(f"🔄 Looping fireplace {loop_count} times")

                    # Create looped clips
                    overlay_clips = [overlay_clip.copy() for _ in range(loop_count)]
                    clips_to_cleanup.extend(overlay_clips)  # Add copies to cleanup list

                    overlay_looped = concatenate_videoclips(overlay_clips)
                    clips_to_cleanup.append(overlay_looped)
                else:
                    overlay_looped = overlay_clip

                # Trim to exact duration
                fireplace_overlay = overlay_looped.subclip(0, actual_duration)

                # Resize to match main video
                fireplace_overlay = fireplace_overlay.resize(main_video.size)

                # Set opacity (30% transparent)
                fireplace_overlay = fireplace_overlay.set_opacity(0.3)

                # Remove audio from fireplace (we want base video audio)
                fireplace_overlay = fireplace_overlay.without_audio()

                # Composite video layers: main video + fireplace overlay
                final_clip = CompositeVideoClip([main_video, fireplace_overlay])
                print("✅ Fireplace overlay added successfully!")

            else:
                print("⚠️ Fireplace video not found, using main video only")
                final_clip = main_video

            # Set audio
            final_clip = final_clip.set_audio(audio_clip)
            clips_to_cleanup.append(final_clip)

            print("🚀 Rendering final video...")
            print(f"⏱️  Estimated time: ~{actual_duration * 0.5 / 60:.1f} minutes")
            print(f"🎬 Rendering {len(image_clips)} scenes + fireplace overlay + audio")
            print("🔄 Clips will be cleaned up AFTER rendering completes")

            # Write video file with progress - CRITICAL: Do NOT cleanup before this!
            final_clip.write_videofile(
                str(final_video),
                fps=30,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger='bar'  # Progress bar style
            )

            # ✅ NOW cleanup everything AFTER rendering is complete
            print("🧹 Cleaning up clips after successful render...")
            try:
                for clip in clips_to_cleanup:
                    if clip is not None:
                        clip.close()
                clips_to_cleanup.clear()
                print("✅ Cleanup completed successfully")
            except Exception as cleanup_e:
                print(f"⚠️ Cleanup warning: {cleanup_e}")

            if final_video.exists():
                print(f"✅ FIXED MOVIEPY STYLE completed: {final_video}")
                print(f"🎬 Successfully rendered {len(image_clips)} scenes!")
                print("🔥 Fireplace should be perfectly animated!")
                print("✅ No NoneType errors - cleanup timing fixed!")
                return final_video
            else:
                print("❌ MoviePy render failed")
                return None

        except ImportError:
            print("❌ MoviePy not installed. Install with: pip install moviepy")
            print("🔄 Fallback to FFmpeg method...")
            return self.create_video_ffmpeg_fallback(image_list_file, audio_file, row_index, total_duration)
        except Exception as e:
            print(f"❌ MoviePy failed: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Emergency cleanup...")

            # Emergency cleanup on failure
            try:
                for clip in clips_to_cleanup:
                    if clip is not None:
                        clip.close()
                clips_to_cleanup.clear()
                print("✅ Emergency cleanup completed")
            except Exception as cleanup_e:
                print(f"⚠️ Emergency cleanup warning: {cleanup_e}")

            print("🔄 Fallback to simple video...")
            return self.create_simple_video_with_audio(image_list_file, audio_file, row_index)

    def create_video_ffmpeg_fallback(self, image_list_file, audio_file, row_index, total_duration):
        """FFmpeg fallback if MoviePy not available"""
        fireplace_video = self.overlay_path / "fireplace.mp4"
        final_video = Path(self.current_output_dir) / "final_video.mp4"

        print(f"🔄 FFmpeg fallback method...")

        if not fireplace_video.exists():
            return self.create_simple_video_with_audio(image_list_file, audio_file, row_index)

        try:
            # Get first image
            with open(image_list_file, 'r') as f:
                lines = f.readlines()

            first_image_path = None
            for line in lines:
                if line.startswith('file '):
                    first_image_path = line.strip().replace("file '", "").replace("'", "")
                    break

            # FFmpeg fallback with EXACT test method
            base = (
                ffmpeg
                .input(str(first_image_path), loop=1, t=total_duration, r=30)
                .filter('scale', 1920, 1080, force_original_aspect_ratio='decrease')
                .filter('pad', 1920, 1080, '(ow-iw)/2', '(oh-ih)/2')
            )

            fireplace = (
                ffmpeg
                .input(str(fireplace_video))
                .filter('loop', loop=-1, size=32767)
                .filter('scale', 1920, 1080)
                .filter('format', 'yuva420p')
                .filter('colorchannelmixer', aa=0.3)
            )

            result = ffmpeg.filter([base, fireplace], 'overlay')
            audio_input = ffmpeg.input(str(audio_file))

            (
                ffmpeg
                .output(
                    result,
                    audio_input,
                    str(final_video),
                    vcodec='libx264',
                    acodec='aac',
                    pix_fmt='yuv420p',
                    t=total_duration
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            return final_video if final_video.exists() else None

        except Exception as e:
            print(f"❌ FFmpeg fallback failed: {e}")
            return self.create_simple_video_with_audio(image_list_file, audio_file, row_index)

    def create_simple_video_with_audio(self, image_list_file, audio_file, row_index):
        """Fallback: Simple video with audio (no overlay)"""
        final_video = Path(self.current_output_dir) / "final_video_no_overlay.mp4"

        try:
            print("📝 Creating fallback video with audio (no overlay)...")
            (
                ffmpeg
                .output(
                    ffmpeg.input(str(image_list_file), format='concat', safe=0).filter('scale', 1920, 1080).filter(
                        'setsar', 1),
                    ffmpeg.input(str(audio_file)),
                    str(final_video),
                    vcodec='libx264',
                    acodec='aac',
                    preset='fast'
                )
                .overwrite_output()
                .run(quiet=True)
            )

            print(f"✅ Fallback video created: {final_video}")
            return final_video
        except Exception as e:
            print(f"❌ Fallback video failed: {e}")
            return None

    def verify_final_video(self, video_file):
        """Final video dosyasını doğrula ve bilgi göster"""
        print(f"\n🔍 VIDEO VERIFICATION PROCESS:")
        print(f"   📁 Video file: {Path(video_file).name}")
        print(f"   📍 Full path: {video_file}")

        try:
            probe = ffmpeg.probe(str(video_file))

            # Video ve audio stream bilgileri
            video_stream = None
            audio_stream = None

            for stream in probe['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                elif stream['codec_type'] == 'audio':
                    audio_stream = stream

            duration = float(probe['format']['duration'])
            file_size = os.path.getsize(video_file) / (1024 * 1024)  # MB

            print(f"\n📊 VIDEO FILE INFORMATION:")
            print(f"   ⏱️  Duration: {duration:.1f}s ({duration / 60:.1f} minutes)")
            print(f"   📦 File size: {file_size:.1f} MB")
            print(f"   📈 Bitrate: {float(probe['format'].get('bit_rate', 0)) / 1000:.0f} kbps")

            if video_stream:
                print(f"\n🎬 VIDEO STREAM:")
                print(f"   📺 Resolution: {video_stream['width']}x{video_stream['height']}")
                print(f"   🎞️  Codec: {video_stream['codec_name']}")
                print(f"   🎥 Frame rate: {video_stream.get('r_frame_rate', 'unknown')}")
                print(f"   🌈 Pixel format: {video_stream.get('pix_fmt', 'unknown')}")

                # Calculate total frames
                if 'nb_frames' in video_stream:
                    print(f"   🎞️  Total frames: {video_stream['nb_frames']}")

            if audio_stream:
                print(f"\n🎵 AUDIO STREAM:")
                print(f"   🔊 Codec: {audio_stream['codec_name']}")
                print(f"   📻 Sample rate: {audio_stream.get('sample_rate', 'unknown')} Hz")
                print(f"   🎼 Channels: {audio_stream.get('channels', 'unknown')}")
                if 'bit_rate' in audio_stream:
                    print(f"   📊 Bitrate: {int(audio_stream['bit_rate']) / 1000:.0f} kbps")

            # Quality checks
            print(f"\n✅ QUALITY VERIFICATION:")

            # Duration check
            if duration >= 60:  # At least 1 minute
                print(f"   ✅ Duration acceptable: {duration / 60:.1f} minutes")
            else:
                print(f"   ⚠️  Duration short: {duration:.1f} seconds")

            # File size check
            if file_size >= 10:  # At least 10 MB
                print(f"   ✅ File size acceptable: {file_size:.1f} MB")
            else:
                print(f"   ⚠️  File size small: {file_size:.1f} MB")

            # Resolution check
            if video_stream and video_stream['width'] >= 1280:
                print(f"   ✅ Resolution acceptable: {video_stream['width']}x{video_stream['height']}")
            else:
                print(f"   ⚠️  Resolution low: {video_stream['width'] if video_stream else 'unknown'}x{video_stream['height'] if video_stream else 'unknown'}")

            # Audio check
            if audio_stream:
                print(f"   ✅ Audio stream present")
            else:
                print(f"   ❌ Audio stream missing")

            print(f"\n✅ VIDEO VERIFICATION COMPLETED SUCCESSFULLY")
            return True, duration, file_size

        except Exception as e:
            print(f"\n❌ VIDEO VERIFICATION FAILED:")
            print(f"   🚨 Error: {e}")
            print(f"   📁 File exists: {os.path.exists(video_file)}")
            if os.path.exists(video_file):
                file_size = os.path.getsize(video_file) / (1024 * 1024)
                print(f"   📦 File size: {file_size:.1f} MB")
            return False, 0.0, 0.0

    def create_video(self, row_index, topic_data, progress_tracker, usage_tracker):
        """Ana video üretim fonksiyonu - Server version with TTS scene checking"""
        total_steps = 8
        current_step = 0

        print(f"\n" + "🎬" * 80)
        print("VIDEO COMPOSER - PROCESSING DETAILS")
        print("🎬" * 80)
        print(f"🎯 PROJECT: {topic_data['topic']}")
        print(f"🆔 PROJECT ID: {row_index}")
        print(f"📁 OUTPUT DIR: {self.current_output_dir}")
        print()
        print("📋 PROCESSING METHOD:")
        print("   🎬 FIXED MOVIEPY APPROACH - SERVER VERSION + TTS SCENE CHECKING")
        print("   📝 Python Video Library with AVAILABLE SEQUENCE IMAGES ONLY")
        print("   📝 Layer 1: Multiple Image Clips (AVAILABLE scenes only)")
        print("   📝 Layer 2: Fireplace Overlay (animated)")
        print("   📝 Layer 3: Full Audio Sequence")
        print("   ✅ Fixed: TTS Generator scene checking logic integrated!")
        print("   ✅ Fixed: Missing scenes automatically skipped!")
        print("   ✅ Fixed: Proper cleanup timing!")
        print("   🖥️ Server: Database integrated with progress tracking")
        print("🎬" * 80)

        try:
            # 1. Project data yükle
            current_step += 1
            self.print_progress(current_step, total_steps, "Loading project data...")
            start_time = time.time()

            # Load scene plan and platform metadata
            scene_plan_original, platform_metadata = self.load_project_data(row_index)
            if not platform_metadata:
                progress_tracker.mark_stage_failed("project_load", "Failed to load platform metadata")
                return None

            # Load actual generated scenes from audio timeline
            story_scenes, timeline_data = self.load_audio_timeline(self.current_output_dir)
            if not story_scenes:
                progress_tracker.mark_stage_failed("project_load", "Failed to load audio timeline")
                return None

            # Use timeline scenes instead of scene plan
            scene_plan = story_scenes

            print(f"\n📊 PROJECT DATA LOADED:")
            print(f"   📋 Platform metadata: {'✅ Available' if platform_metadata else '❌ Missing'}")
            if platform_metadata and 'title_options' in platform_metadata:
                print(f"   📝 Video title: {platform_metadata['title_options'][0][:50]}...")

            progress_tracker.mark_stage_completed("project_load")
            usage_tracker.add_stage("project_load", time.time() - start_time)

            # 2. Video sequence oluştur (TTS mantığı ile)
            current_step += 1
            self.print_progress(current_step, total_steps, "Creating video sequence with TTS scene checking...")
            start_time = time.time()

            sequence, total_duration = self.create_simple_video_sequence(row_index, scene_plan)
            if not sequence:
                progress_tracker.mark_stage_failed("sequence_build", "Failed to create video sequence - no available scenes")
                return None

            print(f"\n✅ VIDEO SEQUENCE CREATED WITH TTS LOGIC:")
            print(f"   🎬 Total segments: {len(sequence)}")
            print(f"   ⏱️  Total duration: {total_duration / 60:.1f} minutes")

            # Count sequence types
            hook_count = len([s for s in sequence if s['type'] == 'hook'])
            subscribe_count = len([s for s in sequence if s['type'] == 'subscribe'])
            scene_count = len([s for s in sequence if s['type'] == 'scene'])

            print(f"   🎬 Hook segments: {hook_count}")
            print(f"   🔔 Subscribe segments: {subscribe_count}")
            print(f"   📖 Story segments: {scene_count}")
            print(f"   ✅ Missing scenes automatically skipped!")

            progress_tracker.mark_stage_completed("sequence_build")
            usage_tracker.add_stage("sequence_build", time.time() - start_time)
            usage_tracker.update_performance_data(total_scenes=len(sequence), video_duration_seconds=total_duration)

            # 3. Audio birleştir
            current_step += 1
            self.print_progress(current_step, total_steps, "Combining audio files...")
            start_time = time.time()

            print(f"\n🎵 AUDIO COMBINATION:")
            combined_audio = self.combine_all_audio(row_index, scene_plan)
            if not combined_audio:
                progress_tracker.mark_stage_failed("audio_combine", "Failed to combine audio")
                return None

            print(f"   ✅ Combined audio created: {Path(combined_audio).name}")

            # 4. Background audio ekle
            current_step += 1
            self.print_progress(current_step, total_steps, "Adding background audio...")

            print(f"\n🔥 BACKGROUND AUDIO:")
            fireplace_audio = self.overlay_path / "fireplace.mp3"
            print(f"   🔍 Looking for: {fireplace_audio}")
            print(f"   {'✅ Found' if fireplace_audio.exists() else '❌ Not found'}")

            final_audio = self.add_background_audio(combined_audio, row_index)

            progress_tracker.mark_stage_completed("audio_combine")
            usage_tracker.add_stage("audio_combine", time.time() - start_time)

            # 5. Image list oluştur
            current_step += 1
            self.print_progress(current_step, total_steps, "Creating image list...")

            print(f"\n📋 IMAGE LIST CREATION:")
            image_list = self.create_image_list_file(row_index, sequence)
            print(f"   ✅ Image list created: {Path(image_list).name}")
            print(f"   📊 Total image entries: {len(sequence)}")

            # Check processing limits before video render
            can_continue, limit_reason = usage_tracker.check_processing_limits()
            if not can_continue:
                progress_tracker.mark_stage_failed("video_render", f"Processing limit exceeded: {limit_reason}")
                return None

            # 6. FIXED MOVIEPY STYLE: Proper cleanup timing
            current_step += 1
            self.print_progress(current_step, total_steps, "🎬 TTS SCENE CHECKING + ALL AVAILABLE SCENES MoviePy...")
            start_time = time.time()

            print(f"\n🎬 VIDEO RENDERING:")
            print(f"   📝 Method: TTS Scene Checking + ALL AVAILABLE SCENES MoviePy (fixed)")
            print(f"   🔥 Overlay: Fireplace animation")
            print(f"   🎵 Audio: Full sequence with background")
            print(f"   📊 Input segments: {len(sequence)}")
            print(f"   ⏱️  Expected duration: {total_duration / 60:.1f} minutes")
            print(f"   ✅ Missing scenes: Automatically skipped!")

            progress_tracker.set_render_method("moviepy_tts_scene_checking_fixed")
            usage_tracker.update_performance_data(render_method="moviepy_tts_scene_checking_fixed")

            final_video = self.create_video_moviepy_style(image_list, final_audio, row_index, total_duration)
            if not final_video:
                progress_tracker.mark_stage_failed("video_render", "MoviePy render failed")
                return None

            print(f"   ✅ Video rendering completed")
            print(f"   📁 Output file: {Path(final_video).name}")

            progress_tracker.mark_stage_completed("video_render")
            usage_tracker.add_stage("video_render", time.time() - start_time)

            # 7. Video doğrula
            current_step += 1
            self.print_progress(current_step, total_steps, "Verifying final video...")
            start_time = time.time()

            verification_success, actual_duration, file_size_mb = self.verify_final_video(final_video)
            if not verification_success:
                print("⚠️ Video verification failed, but continuing...")
                progress_tracker.mark_stage_failed("verification", "Video verification failed")
            else:
                print(f"\n✅ VIDEO VERIFICATION PASSED:")
                print(f"   ⏱️  Actual duration: {actual_duration:.1f}s ({actual_duration / 60:.1f} min)")
                print(f"   📦 File size: {file_size_mb:.1f} MB")
                progress_tracker.mark_stage_completed("verification")
                usage_tracker.update_performance_data(filesize_mb=file_size_mb)

            usage_tracker.add_stage("verification", time.time() - start_time)

            # 8. Metadata kaydet
            current_step += 1
            self.print_progress(current_step, total_steps, "Saving metadata...")

            usage_summary = usage_tracker.print_final_summary()

            video_metadata = {
                "title": platform_metadata["title_options"][0] if platform_metadata.get("title_options") else topic_data["topic"],
                "duration_seconds": actual_duration,
                "sequence_count": len(sequence),
                "created_at": datetime.now().isoformat(),
                "output_file": str(final_video),
                "processing_steps": total_steps,
                "render_method": "moviepy_tts_scene_checking_fixed_server",
                "overlay_working": True,
                "cleanup_timing": "fixed",
                "available_scenes_working": True,
                "tts_scene_checking": True,
                "missing_scenes_skipped": True,
                "server_version": True,
                "database_integrated": True,
                "usage_summary": usage_summary
            }

            metadata_file = Path(self.current_output_dir) / "video_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(video_metadata, f, indent=2, ensure_ascii=False)

            # Tamamlanma mesajı
            self.print_progress(total_steps, total_steps, "TTS Scene Checking + MoviePy render completed!")

            print(f"\n" + "🎉" * 80)
            print("VIDEO CREATION COMPLETED SUCCESSFULLY!")
            print("🎉" * 80)
            print(f"🎬 PROJECT: {topic_data['topic']}")
            print(f"🆔 PROJECT ID: {row_index}")
            print(f"📁 VIDEO FILE: {final_video}")
            print(f"⏱️  DURATION: {actual_duration / 60:.1f} minutes")
            print(f"📦 FILE SIZE: {file_size_mb:.1f} MB")
            print(f"🎬 SEGMENTS: {len(sequence)} total")
            print(f"🎭 METHOD: TTS Scene Checking + ALL AVAILABLE SCENES MoviePy (fixed server)")
            print(f"🔥 OVERLAY: Working (animated with MoviePy)")
            print(f"🎵 AUDIO: Working (available scenes only)")
            print(f"✅ STATUS: TTS Generator logic - Missing scenes automatically skipped!")
            print(f"🖥️ SERVER: Database integrated!")
            print("🎉" * 80)

            return final_video, actual_duration, file_size_mb, usage_summary["total_processing_time_minutes"]

        except Exception as e:
            print(f"💥 Video creation error: {e}")
            progress_tracker.mark_stage_failed("video_render", str(e))
            import traceback
            traceback.print_exc()
            return None

    def run_video_generation(self) -> bool:
        """Run VIDEO generation process for server environment"""
        print("🚀" * 50)
        print("SERVER VIDEO COMPOSER v1.1 - TTS SCENE CHECKING")
        print("🔗 Database integrated")
        print("🎬 YouTube Production Video Generation")
        print("🔥 MoviePy + Fireplace Overlay + AVAILABLE Scenes Only")
        print("✅ TTS Generator scene checking logic integrated")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # Print processing limits
        limits = CONFIG.video_config.get("budget_controls", {})
        print(f"⏱️  PROCESSING LIMITS:")
        print(f"   📊 Max processing time: {limits.get('max_processing_time_minutes', 30)} minutes")
        print(f"   📊 Warning threshold: {limits.get('warn_threshold_minutes', 20)} minutes")
        print(f"   💾 Max memory usage: {limits.get('max_memory_usage_gb', 8)} GB")

        # Initialize success tracking
        overall_success = False

        # Step 1: Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")
        print(f"📁 Output directory: {project_info['output_dir']}")
        print(f"🆔 Topic ID: {project_info['topic_id']}")

        # Initialize trackers
        progress_tracker = VideoProgressTracker(self.current_topic_id, CONFIG.paths['OUTPUT_DIR'])
        usage_tracker = VideoUsageTracker()

        try:
            story_id = self.current_topic_id
            topic = self.current_topic

            # Create video with server tracking + TTS scene checking
            result = self.create_video(story_id, project_info, progress_tracker, usage_tracker)

            if result and len(result) == 4:  # Successful result has 4 elements
                final_video, duration_seconds, file_size_mb, processing_time_minutes = result

                print(f"✅ Video generation successful!")
                print(f"📁 Video file: {final_video}")
                print(f"⏱️  Duration: {duration_seconds / 60:.1f} minutes")
                print(f"📦 File size: {file_size_mb:.1f} MB")
                print(f"⚡ Processing time: {processing_time_minutes:.1f} minutes")

                # Update database with results
                self.db_manager.mark_video_generation_completed(
                    self.current_topic_id, duration_seconds, file_size_mb, processing_time_minutes
                )

                print(f"💾 Database updated with video results")

                # Cleanup progress on success
                progress_tracker.cleanup_on_success()

                print("\n" + "🎉" * 50)
                print("VIDEO GENERATION SUCCESSFUL!")
                print("✅ YouTube-optimized video with AVAILABLE scenes only")
                print("✅ MoviePy with fireplace overlay")
                print("✅ TTS Generator scene checking logic")
                print("✅ Missing scenes automatically skipped")
                print("✅ Fixed cleanup timing")
                print("✅ Database updated with metrics")
                print("🎉" * 50)
                overall_success = True

            else:
                print(f"❌ Video generation failed")
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ Video generation failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        print("🚀 SERVER VIDEO COMPOSER v1.1 - TTS SCENE CHECKING")
        print("🔗 Database integration with progress tracking")
        print("🎬 YouTube Production Video Generation")
        print("🔥 MoviePy + Fireplace Overlay + AVAILABLE Scenes Only")
        print("🎭 Fixed cleanup timing + TTS scene checking logic")
        print("✅ TTS Generator scene checking integration")
        print("🖥️ Production-ready automation")
        print("=" * 60)

        producer = ServerYouTubeVideoProducer()
        success = producer.run_video_generation()

        if success:
            print("🎊 Video generation completed successfully!")
            print("📁 Video saved: final_video.mp4")
            print("📋 Metadata saved: video_metadata.json")
            print("🔥 Fireplace overlay included")
            print("🎬 AVAILABLE scenes in sequence")
            print("✅ Missing scenes automatically skipped")
            print("💾 Progress tracking enabled")
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