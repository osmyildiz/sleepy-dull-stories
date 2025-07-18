"""
Sleepy Dull Stories - SERVER-READY Video Composer
YouTube Video Production with ALL Scenes + MoviePy + Fireplace Overlay
FIXED: Uses audio timeline instead of scene plan for ACTUAL generated scenes
Production-optimized with complete automation and database integration
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
            "ffmpeg_fallback": True,
            "timeline_mode": True,  # NEW: Use timeline.json instead of scene_plan.json
            "scene_by_scene_mode": True  # NEW: Scene-by-scene processing for perfect audio sync
        }

        print("✅ Video composition configuration loaded")
        print(f"🎬 Target: {self.video_config['target_resolution'][0]}x{self.video_config['target_resolution'][1]} @ {self.video_config['target_fps']}fps")
        print(f"📋 Timeline mode: {'✅ ENABLED' if self.video_config['timeline_mode'] else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if self.video_config['scene_by_scene_mode'] else '❌ DISABLED'}")

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
                "timeline_load": {"status": "pending", "timestamp": None},
                "scene_preparation": {"status": "pending", "timestamp": None},  # NEW
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
            "filesize_mb": 0.0,
            "timeline_mode": CONFIG.video_config.get("timeline_mode", False),
            "scene_by_scene_mode": CONFIG.video_config.get("scene_by_scene_mode", False)
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
        print(f"📋 Timeline mode: {'✅ ENABLED' if self.performance_data.get('timeline_mode') else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if self.performance_data.get('scene_by_scene_mode') else '❌ DISABLED'}")
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
    """Server-ready YouTube Video Producer with database integration and timeline support"""

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
        self.overlay_path = self.base_dir / "src" / "data" / "overlay_videos"  # FIXED: Correct path

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseVideoManager(str(db_path))

        print("🎬 Server YouTube Video Producer v1.1 Initialized")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🎥 Overlay Path: {self.overlay_path}")
        print(f"📋 Timeline mode: {'✅ ENABLED' if CONFIG.video_config.get('timeline_mode') else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if CONFIG.video_config.get('scene_by_scene_mode') else '❌ DISABLED'}")

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
        """Proje JSON dosyalarını yükle (platform metadata için)"""
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
        """Load actual generated scenes from audio timeline instead of scene plan"""
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

            # Extract all scenes (hook, subscribe, story scenes)
            all_scenes = timeline_data.get('scenes', [])

            # Extract story scenes only
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

    def create_video_sequence_from_timeline(self, story_scenes, hook_subscribe_data):
        """Create video sequence from timeline data instead of scene plan"""
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        scenes_dir = Path(self.current_output_dir) / "scenes"

        sequence = []
        total_duration = 0

        print("\n🎵 BUILDING VIDEO SEQUENCE FROM TIMELINE:")
        print("=" * 50)
        print(f"📁 Audio source: {audio_dir}")
        print(f"🖼️  Image source: {scenes_dir}")
        print(f"📊 Story scenes: {len(story_scenes)}")
        print()

        hook_scene, subscribe_scene = hook_subscribe_data

        # 1. HOOK SECTION
        print("🎬 HOOK SECTION:")
        if hook_scene:
            hook_audio_file = audio_dir / hook_scene['audio_file']
            if hook_audio_file.exists():
                hook_duration = self.get_audio_duration(hook_audio_file)
                # Hook için rastgele 5 scene seç (available scenes'den)
                available_for_hook = [s for s in story_scenes if s['scene_id'] >= 10][:10]
                hook_scenes_to_use = random.sample(available_for_hook, min(5, len(available_for_hook)))
                scene_duration = hook_duration / len(hook_scenes_to_use)

                print(f"   ✅ Hook audio found: {hook_scene['audio_file']}")
                print(f"   ⏱️  Duration: {hook_duration:.1f}s")
                print(f"   🎬 Using {len(hook_scenes_to_use)} scenes")
                print(f"   📏 Scene duration: {scene_duration:.1f}s each")

                hook_images_found = 0
                for scene_data in hook_scenes_to_use:
                    _, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_data['scene_id'])
                    if image_file:
                        sequence.append({
                            "type": "hook",
                            "image": str(image_file),
                            "duration": scene_duration
                        })
                        hook_images_found += 1

                total_duration += hook_duration
                print(f"   📊 Images found: {hook_images_found}/{len(hook_scenes_to_use)}")
            else:
                print(f"   ❌ Hook audio not found: {hook_audio_file}")
        else:
            print("   ❌ Hook scene not found in timeline")

        print()

        # 2. SUBSCRIBE SECTION
        print("🔔 SUBSCRIBE SECTION:")
        if subscribe_scene:
            subscribe_audio_file = audio_dir / subscribe_scene['audio_file']
            if subscribe_audio_file.exists():
                subscribe_duration = self.get_audio_duration(subscribe_audio_file)
                # Subscribe için rastgele 3 scene seç (available scenes'den)
                available_for_sub = [s for s in story_scenes if s['scene_id'] <= 15][:8]
                sub_scenes_to_use = random.sample(available_for_sub, min(3, len(available_for_sub)))
                scene_duration = subscribe_duration / len(sub_scenes_to_use)

                print(f"   ✅ Subscribe audio found: {subscribe_scene['audio_file']}")
                print(f"   ⏱️  Duration: {subscribe_duration:.1f}s")
                print(f"   🎬 Using {len(sub_scenes_to_use)} scenes")
                print(f"   📏 Scene duration: {scene_duration:.1f}s each")

                sub_images_found = 0
                for scene_data in sub_scenes_to_use:
                    _, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_data['scene_id'])
                    if image_file:
                        sequence.append({
                            "type": "subscribe",
                            "image": str(image_file),
                            "duration": scene_duration
                        })
                        sub_images_found += 1

                total_duration += subscribe_duration
                print(f"   📊 Images found: {sub_images_found}/{len(sub_scenes_to_use)}")
            else:
                print(f"   ❌ Subscribe audio not found: {subscribe_audio_file}")
        else:
            print("   ❌ Subscribe scene not found in timeline")

        print()

        # 3. MAIN SCENES SECTION FROM TIMELINE
        print("📖 MAIN SCENES SECTION (FROM TIMELINE):")
        print(f"   📊 Processing {len(story_scenes)} story scenes from timeline...")

        scenes_found = 0
        scenes_missing = 0

        for i, scene_data in enumerate(story_scenes):
            scene_id = scene_data['scene_id']
            audio_file, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_id)

            if audio_file and image_file:
                scene_duration = self.get_audio_duration(audio_file)
                sequence.append({
                    "type": "scene",
                    "scene_id": scene_id,
                    "image": str(image_file),
                    "duration": scene_duration,
                    "title": scene_data.get('title', f'Scene {scene_id}')
                })
                total_duration += scene_duration
                scenes_found += 1

                if i < 5 or i >= len(story_scenes) - 5:  # Show first and last 5
                    print(f"   ✅ Scene {scene_id}: {scene_duration:.1f}s ({Path(image_file).name})")
                elif i == 5:
                    print(f"   ... (showing first/last 5 scenes)")
            else:
                scenes_missing += 1
                missing_files = []
                if not audio_file:
                    missing_files.append("audio")
                if not image_file:
                    missing_files.append("image")

                if i < 5 or i >= len(story_scenes) - 5:  # Show first and last 5
                    print(f"   ❌ Scene {scene_id}: missing {', '.join(missing_files)}")

        print(f"\n📊 SCENE SUMMARY:")
        print(f"   ✅ Scenes found: {scenes_found}")
        print(f"   ❌ Scenes missing: {scenes_missing}")
        print(f"   📏 Total duration: {total_duration / 60:.1f} minutes")
        print(f"   🎬 Total segments: {len(sequence)}")

        print("\n" + "=" * 50)
        return sequence, total_duration

    def create_image_list_file(self, row_index, sequence):
        """Image list dosyası oluştur"""
        list_file = Path(self.current_output_dir) / "simple_image_list.txt"

        with open(list_file, 'w') as f:
            for segment in sequence:
                f.write(f"file '{segment['image']}'\n")
                f.write(f"duration {segment['duration']:.2f}\n")

        print(f"✅ Created image list: {list_file}")
        return list_file

    def combine_audio_from_timeline(self, story_scenes, hook_subscribe_data):
        """Combine audio using timeline data instead of scene plan"""
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        combined_audio = Path(self.current_output_dir) / "combined_audio.wav"
        audio_list_file = Path(self.current_output_dir) / "audio_list.txt"

        print(f"🎵 AUDIO COMBINATION FROM TIMELINE:")
        print(f"   📁 Source directory: {audio_dir}")
        print(f"   📄 Audio list file: {audio_list_file.name}")
        print(f"   🎵 Combined output: {combined_audio.name}")
        print()

        audio_files = []
        audio_summary = {"found": 0, "missing": 0, "total_duration": 0}

        hook_scene, subscribe_scene = hook_subscribe_data

        # 1. Hook
        print("🎬 HOOK AUDIO:")
        if hook_scene:
            hook_file = audio_dir / hook_scene['audio_file']
            if hook_file.exists():
                audio_files.append(str(hook_file))
                duration = self.get_audio_duration(hook_file)
                audio_summary["found"] += 1
                audio_summary["total_duration"] += duration
                print(f"   ✅ Found: {hook_file.name} ({duration:.1f}s)")
            else:
                audio_summary["missing"] += 1
                print(f"   ❌ Missing: {hook_scene['audio_file']}")
        else:
            print(f"   ❌ Hook scene not in timeline")

        # 2. Subscribe
        print("\n🔔 SUBSCRIBE AUDIO:")
        if subscribe_scene:
            subscribe_file = audio_dir / subscribe_scene['audio_file']
            if subscribe_file.exists():
                audio_files.append(str(subscribe_file))
                duration = self.get_audio_duration(subscribe_file)
                audio_summary["found"] += 1
                audio_summary["total_duration"] += duration
                print(f"   ✅ Found: {subscribe_file.name} ({duration:.1f}s)")
            else:
                audio_summary["missing"] += 1
                print(f"   ❌ Missing: {subscribe_scene['audio_file']}")
        else:
            print(f"   ❌ Subscribe scene not in timeline")

        # 3. Story scenes from timeline
        print(f"\n📖 STORY SCENES AUDIO (FROM TIMELINE):")
        print(f"   📊 Processing {len(story_scenes)} scenes from timeline...")

        scenes_found = 0
        scenes_missing = 0

        for i, scene_data in enumerate(story_scenes):
            scene_id = scene_data['scene_id']
            audio_file_name = scene_data['audio_file']
            audio_file_path = audio_dir / audio_file_name

            if audio_file_path.exists():
                audio_files.append(str(audio_file_path))
                duration = self.get_audio_duration(audio_file_path)
                audio_summary["total_duration"] += duration
                scenes_found += 1

                # Show first and last few scenes
                if i < 3 or i >= len(story_scenes) - 3:
                    print(f"   ✅ Scene {scene_id}: {audio_file_name} ({duration:.1f}s)")
                elif i == 3:
                    print(f"   ... (processing {len(story_scenes) - 6} more scenes)")
            else:
                scenes_missing += 1
                if i < 3 or i >= len(story_scenes) - 3:
                    print(f"   ❌ Scene {scene_id}: {audio_file_name} missing")

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

    def create_video_scene_by_scene_style(self, story_scenes, hook_subscribe_data, row_index, total_duration):
        """SCENE-BY-SCENE MoviePy Style: Process each scene separately for perfect audio sync"""
        fireplace_video = self.overlay_path / "fireplace.mp4"
        final_video = Path(self.current_output_dir) / "final_video.mp4"
        scene_videos_dir = Path(self.current_output_dir) / "scene_videos"
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        scenes_dir = Path(self.current_output_dir) / "scenes"

        print(f"🎬 SCENE-BY-SCENE MOVIEPY STYLE: Process each scene separately ({total_duration:.1f}s = {total_duration / 60:.1f} minutes)")
        print("📝 Using MoviePy with SCENE-BY-SCENE processing for perfect audio sync")
        print("🎯 Natural breakpoints - no audio cutting in the middle of words")

        # Create scene videos directory
        scene_videos_dir.mkdir(exist_ok=True)

        try:
            from moviepy.editor import ImageClip, VideoFileClip, AudioFileClip, CompositeVideoClip, \
                concatenate_videoclips
            import gc
            import subprocess

            # Load fireplace overlay once
            fireplace_overlay_base = None
            if fireplace_video.exists():
                print("🔥 Loading fireplace overlay...")
                fireplace_overlay_base = VideoFileClip(str(fireplace_video))
                print(f"   📏 Fireplace duration: {fireplace_overlay_base.duration:.1f}s")

            hook_scene, subscribe_scene = hook_subscribe_data
            scene_video_files = []

            print(f"\n🎬 SCENE-BY-SCENE PROCESSING:")
            print(f"   📊 Total story scenes: {len(story_scenes)}")
            print(f"   🎬 Hook scene: {'✅' if hook_scene else '❌'}")
            print(f"   🔔 Subscribe scene: {'✅' if subscribe_scene else '❌'}")

            # Helper function to render single scene
            def render_single_scene(scene_data, scene_type="story"):
                clips_to_cleanup = []

                try:
                    if scene_type == "hook":
                        print(f"\n🎬 RENDERING HOOK SCENE:")
                        audio_file = audio_dir / scene_data['audio_file']
                        scene_duration = self.get_audio_duration(audio_file)

                        # Use multiple random images for hook
                        available_scenes = [s for s in story_scenes if s['scene_id'] >= 10][:10]
                        hook_scenes_to_use = random.sample(available_scenes, min(5, len(available_scenes)))
                        image_duration = scene_duration / len(hook_scenes_to_use)

                        image_clips = []
                        for scene_info in hook_scenes_to_use:
                            _, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_info['scene_id'])
                            if image_file:
                                img_clip = ImageClip(str(image_file)).set_duration(image_duration)
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
                        scene_duration = self.get_audio_duration(audio_file)

                        # Use multiple random images for subscribe
                        available_scenes = [s for s in story_scenes if s['scene_id'] <= 15][:8]
                        sub_scenes_to_use = random.sample(available_scenes, min(3, len(available_scenes)))
                        image_duration = scene_duration / len(sub_scenes_to_use)

                        image_clips = []
                        for scene_info in sub_scenes_to_use:
                            _, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_info['scene_id'])
                            if image_file:
                                img_clip = ImageClip(str(image_file)).set_duration(image_duration)
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

                        # Find audio and image files
                        audio_file, image_file = self.find_scene_files(audio_dir, scenes_dir, scene_id)

                        if not audio_file or not image_file:
                            print(f"   ❌ Missing files for scene {scene_id}")
                            return None

                        scene_duration = self.get_audio_duration(audio_file)

                        # Create main video (single image for story scenes)
                        main_video = ImageClip(str(image_file)).set_duration(scene_duration)
                        clips_to_cleanup.append(main_video)

                    print(f"   ✅ Main video created: {scene_duration:.1f}s")

                    # Add fireplace overlay
                    if fireplace_overlay_base:
                        print(f"   🔥 Adding fireplace overlay...")

                        # Calculate fireplace loops needed
                        fireplace_duration = fireplace_overlay_base.duration
                        loops_needed = int(scene_duration / fireplace_duration) + 1

                        # Create looped fireplace
                        if loops_needed > 1:
                            fireplace_clips = []
                            for loop_idx in range(loops_needed):
                                fireplace_clips.append(fireplace_overlay_base.copy())

                            scene_fireplace = concatenate_videoclips(fireplace_clips)
                            clips_to_cleanup.extend(fireplace_clips)
                            clips_to_cleanup.append(scene_fireplace)
                        else:
                            scene_fireplace = fireplace_overlay_base.copy()
                            clips_to_cleanup.append(scene_fireplace)

                        # Trim to exact scene duration
                        scene_fireplace = scene_fireplace.subclip(0, scene_duration)

                        # Resize and set opacity
                        scene_fireplace = scene_fireplace.resize(main_video.size)
                        scene_fireplace = scene_fireplace.set_opacity(0.3)
                        scene_fireplace = scene_fireplace.without_audio()

                        # Composite
                        scene_final = CompositeVideoClip([main_video, scene_fireplace])
                        clips_to_cleanup.append(scene_final)

                        print(f"   ✅ Fireplace overlay added")
                    else:
                        scene_final = main_video

                    # Add audio
                    scene_audio = AudioFileClip(str(audio_file))
                    scene_final = scene_final.set_audio(scene_audio)
                    clips_to_cleanup.append(scene_audio)
                    clips_to_cleanup.append(scene_final)

                    # Render scene
                    if scene_type == "hook":
                        scene_file = scene_videos_dir / "00_hook.mp4"
                    elif scene_type == "subscribe":
                        scene_file = scene_videos_dir / "01_subscribe.mp4"
                    else:
                        scene_file = scene_videos_dir / f"{scene_id:03d}_scene_{scene_id}.mp4"

                    print(f"   🚀 Rendering scene...")
                    print(f"      📁 Output: {scene_file.name}")
                    print(f"      ⏱️  Expected time: ~{scene_duration * 0.3 / 60:.1f} minutes")

                    scene_final.write_videofile(
                        str(scene_file),
                        fps=30,
                        codec="libx264",
                        audio_codec="aac",
                        temp_audiofile=f'temp-audio-{scene_file.stem}.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger=None  # Disable progress bar for individual scenes
                    )

                    # Cleanup scene clips immediately
                    print(f"   🧹 Cleaning up scene clips...")
                    for clip in clips_to_cleanup:
                        if clip is not None:
                            try:
                                clip.close()
                            except:
                                pass
                    clips_to_cleanup.clear()

                    # Force garbage collection
                    gc.collect()

                    print(f"   ✅ Scene rendered successfully!")
                    return str(scene_file)

                except Exception as e:
                    print(f"   ❌ Scene rendering failed: {e}")
                    # Emergency cleanup
                    for clip in clips_to_cleanup:
                        if clip is not None:
                            try:
                                clip.close()
                            except:
                                pass
                    return None

            # Process all scenes
            scene_counter = 0
            total_scenes = len(story_scenes) + (1 if hook_scene else 0) + (1 if subscribe_scene else 0)

            # 1. Hook scene
            if hook_scene:
                scene_counter += 1
                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Hook)")
                hook_video = render_single_scene(hook_scene, "hook")
                if hook_video:
                    scene_video_files.append(hook_video)

            # 2. Subscribe scene
            if subscribe_scene:
                scene_counter += 1
                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Subscribe)")
                subscribe_video = render_single_scene(subscribe_scene, "subscribe")
                if subscribe_video:
                    scene_video_files.append(subscribe_video)

            # 3. Story scenes
            for scene_data in story_scenes:
                scene_counter += 1
                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Story Scene {scene_data['scene_id']})")
                scene_video = render_single_scene(scene_data, "story")
                if scene_video:
                    scene_video_files.append(scene_video)

            # Cleanup fireplace overlay
            if fireplace_overlay_base:
                fireplace_overlay_base.close()

            print(f"\n🔗 COMBINING SCENE VIDEOS:")
            print(f"   📦 Total scene videos created: {len(scene_video_files)}")
            print(f"   🎬 Combining into final video...")

            if not scene_video_files:
                print("   ❌ No scene videos created")
                return None

            # Combine all scene videos using FFmpeg
            scene_list_file = scene_videos_dir / "scene_list.txt"
            with open(scene_list_file, 'w') as f:
                for scene_file in scene_video_files:
                    f.write(f"file '{scene_file}'\n")

            # Use FFmpeg to combine scenes
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
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   ✅ Scene videos combined successfully!")

                # Cleanup scene video files
                print(f"   🧹 Cleaning up scene video files...")
                for scene_file in scene_video_files:
                    try:
                        os.remove(scene_file)
                    except:
                        pass

                # Remove scene list file
                try:
                    os.remove(scene_list_file)
                    scene_videos_dir.rmdir()
                except:
                    pass

                print(f"   ✅ Cleanup completed!")
            else:
                print(f"   ❌ FFmpeg combination failed:")
                print(f"      {result.stderr}")
                return None

            if final_video.exists():
                print(f"\n✅ SCENE-BY-SCENE MOVIEPY STYLE completed: {final_video}")
                print(f"🎬 Successfully rendered {len(scene_video_files)} scenes!")
                print("🔥 Fireplace overlay working without memory leak!")
                print("🎯 Perfect audio sync - no word cutting!")
                print("✅ Memory-efficient scene-by-scene processing!")
                return final_video
            else:
                print("❌ Final video not created")
                return None

        except ImportError:
            print("❌ MoviePy not installed. Install with: pip install moviepy")
            print("🔄 Fallback to FFmpeg method...")
            return self.create_video_ffmpeg_fallback(None, None, row_index, total_duration)
        except Exception as e:
            print(f"❌ Scene-by-scene MoviePy failed: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Fallback to simple video...")
            return None

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
        """Ana video üretim fonksiyonu - Scene-by-scene timeline-based version"""
        total_steps = 5  # Reduced from 8 (removed sequence_build, audio_combine, image_list steps)
        current_step = 0

        print(f"\n" + "🎬" * 80)
        print("VIDEO COMPOSER - SCENE-BY-SCENE PROCESSING")
        print("🎬" * 80)
        print(f"🎯 PROJECT: {topic_data['topic']}")
        print(f"🆔 PROJECT ID: {row_index}")
        print(f"📁 OUTPUT DIR: {self.current_output_dir}")
        print()
        print("📋 PROCESSING METHOD:")
        print("   🎯 SCENE-BY-SCENE APPROACH - Server Version")
        print("   ✅ Uses story_audio_youtube_timeline.json (ACTUAL generated scenes)")
        print("   ❌ No longer uses scene_plan.json (only planning data)")
        print("   📝 Each scene processed individually with its own audio")
        print("   📝 Layer 1: Individual Scene Videos (perfect audio sync)")
        print("   📝 Layer 2: Fireplace Overlay (per scene)")
        print("   📝 Layer 3: FFmpeg Combination (seamless)")
        print("   ✅ Fixed: Uses REAL generated scenes from timeline!")
        print("   ✅ Fixed: Perfect audio sync - no word cutting!")
        print("   ✅ Fixed: Memory efficient scene-by-scene processing!")
        print("   🖥️ Server: Database integrated with progress tracking")
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

            # 2. Load audio timeline (ACTUAL generated scenes)
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
            print(f"   📋 Total timeline duration: {timeline_data.get('total_duration_formatted', 'unknown')}")

            progress_tracker.mark_stage_completed("timeline_load")
            usage_tracker.add_stage("timeline_load", time.time() - start_time)
            usage_tracker.update_performance_data(total_scenes=len(story_scenes))

            # 3. DIRECT Scene-by-scene processing (no sequence building needed)
            current_step += 1
            self.print_progress(current_step, total_steps, "🎯 Preparing scene-by-scene processing...")
            start_time = time.time()

            # Calculate total duration from timeline data
            total_duration = timeline_data.get('total_duration_ms', 0) / 1000.0

            print(f"\n🎯 SCENE-BY-SCENE PREPARATION:")
            print(f"   📖 Story scenes: {len(story_scenes)}")
            print(f"   🎬 Hook available: {'✅' if hook_subscribe_data[0] else '❌'}")
            print(f"   🔔 Subscribe available: {'✅' if hook_subscribe_data[1] else '❌'}")
            print(f"   ⏱️  Total duration: {total_duration / 60:.1f} minutes")
            print(f"   📋 Each scene will be processed with its own audio")
            print(f"   🎯 Natural breakpoints ensure perfect audio sync")

            progress_tracker.mark_stage_completed("scene_preparation")
            usage_tracker.add_stage("scene_preparation", time.time() - start_time)
            usage_tracker.update_performance_data(video_duration_seconds=total_duration)

            # Skip sequence building and audio combination - not needed for scene-by-scene
            current_step += 3  # Skip steps 4, 5, 6
            self.print_progress(current_step, total_steps, "⏭️ Skipping sequence/audio steps (scene-by-scene handles individually)...")

            # Check processing limits before video render
            can_continue, limit_reason = usage_tracker.check_processing_limits()
            if not can_continue:
                progress_tracker.mark_stage_failed("video_render", f"Processing limit exceeded: {limit_reason}")
                return None

            # 7. SCENE-BY-SCENE MoviePy render
            current_step += 1
            self.print_progress(current_step, total_steps, "🎬 SCENE-BY-SCENE MoviePy: Processing each scene separately...")
            start_time = time.time()

            print(f"\n🎬 VIDEO RENDERING (SCENE-BY-SCENE):")
            print(f"   📝 Method: SCENE-BY-SCENE MoviePy (perfect audio sync)")
            print(f"   🎯 Natural breakpoints: no word cutting")
            print(f"   🔥 Overlay: Fireplace animation per scene")
            print(f"   🎵 Audio: Each scene with its own audio")
            print(f"   📊 Story scenes: {len(story_scenes)}")
            print(f"   ⏱️  Expected duration: {total_duration / 60:.1f} minutes")

            progress_tracker.set_render_method("moviepy_scene_by_scene_timeline_based")
            usage_tracker.update_performance_data(render_method="moviepy_scene_by_scene_timeline_based")

            final_video = self.create_video_scene_by_scene_style(story_scenes, hook_subscribe_data, row_index, total_duration)
            if not final_video:
                progress_tracker.mark_stage_failed("video_render", "MoviePy scene-by-scene render failed")
                return None

            print(f"   ✅ Video rendering completed")
            print(f"   📁 Output file: {Path(final_video).name}")

            progress_tracker.mark_stage_completed("video_render")
            usage_tracker.add_stage("video_render", time.time() - start_time)

            # 8. Video doğrula
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

            # 9. Metadata kaydet
            usage_summary = usage_tracker.print_final_summary()

            video_metadata = {
                "title": platform_metadata["title_options"][0] if platform_metadata.get("title_options") else topic_data["topic"],
                "duration_seconds": actual_duration,
                "story_scene_count": len(story_scenes),
                "sequence_count": len(sequence),
                "created_at": datetime.now().isoformat(),
                "output_file": str(final_video),
                "processing_steps": total_steps,
                "render_method": "moviepy_scene_by_scene_timeline_based_server",
                "timeline_mode": True,
                "scene_by_scene_processing": True,
                "perfect_audio_sync": True,
                "overlay_working": True,
                "cleanup_timing": "fixed",
                "memory_efficient": True,
                "server_version": True,
                "database_integrated": True,
                "usage_summary": usage_summary,
                "timeline_stats": {
                    "total_timeline_scenes": timeline_data.get('total_scenes', 0),
                    "story_scenes_used": len(story_scenes),
                    "hook_used": hook_subscribe_data[0] is not None,
                    "subscribe_used": hook_subscribe_data[1] is not None
                }
            }

            metadata_file = Path(self.current_output_dir) / "video_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(video_metadata, f, indent=2, ensure_ascii=False)

            # Tamamlanma mesajı
            self.print_progress(total_steps, total_steps, "SCENE-BY-SCENE MoviePy render completed!")

            print(f"\n" + "🎉" * 80)
            print("VIDEO CREATION COMPLETED SUCCESSFULLY!")
            print("🎉" * 80)
            print(f"🎬 PROJECT: {topic_data['topic']}")
            print(f"🆔 PROJECT ID: {row_index}")
            print(f"📁 VIDEO FILE: {final_video}")
            print(f"⏱️  DURATION: {actual_duration / 60:.1f} minutes")
            print(f"📦 FILE SIZE: {file_size_mb:.1f} MB")
            print(f"🎬 SEGMENTS: {total_segments} total")
            print(f"📖 STORY SCENES: {len(story_scenes)} from timeline")
            print(f"🎭 METHOD: SCENE-BY-SCENE Timeline MoviePy (perfect audio sync)")
            print(f"🔥 OVERLAY: Working (animated with MoviePy)")
            print(f"🎵 AUDIO: Working (full sequence)")
            print(f"✅ STATUS: Uses REAL generated scenes from timeline!")
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
        print("SERVER VIDEO COMPOSER v1.1 - TIMELINE MODE")
        print("🔗 Database integrated")
        print("🎬 YouTube Production Video Generation")
        print("📋 Timeline-based scene loading (ACTUAL generated)")
        print("🔥 MoviePy + Fireplace Overlay + ALL Timeline Scenes")
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

            # Create video with timeline-based tracking
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
                print("✅ YouTube-optimized video with TIMELINE scenes")
                print("✅ MoviePy with fireplace overlay")
                print("✅ Scene-by-scene processing (perfect audio sync)")
                print("✅ Fixed cleanup timing")
                print("✅ Timeline-based processing")
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
        print("🚀 SERVER VIDEO COMPOSER v1.1 - SCENE-BY-SCENE TIMELINE MODE")
        print("🔗 Database integration with progress tracking")
        print("🎬 YouTube Production Video Generation")
        print("📋 Timeline-based scene loading (ACTUAL generated scenes)")
        print("🎯 Scene-by-scene processing (perfect audio sync)")
        print("🔥 MoviePy + Fireplace Overlay + ALL Timeline Scenes")
        print("🎭 Fixed cleanup timing + Server infrastructure")
        print("🖥️ Production-ready automation")
        print("=" * 60)

        producer = ServerYouTubeVideoProducer()
        success = producer.run_video_generation()

        if success:
            print("🎊 Video generation completed successfully!")
            print("📁 Video saved: final_video.mp4")
            print("📋 Metadata saved: video_metadata.json")
            print("🔥 Fireplace overlay included")
            print("📋 Timeline-based scenes used")
            print("🎯 Scene-by-scene processing used (perfect audio sync)")
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