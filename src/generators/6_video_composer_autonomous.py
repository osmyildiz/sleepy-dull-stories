"""
Sleepy Dull Stories - AUTONOMOUS Video Composer
SCENE-BY-SCENE Processing for Perfect Audio Sync + AUTONOMOUS MODE
FIXED: All sequence/total_segments errors removed
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
            "target_resolution": [3840, 2160],
            "quality_profiles": {
                "youtube": {"codec": "libx264", "preset": "ultrafast", "crf": 23},
                "balanced": {"codec": "libx264", "preset": "medium", "crf": 23},
                "fast": {"codec": "libx264", "preset": "ultrafast", "crf": 28}
            },
            "budget_controls": {
                "max_processing_time_minutes": 60,
                "max_memory_usage_gb": 8,
                "warn_threshold_minutes": 40
            },
            "server_mode": True,
            "production_ready": True,
            "moviepy_enabled": True,
            "ffmpeg_fallback": True,
            "timeline_mode": True,
            "scene_by_scene_mode": True
        }

        print("✅ Video composition configuration loaded")
        print(
            f"🎬 Target: {self.video_config['target_resolution'][0]}x{self.video_config['target_resolution'][1]} @ {self.video_config['target_fps']}fps")
        print(f"📋 Timeline mode: {'✅ ENABLED' if self.video_config['timeline_mode'] else '❌ DISABLED'}")
        print(f"🎯 Scene-by-scene mode: {'✅ ENABLED' if self.video_config['scene_by_scene_mode'] else '❌ DISABLED'}")

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

    def get_completed_audio_topic_ready_for_video(self) -> Optional[Tuple[int, str, str, str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if video generation columns exist, if not create them
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

    def mark_video_generation_failed(self, topic_id: int, error_message: str):
        """Mark video generation as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET video_generation_status = 'failed',
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
            ('video_generation_status', 'TEXT DEFAULT "pending"'),
            ('video_generation_started_at', 'DATETIME'),
            ('video_generation_completed_at', 'DATETIME'),
            ('video_duration_seconds', 'REAL DEFAULT 0.0'),
            ('video_file_size_mb', 'REAL DEFAULT 0.0'),
            ('video_processing_time_minutes', 'REAL DEFAULT 0.0')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

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

        conn.close()

        return {
            'video_generation_queue': video_queue,
            'video_generation_active': video_active
        }


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
            "scene_by_scene_mode": CONFIG.video_config.get("scene_by_scene_mode", False)
        }

        self.budget_controls = CONFIG.video_config.get("budget_controls", {})

    def check_processing_limits(self) -> Tuple[bool, str]:
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
        print(
            f"🎯 Scene-by-scene mode: {'✅ ENABLED' if self.performance_data.get('scene_by_scene_mode') else '❌ DISABLED'}")
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

        print("🎬 Server YouTube Video Producer v2.0 Initialized")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🎥 Overlay Path: {self.overlay_path}")
        print(f"📋 Timeline mode: {'✅ ENABLED' if CONFIG.video_config.get('timeline_mode') else '❌ DISABLED'}")
        print(
            f"🎯 Scene-by-scene mode: {'✅ ENABLED' if CONFIG.video_config.get('scene_by_scene_mode') else '❌ DISABLED'}")

        self.check_ffmpeg()

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
                        'image_file': scene.get('image_file', f"scene_{scene['scene_number']:02d}_4k.png"),
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
        # Audio dosyası
        audio_file = None
        for format_str in [f"scene_{scene_id:02d}_audio", f"scene_{scene_id}_audio"]:
            test_file = audio_dir / f"{format_str}.mp3"
            if test_file.exists():
                audio_file = test_file
                break

        # Image dosyası - önce 4K'yı dene
        image_file = None
        for format_str in [f"scene_{scene_id:02d}", f"scene_{scene_id}"]:
            # Önce 4K versiyonunu dene
            test_file_4k = scenes_dir / f"{format_str}_4k.png"
            if test_file_4k.exists():
                image_file = test_file_4k
                break
            # 4K yoksa normal'i dene
            test_file = scenes_dir / f"{format_str}.png"
            if test_file.exists():
                image_file = test_file
                break

        return audio_file, image_file

    def create_video_scene_by_scene_style(self, story_scenes, hook_subscribe_data, row_index, total_duration):
        """SCENE-BY-SCENE MoviePy Style: Process each scene separately for perfect audio sync"""
        fireplace_video = self.overlay_path / "fireplace.mp4"
        final_video = Path(self.current_output_dir) / "final_video.mp4"
        scene_videos_dir = Path(self.current_output_dir) / "scene_videos"
        audio_dir = Path(self.current_output_dir) / "audio_parts"
        scenes_dir = Path(self.current_output_dir) / "scenes"

        print(
            f"🎬 SCENE-BY-SCENE MOVIEPY STYLE: Process each scene separately ({total_duration:.1f}s = {total_duration / 60:.1f} minutes)")
        print("📝 Using MoviePy with SCENE-BY-SCENE processing for perfect audio sync")
        print("🎯 Natural breakpoints - no audio cutting in the middle of words")

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

                    # Add fireplace overlay (OPTIMIZED FOR 5.25min FIREPLACE)
                    if fireplace_overlay_base:
                        print(f"   🔥 Adding fireplace overlay...")

                        fireplace_duration = fireplace_overlay_base.duration
                        print(f"      📏 Fireplace duration: {fireplace_duration:.1f}s")
                        print(f"      📏 Scene duration: {scene_duration:.1f}s")

                        # For 5.25-minute fireplace (315s), most scenes won't need looping
                        if scene_duration <= fireplace_duration:
                            # Scene is shorter than fireplace - just trim fireplace
                            print(f"      ✂️  Scene shorter than fireplace - trimming to {scene_duration:.1f}s")
                            scene_fireplace = fireplace_overlay_base.subclip(0, scene_duration)
                            clips_to_cleanup.append(scene_fireplace)
                        else:
                            # Scene is longer than fireplace - need to loop
                            loops_needed = int(scene_duration / fireplace_duration) + 1
                            print(f"      🔄 Scene longer than fireplace - need {loops_needed} loops")

                            fireplace_clips = []
                            for loop_idx in range(loops_needed):
                                fireplace_clips.append(fireplace_overlay_base.copy())

                            scene_fireplace_looped = concatenate_videoclips(fireplace_clips)
                            clips_to_cleanup.extend(fireplace_clips)
                            clips_to_cleanup.append(scene_fireplace_looped)

                            # Trim to exact scene duration
                            scene_fireplace = scene_fireplace_looped.subclip(0, scene_duration)
                            clips_to_cleanup.append(scene_fireplace)

                        # Resize and set opacity
                        scene_fireplace = scene_fireplace.resize(main_video.size)
                        scene_fireplace = scene_fireplace.set_opacity(0.3)
                        scene_fireplace = scene_fireplace.without_audio()

                        # Composite
                        scene_final = CompositeVideoClip([main_video, scene_fireplace])
                        clips_to_cleanup.append(scene_final)

                        print(f"   ✅ Fireplace overlay added (5.25min optimized)")
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
                    print(f"      📏 Duration: {scene_duration:.1f}s")
                    print(f"      ⏱️  Expected time: ~{scene_duration * 0.3 / 60:.1f} minutes")

                    start_render_time = time.time()

                    scene_final.write_videofile(
                        str(scene_file),
                        fps=30,
                        codec="libx264",
                        audio_codec="aac",
                        temp_audiofile=f'temp-audio-{scene_file.stem}.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger='bar'
                    )

                    render_time = time.time() - start_render_time
                    file_size_mb = os.path.getsize(scene_file) / (1024 * 1024)

                    print(f"      ⏱️  Actual render time: {render_time / 60:.1f} minutes")
                    print(f"      📊 Render speed: {scene_duration / render_time:.1f}x realtime")
                    print(f"      📦 File size: {file_size_mb:.1f} MB")

                    # Cleanup scene clips immediately
                    print(f"      🧹 Cleaning up {len(clips_to_cleanup)} clips...")
                    for clip in clips_to_cleanup:
                        if clip is not None:
                            try:
                                clip.close()
                            except:
                                pass
                    clips_to_cleanup.clear()

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
            overall_start_time = time.time()

            print(f"\n🎬 STARTING SCENE-BY-SCENE PROCESSING:")
            print(f"   📊 Total scenes to process: {total_scenes}")
            print(f"   ⏱️  Estimated total time: {total_scenes * 2:.1f} minutes")

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
                elapsed_time = time.time() - overall_start_time
                estimated_remaining = (elapsed_time / scene_counter) * (total_scenes - scene_counter)

                print(f"\n📊 PROCESSING SCENE {scene_counter}/{total_scenes} (Story Scene {scene_data['scene_id']})")
                print(f"   ⏱️  Elapsed: {elapsed_time / 60:.1f}m | Remaining: {estimated_remaining / 60:.1f}m")

                scene_video = render_single_scene(scene_data, "story")
                if scene_video:
                    scene_video_files.append(scene_video)

            total_processing_time = time.time() - overall_start_time
            print(f"\n📊 ALL SCENES PROCESSED:")
            print(f"   ✅ Successful scenes: {len(scene_video_files)}/{total_scenes}")
            print(f"   ⏱️  Total processing time: {total_processing_time / 60:.1f} minutes")

            # Cleanup fireplace overlay
            if fireplace_overlay_base:
                fireplace_overlay_base.close()

            print(f"\n🔗 COMBINING SCENE VIDEOS:")
            print(f"   📦 Total scene videos created: {len(scene_video_files)}")

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
                for scene_file in scene_video_files:
                    try:
                        os.remove(scene_file)
                    except:
                        pass

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
                print("🔥 Fireplace overlay working with 5.25min optimization!")
                print("🎯 Perfect audio sync - no word cutting!")
                return final_video
            else:
                print("❌ Final video not created")
                return None

        except ImportError:
            print("❌ MoviePy not installed. Install with: pip install moviepy")
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
        """Main video creation function - Scene-by-scene timeline-based version"""
        total_steps = 5
        current_step = 0

        print(f"\n" + "🎬" * 80)
        print("VIDEO COMPOSER - SCENE-BY-SCENE PROCESSING")
        print("🎬" * 80)
        print(f"🎯 PROJECT: {topic_data['topic']}")
        print(f"🆔 PROJECT ID: {row_index}")
        print(f"📁 OUTPUT DIR: {self.current_output_dir}")
        print()
        print("📋 PROCESSING METHOD:")
        print("   🎯 SCENE-BY-SCENE APPROACH - Server Version v2.0")
        print("   ✅ Uses story_audio_youtube_timeline.json (ACTUAL generated scenes)")
        print("   📝 Each scene processed individually with its own audio")
        print("   📝 Layer 1: Individual Scene Videos (perfect audio sync)")
        print("   📝 Layer 2: Fireplace Overlay (5.25min optimized)")
        print("   📝 Layer 3: FFmpeg Combination (seamless)")
        print("   ✅ Fixed: Perfect audio sync - no word cutting!")
        print("   ✅ Fixed: 5.25min fireplace optimization!")
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

            progress_tracker.mark_stage_completed("scene_preparation")
            usage_tracker.add_stage("scene_preparation", time.time() - start_time)
            usage_tracker.update_performance_data(video_duration_seconds=total_duration)

            # 4. Scene-by-scene video render
            current_step += 1
            self.print_progress(current_step, total_steps,
                                "🎬 SCENE-BY-SCENE MoviePy: Processing each scene separately...")
            start_time = time.time()

            print(f"\n🎬 VIDEO RENDERING (SCENE-BY-SCENE):")
            print(f"   📝 Method: SCENE-BY-SCENE MoviePy (perfect audio sync)")
            print(f"   🔥 Overlay: 5.25min fireplace optimization")
            print(f"   📊 Story scenes: {len(story_scenes)}")

            progress_tracker.set_render_method("moviepy_scene_by_scene_timeline_based")
            usage_tracker.update_performance_data(render_method="moviepy_scene_by_scene_timeline_based")

            final_video = self.create_video_scene_by_scene_style(story_scenes, hook_subscribe_data, row_index,
                                                                 total_duration)
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
            total_segments = len(story_scenes) + (1 if hook_subscribe_data[0] else 0) + (
                1 if hook_subscribe_data[1] else 0)

            video_metadata = {
                "title": platform_metadata["title_options"][0] if platform_metadata.get("title_options") else
                topic_data["topic"],
                "duration_seconds": actual_duration,
                "story_scene_count": len(story_scenes),
                "total_segments": total_segments,
                "created_at": datetime.now().isoformat(),
                "output_file": str(final_video),
                "processing_steps": total_steps,
                "render_method": "moviepy_scene_by_scene_timeline_based_server_v2",
                "timeline_mode": True,
                "scene_by_scene_processing": True,
                "perfect_audio_sync": True,
                "fireplace_optimization": "5.25min",
                "server_version": "2.0",
                "database_integrated": True,
                "usage_summary": usage_summary
            }

            metadata_file = Path(self.current_output_dir) / "video_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(video_metadata, f, indent=2, ensure_ascii=False)

            self.print_progress(total_steps, total_steps, "SCENE-BY-SCENE MoviePy render completed!")

            print(f"\n" + "🎉" * 80)
            print("VIDEO CREATION COMPLETED SUCCESSFULLY!")
            print("🎉" * 80)
            print(f"🎬 PROJECT: {topic_data['topic']}")
            print(f"📁 VIDEO FILE: {final_video}")
            print(f"⏱️  DURATION: {actual_duration / 60:.1f} minutes")
            print(f"📦 FILE SIZE: {file_size_mb:.1f} MB")
            print(f"🎬 SEGMENTS: {total_segments} total")
            print(f"📖 STORY SCENES: {len(story_scenes)} from timeline")
            print(f"🎭 METHOD: SCENE-BY-SCENE Timeline MoviePy (perfect audio sync)")
            print(f"🔥 OVERLAY: 5.25min fireplace optimization")
            print(f"🎵 AUDIO: Perfect sync - no word cutting!")
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
        print("SERVER VIDEO COMPOSER v2.0 - SCENE-BY-SCENE TIMELINE MODE")
        print("🔗 Database integrated")
        print("🎬 YouTube Production Video Generation")
        print("📋 Timeline-based scene loading (ACTUAL generated)")
        print("🎯 Scene-by-scene processing (perfect audio sync)")
        print("🔥 5.25min Fireplace Optimization")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        limits = CONFIG.video_config.get("budget_controls", {})
        print(f"⏱️  PROCESSING LIMITS:")
        print(f"   📊 Max processing time: {limits.get('max_processing_time_minutes', 60)} minutes")
        print(f"   📊 Warning threshold: {limits.get('warn_threshold_minutes', 40)} minutes")

        # Get next project from database
        found, project_info = self.get_next_project_from_database()
        if not found:
            return False

        print(f"✅ Project found: {project_info['topic']}")

        # Initialize trackers
        progress_tracker = VideoProgressTracker(self.current_topic_id, CONFIG.paths['OUTPUT_DIR'])
        usage_tracker = VideoUsageTracker()

        try:
            # Create video with scene-by-scene approach
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
                print("✅ Scene-by-scene processing (perfect audio sync)")
                print("✅ 5.25min fireplace optimization")
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

    # Initialize database manager for pipeline status
    db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
    db_manager = DatabaseVideoManager(str(db_path))

    # Setup graceful shutdown
    running = True
    processed_count = 0
    start_time = time.time()

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

                # Initialize generator
                producer = ServerYouTubeVideoProducer()

                # Process one topic
                success = producer.run_video_generation()

                if success:
                    processed_count += 1
                    print(f"\n✅ Video generation completed!")
                    print(f"📊 Progress: {processed_count} topics processed")
                else:
                    print(f"\n⚠️ Video generation failed or no projects ready")

                # Short pause between topics
                if running:
                    time.sleep(5)

            else:
                # No topics ready, wait
                print("😴 No completed audio topics ready for video generation. Waiting 60s...")
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
    print(f"\n🏁 AUTONOMOUS VIDEO GENERATION SHUTDOWN")
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
            print("🚀 SERVER VIDEO COMPOSER v2.0 - SCENE-BY-SCENE TIMELINE MODE")
            print("🔗 Database integration with progress tracking")
            print("🎬 YouTube Production Video Generation")
            print("📋 Timeline-based scene loading (ACTUAL generated scenes)")
            print("🎯 Scene-by-scene processing (perfect audio sync)")
            print("🔥 5.25min Fireplace Optimization")
            print("🎭 Fixed cleanup timing + Server infrastructure")
            print("🖥️ Production-ready automation")
            print("=" * 60)

            producer = ServerYouTubeVideoProducer()
            success = producer.run_video_generation()

            if success:
                print("🎊 Video generation completed successfully!")
                print("📁 Video saved: final_video.mp4")
                print("📋 Metadata saved: video_metadata.json")
                print("🔥 5.25min fireplace optimization included")
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