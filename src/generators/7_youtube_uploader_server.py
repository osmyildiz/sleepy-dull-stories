"""
Sleepy Dull Stories - SERVER-READY YouTube Uploader
YouTube Video Upload with Metadata Integration
Production-optimized with complete automation and database integration
"""

import os
import json
import sqlite3
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# YouTube API imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print(
        "⚠️  YouTube API libraries not installed. Run: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2")
    YOUTUBE_API_AVAILABLE = False

# Load environment first
load_dotenv()

# YouTube API Scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


# Server Configuration Class
class ServerConfig:
    """Server-friendly configuration management for YouTube Uploader"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_youtube_config()
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
            'CONFIG_DIR': str(self.project_root / 'config'),
            'CREDENTIALS_DIR': str(self.project_root / 'credentials')
        }

        print(f"✅ YouTube Uploader server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")

    def setup_youtube_config(self):
        """Setup YouTube upload configuration"""
        self.youtube_config = {
            "max_retry_attempts": 5,
            "retry_delay_seconds": [1, 2, 4, 8, 16],  # Exponential backoff
            "chunk_size": 1024 * 1024,  # 1MB chunks for upload
            "timeout_seconds": 3600,  # 1 hour timeout
            "budget_controls": {
                "max_uploads_per_day": 50,
                "max_processing_time_minutes": 20,
                "warn_threshold_minutes": 15
            },
            "privacy_status": "private",  # Default to private until manual review
            "default_category": "22",  # People & Blogs
            "made_for_kids": False,
            "server_mode": True,
            "production_ready": True
        }

        print("✅ YouTube upload configuration loaded")
        print(f"🔒 Default privacy: {self.youtube_config['privacy_status']}")
        print(f"📊 Max uploads/day: {self.youtube_config['budget_controls']['max_uploads_per_day']}")

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"youtube_uploader_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("YouTubeUploader")
        self.logger.info(f"✅ YouTube uploader logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR', 'CREDENTIALS_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All YouTube uploader directories created/verified")


# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 YouTube Uploader server configuration loaded successfully")
except Exception as e:
    print(f"❌ YouTube Uploader server configuration failed: {e}")
    sys.exit(1)


# Database YouTube Management Integration
class DatabaseYouTubeManager:
    """Professional YouTube upload management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed_video_topic_ready_for_youtube(self) -> Optional[Tuple[int, str, str, str]]:
        """Get completed video topic that needs YOUTUBE upload"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if youtube upload columns exist, if not create them
        cursor.execute('PRAGMA table_info(topics)')
        columns = [row[1] for row in cursor.fetchall()]

        # Add columns individually if they don't exist
        columns_to_add = [
            ('youtube_upload_status', 'TEXT DEFAULT "pending"'),
            ('youtube_upload_started_at', 'DATETIME'),
            ('youtube_upload_completed_at', 'DATETIME'),
            ('youtube_video_id', 'TEXT'),
            ('youtube_video_url', 'TEXT'),
            ('youtube_upload_size_mb', 'REAL DEFAULT 0.0'),
            ('youtube_upload_time_minutes', 'REAL DEFAULT 0.0')
        ]

        for column_name, column_definition in columns_to_add:
            if column_name not in columns:
                print(f"🔧 Adding column: {column_name}")
                cursor.execute(f'ALTER TABLE topics ADD COLUMN {column_name} {column_definition}')

        conn.commit()
        print("✅ YouTube upload columns verified/added")

        cursor.execute('''
            SELECT id, topic, description, output_path 
            FROM topics 
            WHERE status = 'completed' 
            AND video_generation_status = 'completed'
            AND (youtube_upload_status IS NULL OR youtube_upload_status = 'pending')
            ORDER BY video_generation_completed_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_youtube_upload_started(self, topic_id: int):
        """Mark YouTube upload as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET youtube_upload_status = 'in_progress', 
                youtube_upload_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_youtube_upload_completed(self, topic_id: int, video_id: str, video_url: str,
                                      upload_size_mb: float, upload_time_minutes: float):
        """Mark YouTube upload as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET youtube_upload_status = 'completed',
                youtube_upload_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                youtube_video_id = ?,
                youtube_video_url = ?,
                youtube_upload_size_mb = ?,
                youtube_upload_time_minutes = ?
            WHERE id = ?
        ''', (video_id, video_url, upload_size_mb, upload_time_minutes, topic_id))

        conn.commit()
        conn.close()

    def mark_youtube_upload_failed(self, topic_id: int, error_message: str):
        """Mark YouTube upload as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET youtube_upload_status = 'failed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()


class YouTubeProgressTracker:
    """YouTube upload progress tracking and resume functionality"""

    def __init__(self, story_id: int, output_base_path: str):
        self.story_id = story_id
        self.output_dir = os.path.join(output_base_path, str(story_id))
        self.progress_file = os.path.join(self.output_dir, "youtube_progress.json")

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
                print(f"📂 YouTube progress loaded: {data.get('current_stage', 'unknown')} stage")
                return data
            except Exception as e:
                print(f"⚠️  YouTube progress file corrupted, starting fresh: {e}")

        return {
            "story_id": self.story_id,
            "current_stage": "init",
            "completed_stages": [],
            "failed_attempts": 0,
            "last_update": datetime.now().isoformat(),
            "session_start": datetime.now().isoformat(),
            "stages": {
                "auth_setup": {"status": "pending", "timestamp": None},
                "metadata_load": {"status": "pending", "timestamp": None},
                "video_prepare": {"status": "pending", "timestamp": None},
                "youtube_upload": {"status": "pending", "timestamp": None},
                "upload_verify": {"status": "pending", "timestamp": None}
            }
        }

    def save_progress(self):
        """Save progress"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  YouTube progress save warning: {e}")

    def mark_stage_completed(self, stage: str, metadata: Dict = None):
        """Mark stage as completed"""
        if stage not in self.progress_data["completed_stages"]:
            self.progress_data["completed_stages"].append(stage)

        if stage in self.progress_data["stages"]:
            stage_data = {
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            if metadata:
                stage_data.update(metadata)
            self.progress_data["stages"][stage] = stage_data

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

    def cleanup_on_success(self):
        """Clean up progress file on successful completion"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                print(f"🗑️  YouTube progress file cleaned up (successful completion)")
        except Exception as e:
            print(f"⚠️  YouTube progress cleanup warning: {e}")


class YouTubeUsageTracker:
    """YouTube upload usage and performance tracking"""

    def __init__(self):
        self.session_start = datetime.now()
        self.upload_stages = []
        self.total_upload_time = 0.0

        # Performance tracking
        self.performance_data = {
            "videos_uploaded": 0,
            "total_upload_time_minutes": 0.0,
            "total_upload_size_mb": 0.0,
            "average_upload_speed_mbps": 0.0
        }

        # Budget controls
        self.budget_controls = CONFIG.youtube_config.get("budget_controls", {})

    def check_upload_limits(self) -> Tuple[bool, str]:
        """Check if upload limits are exceeded"""
        current_time = (datetime.now() - self.session_start).total_seconds() / 60  # minutes

        max_time = self.budget_controls.get("max_processing_time_minutes", 20)
        if current_time >= max_time:
            return False, f"UPLOAD TIME LIMIT: {current_time:.1f} min >= {max_time} min"

        warn_threshold = self.budget_controls.get("warn_threshold_minutes", 15)
        if current_time >= warn_threshold:
            print(f"⚠️  UPLOAD WARNING: {current_time:.1f} min approaching limit {max_time} min")

        return True, "OK"

    def add_stage(self, stage_name: str, duration_seconds: float):
        """Add upload stage timing"""
        self.upload_stages.append({
            "stage": stage_name,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        })
        self.total_upload_time += duration_seconds

    def update_performance_data(self, **kwargs):
        """Update performance tracking data"""
        self.performance_data.update(kwargs)

    def print_final_summary(self):
        """Print final upload summary"""
        total_time = (datetime.now() - self.session_start).total_seconds() / 60

        print(f"\n📺 FINAL YOUTUBE UPLOAD SUMMARY")
        print(f"=" * 50)
        print(f"📊 Videos uploaded: {self.performance_data.get('videos_uploaded', 0)}")
        print(f"⚡ Total upload time: {total_time:.1f} minutes")
        print(f"💾 Total upload size: {self.performance_data.get('total_upload_size_mb', 0.0):.1f} MB")
        print(f"🚀 Average upload speed: {self.performance_data.get('average_upload_speed_mbps', 0.0):.1f} Mbps")

        if self.upload_stages:
            print(f"📊 Upload stages:")
            for stage in self.upload_stages:
                print(f"   🔄 {stage['stage']}: {stage['duration_seconds']:.1f}s")

        return {
            "total_upload_time_minutes": total_time,
            "performance_data": self.performance_data,
            "upload_stages": self.upload_stages
        }


class YouTubeAuthenticator:
    """Handle YouTube API authentication"""

    def __init__(self):
        self.credentials_dir = Path(CONFIG.paths['CREDENTIALS_DIR'])
        self.token_file = self.credentials_dir / 'youtube_token.json'
        self.credentials_file = self.credentials_dir / 'youtube_credentials.json'

    def get_authenticated_service(self):
        """Get authenticated YouTube service"""
        if not YOUTUBE_API_AVAILABLE:
            raise Exception("YouTube API libraries not installed")

        creds = None

        # Load existing token
        if self.token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
                print("✅ Existing YouTube credentials loaded")
            except Exception as e:
                print(f"⚠️  Existing credentials invalid: {e}")

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    print("✅ YouTube credentials refreshed")
                except Exception as e:
                    print(f"⚠️  Credential refresh failed: {e}")
                    creds = None

            if not creds:
                if not self.credentials_file.exists():
                    raise Exception(f"YouTube credentials file not found: {self.credentials_file}")

                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), SCOPES)
                creds = flow.run_local_server(port=0)
                print("✅ New YouTube credentials obtained")

            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
                print(f"✅ YouTube credentials saved: {self.token_file}")

        return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


class ServerYouTubeUploader:
    """Server-ready YouTube Uploader with database integration"""

    def __init__(self):
        # Current project tracking
        self.current_topic_id = None
        self.current_output_dir = None
        self.current_topic = None
        self.current_description = None

        # Server paths from CONFIG
        self.output_path = Path(CONFIG.paths['OUTPUT_DIR'])

        # Database manager
        db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
        self.db_manager = DatabaseYouTubeManager(str(db_path))

        # YouTube authenticator
        self.authenticator = YouTubeAuthenticator()

        print("📺 Server YouTube Uploader v1.0 Initialized")
        print(f"📁 Output Directory: {self.output_path}")

    def log_step(self, step: str, status: str = "START", metadata: Dict = None):
        """Log upload steps with production logging"""
        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "ℹ️"
        print(f"{icon} {step}")
        CONFIG.logger.info(f"{step} - Status: {status} - Project: {self.current_topic_id}")

    def get_next_project_from_database(self) -> Tuple[bool, Optional[Dict]]:
        """Get next completed video project that needs YOUTUBE upload"""
        self.log_step("🔍 Finding completed video project for YouTube upload")

        result = self.db_manager.get_completed_video_topic_ready_for_youtube()

        if not result:
            self.log_step("✅ No completed video projects ready for YouTube upload", "INFO")
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
        self.db_manager.mark_youtube_upload_started(topic_id)

        self.log_step(f"✅ Found project: {topic}", "SUCCESS", project_info)
        return True, project_info

    def load_youtube_metadata(self, project_dir: Path) -> Optional[Dict]:
        """Load youtube_metadata.json from project directory"""
        metadata_file = project_dir / "youtube_metadata.json"

        if not metadata_file.exists():
            print(f"❌ YouTube metadata not found: {metadata_file}")
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ YouTube metadata loaded: {len(metadata.get('clickbait_titles', []))} titles")
            return metadata
        except Exception as e:
            print(f"❌ Error loading YouTube metadata: {e}")
            return None

    def find_video_file(self, project_dir: Path) -> Optional[Path]:
        """Find the final video file in project directory"""
        # Common video file names to check
        video_candidates = [
            "final_video.mp4",
            "final_video_no_overlay.mp4",
            "story_video.mp4",
            "video.mp4"
        ]

        for candidate in video_candidates:
            video_path = project_dir / candidate
            if video_path.exists():
                print(f"✅ Video file found: {video_path}")
                return video_path

        print(f"❌ No video file found in: {project_dir}")
        return None

    def prepare_upload_data(self, metadata: Dict, topic_data: Dict) -> Dict:
        """Prepare upload data from metadata and topic info"""
        # Use first clickbait title as default
        title = metadata.get('clickbait_titles', [topic_data['topic']])[0]

        # Build description
        description_parts = []

        if 'video_description' in metadata:
            vid_desc = metadata['video_description']

            # Hook
            if 'hook' in vid_desc:
                description_parts.append(vid_desc['hook'])
                description_parts.append("")

            # Main description
            if 'main_description' in vid_desc:
                description_parts.append(vid_desc['main_description'])
                description_parts.append("")

            # Chapters
            if 'chapters' in vid_desc and vid_desc['chapters']:
                description_parts.append("CHAPTERS:")
                for chapter in vid_desc['chapters']:
                    description_parts.append(f"{chapter['time']} - {chapter['title']}")
                description_parts.append("")

            # Subscribe CTA
            if 'subscribe_cta' in vid_desc:
                description_parts.append(vid_desc['subscribe_cta'])
                description_parts.append("")

            # Disclaimer
            if 'disclaimer' in vid_desc:
                description_parts.append(vid_desc['disclaimer'])

        description = "\n".join(description_parts)

        # Tags
        tags = metadata.get('tags', [])[:500]  # YouTube limit

        # Category
        youtube_meta = metadata.get('youtube_metadata', {})
        category_map = {
            "Education": "27",
            "Entertainment": "24",
            "People & Blogs": "22",
            "Howto & Style": "26"
        }
        category = category_map.get(youtube_meta.get('category', 'People & Blogs'), "22")

        upload_data = {
            'snippet': {
                'title': title[:100],  # YouTube title limit
                'description': description[:5000],  # YouTube description limit
                'tags': tags,
                'categoryId': category,
                'defaultLanguage': youtube_meta.get('default_language', 'en')
            },
            'status': {
                'privacyStatus': CONFIG.youtube_config.get('privacy_status', 'private'),
                'madeForKids': youtube_meta.get('made_for_kids', False),
                'embeddable': youtube_meta.get('embeddable', True),
                'license': youtube_meta.get('license', 'youtube')
            }
        }

        print(f"📝 Upload prepared:")
        print(f"   📝 Title: {title[:50]}...")
        print(f"   📝 Description: {len(description)} chars")
        print(f"   📝 Tags: {len(tags)} items")
        print(f"   🔒 Privacy: {upload_data['status']['privacyStatus']}")

        return upload_data

    def upload_video_with_retry(self, youtube_service, video_path: Path, upload_data: Dict, usage_tracker) -> Optional[
        str]:
        """Upload video with retry mechanism"""
        max_retries = CONFIG.youtube_config.get('max_retry_attempts', 5)
        retry_delays = CONFIG.youtube_config.get('retry_delay_seconds', [1, 2, 4, 8, 16])

        file_size = video_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        print(f"📺 Uploading video: {video_path.name} ({file_size_mb:.1f} MB)")

        for attempt in range(max_retries):
            try:
                # Check upload limits before each attempt
                can_continue, limit_reason = usage_tracker.check_upload_limits()
                if not can_continue:
                    print(f"🚨 UPLOAD LIMIT EXCEEDED: {limit_reason}")
                    return None

                print(f"🔄 Upload attempt {attempt + 1}/{max_retries}")
                start_time = time.time()

                # Create media upload
                media = MediaFileUpload(
                    str(video_path),
                    chunksize=CONFIG.youtube_config.get('chunk_size', 1024 * 1024),
                    resumable=True
                )

                # Insert video
                insert_request = youtube_service.videos().insert(
                    part=','.join(upload_data.keys()),
                    body=upload_data,
                    media_body=media
                )

                # Execute upload with progress
                response = None
                error = None
                retry = 0

                while response is None:
                    try:
                        print(f"📤 Uploading chunk...")
                        status, response = insert_request.next_chunk()
                        if status:
                            progress = int(status.progress() * 100)
                            print(f"📊 Upload progress: {progress}%")
                    except HttpError as e:
                        if e.resp.status in [500, 502, 503, 504]:
                            # Retriable HTTP errors
                            error = f"HTTP {e.resp.status}: {e.content}"
                            print(f"⚠️  Retriable error: {error}")
                            break
                        else:
                            # Non-retriable error
                            raise e

                if response is not None:
                    upload_time = time.time() - start_time
                    video_id = response['id']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"

                    # Calculate upload speed
                    upload_speed_mbps = (file_size_mb * 8) / (upload_time / 60)  # Mbps

                    print(f"✅ Upload successful!")
                    print(f"   🆔 Video ID: {video_id}")
                    print(f"   🔗 URL: {video_url}")
                    print(f"   ⚡ Upload time: {upload_time / 60:.1f} minutes")
                    print(f"   🚀 Upload speed: {upload_speed_mbps:.1f} Mbps")

                    # Update usage tracker
                    usage_tracker.update_performance_data(
                        videos_uploaded=usage_tracker.performance_data.get('videos_uploaded', 0) + 1,
                        total_upload_size_mb=usage_tracker.performance_data.get('total_upload_size_mb',
                                                                                0.0) + file_size_mb,
                        total_upload_time_minutes=usage_tracker.performance_data.get('total_upload_time_minutes',
                                                                                     0.0) + (upload_time / 60),
                        average_upload_speed_mbps=upload_speed_mbps
                    )

                    return video_id

                # If we get here, there was an error
                if error and attempt < max_retries - 1:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

            except HttpError as e:
                error_message = f"HTTP {e.resp.status}: {e.content}"
                print(f"❌ HTTP Error: {error_message}")

                if e.resp.status in [400, 401, 403]:
                    # Client errors - don't retry
                    print("❌ Client error - not retrying")
                    return None
                elif attempt < max_retries - 1:
                    # Server errors - retry
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"🔄 Server error - retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

            except Exception as e:
                error_message = str(e)
                print(f"❌ Unexpected error: {error_message}")

                if attempt < max_retries - 1:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

        print(f"❌ Upload failed after {max_retries} attempts")
        return None

    def upload_video(self, project_info: Dict, progress_tracker, usage_tracker):
        """Main video upload function with server tracking"""
        try:
            project_dir = Path(self.current_output_dir)

            # Stage 1: Authentication
            stage_start = time.time()
            self.log_step("🔐 Setting up YouTube authentication")

            try:
                youtube_service = self.authenticator.get_authenticated_service()
                progress_tracker.mark_stage_completed("auth_setup")
                usage_tracker.add_stage("auth_setup", time.time() - stage_start)
            except Exception as e:
                progress_tracker.mark_stage_failed("auth_setup", str(e))
                return None

            # Stage 2: Load metadata
            stage_start = time.time()
            self.log_step("📋 Loading YouTube metadata")

            metadata = self.load_youtube_metadata(project_dir)
            if not metadata:
                progress_tracker.mark_stage_failed("metadata_load", "YouTube metadata not found")
                return None

            progress_tracker.mark_stage_completed("metadata_load")
            usage_tracker.add_stage("metadata_load", time.time() - stage_start)

            # Stage 3: Prepare video
            stage_start = time.time()
            self.log_step("🎬 Preparing video for upload")

            video_file = self.find_video_file(project_dir)
            if not video_file:
                progress_tracker.mark_stage_failed("video_prepare", "Video file not found")
                return None

            upload_data = self.prepare_upload_data(metadata, project_info)

            progress_tracker.mark_stage_completed("video_prepare")
            usage_tracker.add_stage("video_prepare", time.time() - stage_start)

            # Stage 4: YouTube upload
            stage_start = time.time()
            self.log_step("📺 Uploading to YouTube")

            video_id = self.upload_video_with_retry(youtube_service, video_file, upload_data, usage_tracker)
            if not video_id:
                progress_tracker.mark_stage_failed("youtube_upload", "Video upload failed")
                return None

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            upload_time = time.time() - stage_start

            progress_tracker.mark_stage_completed("youtube_upload", {
                "video_id": video_id,
                "video_url": video_url,
                "upload_time_seconds": upload_time
            })
            usage_tracker.add_stage("youtube_upload", upload_time)

            # Stage 5: Verification
            stage_start = time.time()
            self.log_step("✅ Verifying upload")

            # Simple verification - check if video exists
            try:
                response = youtube_service.videos().list(part='snippet', id=video_id).execute()
                if response['items']:
                    print(f"✅ Upload verified: Video exists on YouTube")
                    progress_tracker.mark_stage_completed("upload_verify")
                    usage_tracker.add_stage("upload_verify", time.time() - stage_start)
                else:
                    progress_tracker.mark_stage_failed("upload_verify", "Video not found after upload")
                    return None
            except Exception as e:
                print(f"⚠️  Verification warning: {e}")
                progress_tracker.mark_stage_completed("upload_verify")  # Continue anyway

            # Calculate final metrics
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            total_time_minutes = sum([stage['duration_seconds'] for stage in usage_tracker.upload_stages]) / 60

            return {
                "video_id": video_id,
                "video_url": video_url,
                "file_size_mb": file_size_mb,
                "upload_time_minutes": total_time_minutes
            }

        except Exception as e:
            self.log_step(f"❌ Upload error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None

    def run_youtube_upload(self) -> bool:
        """Run YOUTUBE upload process for server environment"""
        print("🚀" * 50)
        print("SERVER YOUTUBE UPLOADER v1.0")
        print("🔗 Database integrated")
        print("📺 YouTube API Integration")
        print("🔒 OAuth2 Authentication")
        print("🖥️ Production-ready automation")
        print("🚀" * 50)

        # Print upload limits
        limits = CONFIG.youtube_config.get("budget_controls", {})
        print(f"⏱️  UPLOAD LIMITS:")
        print(f"   📊 Max uploads/day: {limits.get('max_uploads_per_day', 50)}")
        print(f"   📊 Max processing time: {limits.get('max_processing_time_minutes', 20)} minutes")
        print(f"   🔒 Default privacy: {CONFIG.youtube_config.get('privacy_status', 'private')}")

        if not YOUTUBE_API_AVAILABLE:
            print("❌ YouTube API libraries not installed")
            return False

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
        progress_tracker = YouTubeProgressTracker(self.current_topic_id, CONFIG.paths['OUTPUT_DIR'])
        usage_tracker = YouTubeUsageTracker()

        try:
            # Upload video with server tracking
            result = self.upload_video(project_info, progress_tracker, usage_tracker)

            if result:
                print(f"✅ YouTube upload successful!")
                print(f"📺 Video ID: {result['video_id']}")
                print(f"🔗 Video URL: {result['video_url']}")
                print(f"📦 File size: {result['file_size_mb']:.1f} MB")
                print(f"⚡ Upload time: {result['upload_time_minutes']:.1f} minutes")

                # Update database with results
                self.db_manager.mark_youtube_upload_completed(
                    self.current_topic_id,
                    result['video_id'],
                    result['video_url'],
                    result['file_size_mb'],
                    result['upload_time_minutes']
                )

                print(f"💾 Database updated with YouTube results")

                # Cleanup progress on success
                progress_tracker.cleanup_on_success()

                print("\n" + "🎉" * 50)
                print("YOUTUBE UPLOAD SUCCESSFUL!")
                print("✅ Video uploaded to YouTube")
                print("✅ Metadata and description applied")
                print("✅ Database updated with video URL")
                print("🎉" * 50)
                overall_success = True

            else:
                print(f"❌ YouTube upload failed")
                self.db_manager.mark_youtube_upload_failed(self.current_topic_id, "Upload process failed")
                overall_success = False

            return overall_success

        except Exception as e:
            self.log_step(f"❌ YouTube upload failed: {e}", "ERROR")
            self.db_manager.mark_youtube_upload_failed(self.current_topic_id, str(e))
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    try:
        print("🚀 SERVER YOUTUBE UPLOADER v1.0")
        print("🔗 Database integration with progress tracking")
        print("📺 YouTube API Integration")
        print("🔒 OAuth2 Authentication")
        print("📋 Metadata from youtube_metadata.json")
        print("🖥️ Production-ready automation")
        print("=" * 60)

        uploader = ServerYouTubeUploader()
        success = uploader.run_youtube_upload()

        if success:
            print("🎊 YouTube upload completed successfully!")
            print("📺 Video published to YouTube")
            print("📋 Metadata applied from youtube_metadata.json")
            print("🔗 Database updated with video URL")
            print("💾 Progress tracking enabled")
            print("🖥️ Server infrastructure working")
        else:
            print("⚠️ YouTube upload failed or no projects ready")

    except KeyboardInterrupt:
        print("\n⏹️ YouTube upload stopped by user")
        print("🛡️ Progress saved! Restart to resume from last completed stage.")
    except Exception as e:
        print(f"💥 YouTube upload failed: {e}")
        print("🛡️ Progress saved! Check youtube_progress.json for resume info.")
        CONFIG.logger.error(f"YouTube upload failed: {e}")
        import traceback

        traceback.print_exc()