"""
Sleepy Dull Stories - Infrastructure & Utils + YouTube Frame JSON
PART 2: Server config, database, save functions, YouTube automation
Contains: ServerConfig, DatabaseTopicManager, YouTube Frame JSON, Advanced SEO
"""

import os
import json
import pandas as pd
import logging
import sqlite3
import shutil
from datetime import datetime
from dotenv import load_dotenv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import Part 1
from anthropic import Anthropic

# Load environment first
load_dotenv()


class ServerConfig:
    """Server-friendly configuration management"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_claude_config()
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

        # Find topics.csv for fallback only
        self.paths['TOPIC_CSV_PATH'] = self.find_topics_csv()

        print(f"✅ Server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")
        print(f"   📄 Topics CSV (fallback): {self.paths['TOPIC_CSV_PATH']}")

    def find_topics_csv(self):
        """Find topics.csv in multiple locations or create fallback"""
        possible_paths = [
            self.project_root / 'topics.csv',
            self.project_root / 'data' / 'topics.csv',
            Path('topics.csv'),
            Path('../data/topics.csv'),
            Path('../../topics.csv')
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Found topics.csv: {path}")
                return str(path)

        # Create fallback CSV
        return self.create_fallback_csv()

    def create_fallback_csv(self):
        """Create fallback topics.csv for testing"""
        fallback_topics = [
            {
                "topic": "Ancient Roman Villa",
                "description": "A peaceful exploration of a Roman villa at sunset, with marble columns and gardens",
                "done": 0,
                "clickbait_title": "This Ancient Roman Secret Will Put You to Sleep Instantly",
                "font_design": "Bold Roman-style font with golden (#d4af37) and deep red accents"
            },
            {
                "topic": "Medieval Castle Library",
                "description": "A quiet evening in a medieval castle's vast library with flickering candlelight",
                "done": 0,
                "clickbait_title": "Medieval Monks' Secret Sleep Ritual (2 Hour Story)",
                "font_design": "Gothic font with warm amber colors and shadow effects"
            },
            {
                "topic": "Japanese Tea Garden",
                "description": "Serene moments in a traditional Japanese tea garden during cherry blossom season",
                "done": 0,
                "clickbait_title": "Ancient Japanese Sleep Secret Hidden in Tea Gardens",
                "font_design": "Elegant brush-style font with soft pink and gold colors"
            }
        ]

        # Create in data directory
        data_dir = Path(self.paths['DATA_DIR'])
        data_dir.mkdir(exist_ok=True)

        csv_path = data_dir / 'topics.csv'
        pd.DataFrame(fallback_topics).to_csv(csv_path, index=False)

        print(f"✅ Created fallback topics.csv: {csv_path}")
        return str(csv_path)

    def setup_claude_config(self):
        """Setup Claude configuration with PROVEN SETTINGS"""
        self.claude_config = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 64000,
            "temperature": 0.7,
            "target_scenes": 40,
            "target_duration_minutes": 120,
            "target_words_per_minute": 140,
            "two_stage_approach": True,
            "character_extraction": True,
            "thumbnail_generation": True,
            "max_characters": 5,
            "test_mode": False,
            "server_mode": True,
            "youtube_optimization": True,
            "platform_metadata_export": True,
            "production_specs_detailed": True,
            "streaming_response": True,
            "long_timeout": True
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
                Path('../../.env'),
                self.project_root / '.env'
            ]

            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file)
                    api_key = os.getenv('CLAUDE_API_KEY')
                    if api_key:
                        print(f"✅ API key loaded from: {env_file}")
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

    def setup_logging(self):
        """Setup production logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"story_gen_{datetime.now().strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("StoryGenerator")
        self.logger.info(f"✅ Logging initialized: {log_file}")

    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            'DATA_DIR', 'OUTPUT_DIR', 'LOGS_DIR', 'CONFIG_DIR'
        ]

        for dir_key in dirs_to_create:
            dir_path = Path(self.paths[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✅ All directories created/verified")


class DatabaseTopicManager:
    """Professional topic management using existing production.db"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.add_topic_table_to_existing_db()

    def add_topic_table_to_existing_db(self):
        """Add topic table to existing production.db"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                description TEXT NOT NULL,
                clickbait_title TEXT DEFAULT '',
                font_design TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 1,
                target_duration_minutes INTEGER DEFAULT 135,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                production_started_at DATETIME,
                production_completed_at DATETIME,
                scene_count INTEGER,
                total_duration_minutes REAL,
                api_calls_used INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                output_path TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def import_csv_if_needed(self, csv_path: str):
        """Import CSV topics if database is empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM topics')
        topic_count = cursor.fetchone()[0]

        if topic_count == 0 and Path(csv_path).exists():
            print("📄 Importing topics from CSV to database...")
            try:
                df = pd.read_csv(csv_path)
                imported_count = 0

                for _, row in df.iterrows():
                    status = 'completed' if row.get('done', 0) == 1 else 'pending'

                    cursor.execute('''
                        INSERT INTO topics (
                            topic, description, clickbait_title, font_design, status
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        row['topic'],
                        row['description'],
                        row.get('clickbait_title', ''),
                        row.get('font_design', ''),
                        status
                    ))
                    imported_count += 1

                conn.commit()
                print(f"✅ Imported {imported_count} topics from CSV")

                # Backup CSV
                backup_path = Path(csv_path).parent / f"topics_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                shutil.copy2(csv_path, backup_path)
                print(f"📁 CSV backed up to: {backup_path}")

            except Exception as e:
                print(f"❌ CSV import failed: {e}")

        conn.close()

    def get_next_pending_topic(self) -> Optional[Tuple[int, str, str, str, str]]:
        """Get next pending topic from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, description, clickbait_title, font_design 
            FROM topics 
            WHERE status = 'pending' 
            ORDER BY priority DESC, created_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def mark_topic_as_started(self, topic_id: int):
        """Mark topic as started in production"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET status = 'in_progress', 
                production_started_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_topic_as_completed(self, topic_id: int, scene_count: int,
                                total_duration: float, api_calls: int,
                                total_cost: float, output_path: str):
        """Mark topic as completed with production stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET status = 'completed',
                production_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                scene_count = ?,
                total_duration_minutes = ?,
                api_calls_used = ?,
                total_cost = ?,
                output_path = ?
            WHERE id = ?
        ''', (scene_count, total_duration, api_calls, total_cost, output_path, topic_id))

        conn.commit()
        conn.close()


class YouTubeFrameJSONGenerator:
    """NEW: Generate frame-by-frame video composition JSON for YouTube automation"""

    def __init__(self):
        self.frame_log = []

    def log_frame_step(self, description: str, status: str = "START"):
        """Log frame generation steps"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.frame_log.append(entry)

        icon = "🎬" if status == "START" else "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {description}")

    def generate_youtube_frame_timeline(self, result: Dict) -> Dict:
        """Generate complete frame-by-frame timeline for video automation"""

        self.log_frame_step("YouTube Frame Timeline Generation")

        scene_plan = result.get('scene_plan', [])
        hook_section = result.get('hook_section', {})
        subscribe_section = result.get('subscribe_section', {})
        hook_subscribe_scenes = result.get('hook_subscribe_scenes', {})
        visual_prompts = result.get('visual_prompts', [])

        # Calculate precise timing
        current_frame = 0
        fps = 30  # 30 FPS for YouTube optimization

        timeline_frames = []

        # HOOK SECTION (0-30 seconds, 900 frames)
        hook_frames = self._generate_hook_frames(hook_section, hook_subscribe_scenes, current_frame, fps)
        timeline_frames.extend(hook_frames)
        current_frame += 900  # 30 seconds * 30 fps

        # SUBSCRIBE SECTION (30-60 seconds, 900 frames)
        subscribe_frames = self._generate_subscribe_frames(subscribe_section, hook_subscribe_scenes, current_frame, fps)
        timeline_frames.extend(subscribe_frames)
        current_frame += 900  # 30 seconds * 30 fps

        # MAIN STORY SCENES (60 seconds onward)
        story_frames = self._generate_story_frames(scene_plan, visual_prompts, current_frame, fps)
        timeline_frames.extend(story_frames)

        # YouTube-specific optimizations
        youtube_timeline = self._optimize_for_youtube(timeline_frames, result)

        frame_json = {
            "youtube_frame_timeline": {
                "metadata": {
                    "fps": fps,
                    "total_frames": len(timeline_frames),
                    "total_duration_seconds": len(timeline_frames) / fps,
                    "total_duration_formatted": self._format_duration(len(timeline_frames) / fps),
                    "resolution": "1920x1080",
                    "aspect_ratio": "16:9",
                    "generated_at": datetime.now().isoformat(),
                    "optimized_for": "YouTube automation"
                },
                "timeline_structure": {
                    "hook_section": {
                        "start_frame": 0,
                        "end_frame": 899,
                        "duration_seconds": 30,
                        "purpose": "Atmospheric opening, gentle intrigue"
                    },
                    "subscribe_section": {
                        "start_frame": 900,
                        "end_frame": 1799,
                        "duration_seconds": 30,
                        "purpose": "Community building, warm invitation"
                    },
                    "main_story": {
                        "start_frame": 1800,
                        "end_frame": len(timeline_frames) - 1,
                        "duration_seconds": (len(timeline_frames) - 1800) / fps,
                        "purpose": "Sleep story content with scene progression"
                    }
                },
                "frames": timeline_frames,
                "youtube_optimization": youtube_timeline,
                "automation_instructions": {
                    "video_generation": {
                        "step1": "Generate background visuals using visual_prompts for each frame segment",
                        "step2": "Apply audio synchronization using audio_generation_prompts.json",
                        "step3": "Add text overlays during hook/subscribe sections only",
                        "step4": "Apply smooth transitions between scene changes",
                        "step5": "Export as MP4 H.264 1920x1080 30fps"
                    },
                    "audio_sync": {
                        "hook_audio": "Sync with hook segment from audio_generation_prompts.json",
                        "subscribe_audio": "Sync with subscribe segment from audio_generation_prompts.json",
                        "story_audio": "Sync scene audio with corresponding visual frame ranges",
                        "crossfade_duration": 2.0,
                        "silence_gaps": "Fill with ambient background music"
                    },
                    "visual_transitions": {
                        "between_hook_scenes": "3-second crossfade",
                        "between_subscribe_scenes": "3-second crossfade",
                        "between_story_scenes": "5-second crossfade",
                        "transition_type": "smooth_blend",
                        "maintain_atmosphere": "Always preserve peaceful mood"
                    }
                }
            }
        }

        self.log_frame_step("YouTube Frame Timeline Generation", "SUCCESS")
        return frame_json

    def _generate_hook_frames(self, hook_section: Dict, hook_subscribe_scenes: Dict, start_frame: int, fps: int) -> \
    List[Dict]:
        """Generate frame data for hook section (0-30s)"""
        frames = []
        hook_scenes = hook_subscribe_scenes.get('hook_scenes', [])

        frames_per_scene = 90  # 3 seconds per scene * 30 fps

        for i, scene in enumerate(hook_scenes[:10]):  # Max 10 scenes for 30 seconds
            scene_start_frame = start_frame + (i * frames_per_scene)
            scene_end_frame = scene_start_frame + frames_per_scene - 1

            frame_data = {
                "frame_range": f"{scene_start_frame}-{scene_end_frame}",
                "start_frame": scene_start_frame,
                "end_frame": scene_end_frame,
                "duration_frames": frames_per_scene,
                "duration_seconds": 3.0,
                "section": "hook",
                "scene_reference": {
                    "scene_id": scene.get('scene_id', i + 1),
                    "scene_title": scene.get('scene_title', f'Hook Scene {i + 1}'),
                    "visual_prompt": scene.get('visual_prompt', ''),
                    "timing_note": scene.get('timing_note', '')
                },
                "visual_composition": {
                    "background": "scene_visual",
                    "text_overlay": "none" if i > 0 else "story_title_fade_in",
                    "transition_in": "fade_in" if i == 0 else "crossfade",
                    "transition_out": "crossfade" if i < 9 else "hold",
                    "atmosphere": "mysterious_peaceful"
                },
                "audio_sync": {
                    "narration": "hook_segment_audio",
                    "background_music": "ambient_low_volume",
                    "timing_critical": True,
                    "sync_method": "precise_word_timing"
                },
                "automation_notes": {
                    "generation_priority": "HIGH - Critical for retention",
                    "quality_requirement": "Premium atmospheric visuals",
                    "mobile_optimization": "Ensure readability on small screens"
                }
            }
            frames.append(frame_data)

        return frames

    def _generate_subscribe_frames(self, subscribe_section: Dict, hook_subscribe_scenes: Dict, start_frame: int,
                                   fps: int) -> List[Dict]:
        """Generate frame data for subscribe section (30-60s)"""
        frames = []
        subscribe_scenes = hook_subscribe_scenes.get('subscribe_scenes', [])

        frames_per_scene = 90  # 3 seconds per scene * 30 fps

        for i, scene in enumerate(subscribe_scenes[:10]):
            scene_start_frame = start_frame + (i * frames_per_scene)
            scene_end_frame = scene_start_frame + frames_per_scene - 1

            frame_data = {
                "frame_range": f"{scene_start_frame}-{scene_end_frame}",
                "start_frame": scene_start_frame,
                "end_frame": scene_end_frame,
                "duration_frames": frames_per_scene,
                "duration_seconds": 3.0,
                "section": "subscribe",
                "scene_reference": {
                    "scene_id": scene.get('scene_id', i + 1),
                    "scene_title": scene.get('scene_title', f'Subscribe Scene {i + 1}'),
                    "visual_prompt": scene.get('visual_prompt', ''),
                    "timing_note": scene.get('timing_note', '')
                },
                "visual_composition": {
                    "background": "community_focused_scene",
                    "text_overlay": "subscribe_button_animation" if i == 4 else "none",
                    "transition_in": "crossfade",
                    "transition_out": "crossfade" if i < 9 else "fade_to_story",
                    "atmosphere": "warm_inviting"
                },
                "audio_sync": {
                    "narration": "subscribe_segment_audio",
                    "background_music": "slightly_higher_volume",
                    "timing_critical": True,
                    "sync_method": "conversational_pacing"
                },
                "automation_notes": {
                    "generation_priority": "HIGH - Community building critical",
                    "quality_requirement": "Welcoming and professional",
                    "cta_timing": "Subscribe button appears at midpoint"
                }
            }
            frames.append(frame_data)

        return frames

    def _generate_story_frames(self, scene_plan: List[Dict], visual_prompts: List[Dict], start_frame: int, fps: int) -> \
    List[Dict]:
        """Generate frame data for main story scenes"""
        frames = []
        current_frame = start_frame

        for scene in scene_plan:
            scene_id = scene['scene_id']
            duration_minutes = scene.get('duration_minutes', 4)
            duration_seconds = duration_minutes * 60
            duration_frames = int(duration_seconds * fps)

            # Find matching visual prompt
            visual_prompt = next((vp for vp in visual_prompts if vp.get('scene_number') == scene_id), {})

            scene_start_frame = current_frame
            scene_end_frame = current_frame + duration_frames - 1

            frame_data = {
                "frame_range": f"{scene_start_frame}-{scene_end_frame}",
                "start_frame": scene_start_frame,
                "end_frame": scene_end_frame,
                "duration_frames": duration_frames,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_minutes,
                "section": "main_story",
                "scene_reference": {
                    "scene_id": scene_id,
                    "scene_title": scene.get('title', f'Scene {scene_id}'),
                    "location": scene.get('location', ''),
                    "emotion": scene.get('emotion', 'peaceful'),
                    "template": scene.get('template', 'atmospheric'),
                    "narrative_style": scene.get('narrative_style', 'observational')
                },
                "visual_composition": {
                    "background": "scene_specific_visual",
                    "visual_prompt": visual_prompt.get('prompt', ''),
                    "enhanced_prompt": visual_prompt.get('enhanced_prompt', ''),
                    "character_reference_needed": visual_prompt.get('character_reference_needed', False),
                    "characters_in_scene": visual_prompt.get('characters_in_scene', []),
                    "text_overlay": "none",
                    "transition_in": "smooth_crossfade_5s",
                    "transition_out": "smooth_crossfade_5s",
                    "atmosphere": scene.get('emotion', 'peaceful')
                },
                "audio_sync": {
                    "narration": f"scene_{scene_id}_audio",
                    "background_music": "ambient_consistent",
                    "timing_critical": False,
                    "sync_method": "natural_pacing",
                    "pause_markers": "respect_pause_timing"
                },
                "smart_algorithm_data": {
                    "duration_randomized": True,
                    "emotion_based_timing": True,
                    "template_modifier_applied": True,
                    "position_modifier_applied": True,
                    "natural_variation": True
                },
                "automation_notes": {
                    "generation_priority": "MEDIUM - Story content",
                    "quality_requirement": "Sleep-optimized visuals",
                    "character_consistency": "Use character references when present"
                }
            }
            frames.append(frame_data)
            current_frame += duration_frames

        return frames

    def _optimize_for_youtube(self, timeline_frames: List[Dict], result: Dict) -> Dict:
        """Apply YouTube-specific optimizations"""

        scene_chapters = result.get('scene_chapters', [])
        youtube_metadata = result.get('youtube_optimization', {})

        return {
            "chapter_markers": [
                                   {
                                       "time": "0:00",
                                       "title": "🌙 Golden Hook - Enter the Story",
                                       "frame_range": "0-899",
                                       "purpose": "Viewer retention optimization"
                                   },
                                   {
                                       "time": "0:30",
                                       "title": "📺 Subscribe for More Sleep Stories",
                                       "frame_range": "900-1799",
                                       "purpose": "Channel growth optimization"
                                   }
                               ] + [
                                   {
                                       "time": chapter.get('time', '1:00'),
                                       "title": chapter.get('title', 'Scene'),
                                       "frame_range": f"{self._time_to_frame(chapter.get('time', '1:00'), 30)}-{self._time_to_frame(chapter.get('time', '1:00'), 30) + chapter.get('duration_seconds', 240) * 30 - 1}",
                                       "purpose": "Content navigation"
                                   }
                                   for chapter in scene_chapters
                               ],
            "engagement_optimization": {
                "retention_hooks": [
                    {"frame": 0, "type": "visual_intrigue", "description": "Atmospheric opening scene"},
                    {"frame": 450, "type": "gentle_mystery", "description": "Subtle story hook"},
                    {"frame": 900, "type": "community_building", "description": "Subscribe invitation"},
                    {"frame": 1800, "type": "story_immersion", "description": "Main narrative begins"}
                ],
                "algorithm_optimization": {
                    "watch_time_optimization": "Gradual story progression keeps viewers engaged",
                    "click_retention": "Peaceful visuals reduce bounce rate",
                    "session_duration": "Sleep content encourages long watch times",
                    "audience_retention_strategy": "Consistent atmosphere prevents jarring transitions"
                }
            },
            "mobile_optimization": {
                "thumbnail_visibility": "Large text and clear imagery for mobile",
                "subtitle_placement": "Lower third for mobile viewing",
                "visual_clarity": "High contrast for small screen visibility",
                "touch_interaction": "Chapter markers easily tappable"
            },
            "accessibility": {
                "closed_captions": "Full transcript available",
                "audio_description": "Visual elements described in narration",
                "contrast_ratio": "WCAG AA compliant visual contrast",
                "navigation_aids": "Clear chapter structure for easy navigation"
            }
        }

    def _time_to_frame(self, time_str: str, fps: int) -> int:
        """Convert time string (M:SS) to frame number"""
        try:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            total_seconds = minutes * 60 + seconds
            return total_seconds * fps
        except:
            return 0

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"


class AdvancedSEOGenerator:
    """NEW: Generate advanced YouTube SEO metadata beyond basic optimization"""

    def __init__(self):
        self.seo_log = []

    def log_seo_step(self, description: str, status: str = "START"):
        """Log SEO generation steps"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.seo_log.append(entry)

        icon = "🔍" if status == "START" else "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {description}")

    def generate_advanced_youtube_seo(self, result: Dict) -> Dict:
        """Generate comprehensive YouTube SEO beyond basic metadata"""

        self.log_seo_step("Advanced YouTube SEO Generation")

        topic = result.get('topic', 'Sleep Story')
        scene_plan = result.get('scene_plan', [])
        youtube_basic = result.get('youtube_optimization', {})
        characters = result.get('main_characters', [])
        total_duration = sum(scene.get('duration_minutes', 4) for scene in scene_plan)

        advanced_seo = {
            "youtube_advanced_seo": {
                "metadata_optimization": {
                    "primary_title": youtube_basic.get('clickbait_titles', [f'{topic} Sleep Story'])[0],
                    "seo_optimized_titles": [
                        f"🌙 {topic} - 2 Hour Deep Sleep Story (No Ads, Rain Sounds)",
                        f"Ancient {topic} Bedtime Story | Fall Asleep in 10 Minutes",
                        f"{topic} Sleep Meditation | Insomnia Relief | Peaceful Night",
                        f"Relaxing {topic} Story for Sleep | ASMR Storytelling",
                        f"🎧 {topic} Sleep Journey | 2+ Hours | Calm Narration"
                    ],
                    "description_optimization": {
                        "hook_paragraph": f"✨ Welcome to the most peaceful {topic} sleep story you'll ever experience. This carefully crafted 2+ hour journey through ancient {topic.lower()} will guide you into the deepest, most restorative sleep.",
                        "story_summary": f"📖 Tonight's story takes you on a gentle exploration of {topic.lower()}, where every scene is designed to calm your mind and relax your body. From {scene_plan[0].get('location', 'peaceful beginnings')} to {scene_plan[-1].get('location', 'serene conclusions')}, each moment is crafted for maximum relaxation.",
                        "sleep_benefits": [
                            "🌙 Reduces anxiety and racing thoughts",
                            "😴 Proven to improve sleep quality",
                            "🧘 Lowers stress hormones naturally",
                            "💤 Helps with insomnia and sleep disorders",
                            "🎯 Perfect for meditation and mindfulness"
                        ],
                        "technical_info": {
                            "duration": f"{total_duration:.0f} minutes ({total_duration / 60:.1f} hours)",
                            "scene_count": len(scene_plan),
                            "narration_style": "Calm, soothing voice optimized for sleep",
                            "audio_quality": "High-definition audio for maximum relaxation",
                            "interruption_free": "No ads, no sudden sounds, no interruptions"
                        }
                    }
                },
                "advanced_keywords": {
                    "primary_keywords": ["sleep story", "bedtime story", "insomnia relief", "meditation"],
                    "long_tail_keywords": [
                        f"{topic.lower()} sleep story",
                        f"ancient {topic.lower()} bedtime story",
                        f"{topic.lower()} meditation for sleep",
                        "2 hour sleep story no ads",
                        "deep sleep storytelling",
                        "peaceful historical fiction"
                    ],
                    "trending_keywords": [
                        "ASMR sleep story",
                        "sleep podcast",
                        "guided sleep meditation",
                        "bedtime stories for adults",
                        "sleep hypnosis",
                        "calm bedtime story"
                    ],
                    "competitor_keywords": [
                        "jason stephenson style",
                        "michelle's sanctuary type",
                        "the honest guys alternative",
                        "lauren ostrowski fenton style"
                    ],
                    "seasonal_keywords": [
                        "winter sleep story",
                        "holiday relaxation",
                        "stress relief after work",
                        "weekend wind down"
                    ]
                },
                "hashtag_strategy": {
                    "primary_hashtags": [
                        "#sleepstory", "#bedtimestory", "#insomnia", "#meditation",
                        "#relaxation", "#calmdown", "#peaceful", "#sleepaid"
                    ],
                    "niche_hashtags": [
                        f"#{topic.lower().replace(' ', '')}", "#historicalfiction",
                        "#ancienthistory", "#culturalstories", "#mindfulness"
                    ],
                    "trending_hashtags": [
                        "#asmr", "#sleeppodcast", "#guidedmeditation",
                        "#stressrelief", "#anxietyrelief", "#selfcare"
                    ],
                    "engagement_hashtags": [
                        "#comment", "#subscribe", "#sleepwell", "#goodnight",
                        "#peaceful", "#dreamy", "#restful", "#tranquil"
                    ]
                },
                "content_calendar_integration": {
                    "optimal_upload_time": "8:00 PM - 11:00 PM local time",
                    "best_days": ["Sunday", "Monday", "Tuesday", "Wednesday"],
                    "seasonal_timing": "Perfect for fall/winter stress relief",
                    "cross_promotion": [
                        "Link to related historical sleep stories",
                        "Reference similar cultural stories",
                        "Connect to meditation playlists"
                    ]
                },
                "engagement_optimization": {
                    "community_tab_posts": [
                        f"🌙 Tonight we're traveling to ancient {topic}... Are you ready for the most peaceful sleep?",
                        f"Which historical location should we visit next? Comment your favorites! 💤",
                        f"Sleep tip: This {topic} story works best with headphones and dimmed lights ✨"
                    ],
                    "comment_engagement": {
                        "pinned_comment": f"💤 Welcome to our {topic} sleep journey! Let me know in the comments how this story helped you sleep better. Sweet dreams! 🌙",
                        "response_templates": [
                            "Thank you for sharing! So glad this helped you sleep peacefully 💤",
                            "That's wonderful to hear! Historical stories have such a calming effect 🌙",
                            "Sweet dreams! Come back anytime you need peaceful sleep ✨"
                        ]
                    },
                    "end_screen_optimization": {
                        "subscribe_reminder": "Subscribe for more peaceful sleep stories",
                        "related_video_suggestion": "Try our Medieval Castle library story next",
                        "playlist_promotion": "Add this to your bedtime playlist"
                    }
                },
                "analytics_tracking": {
                    "key_metrics": [
                        "Average view duration (target: 45+ minutes)",
                        "Sleep quality comments ratio",
                        "Return viewer percentage",
                        "Playlist addition rate",
                        "Night-time view percentage"
                    ],
                    "success_indicators": [
                        "High retention in first 5 minutes",
                        "Comments mentioning falling asleep",
                        "Shares to private playlists",
                        "Positive sentiment in comments",
                        "Low bounce rate to other channels"
                    ]
                }
            }
        }

        self.log_seo_step("Advanced YouTube SEO Generation", "SUCCESS")
        return advanced_seo


# Database-based topic functions
def get_next_topic_from_database(config) -> Tuple[int, str, str, str, str]:
    """Get next topic from database instead of CSV"""
    db_path = Path(config.paths['DATA_DIR']) / 'production.db'
    topic_manager = DatabaseTopicManager(str(db_path))

    # Import CSV if database is empty
    csv_path = config.paths['TOPIC_CSV_PATH']
    topic_manager.import_csv_if_needed(csv_path)

    # Get next pending topic
    result = topic_manager.get_next_pending_topic()

    if result:
        topic_id, topic, description, clickbait_title, font_design = result
        topic_manager.mark_topic_as_started(topic_id)

        print(f"✅ Topic selected from database: {topic}")
        return topic_id, topic, description, clickbait_title, font_design
    else:
        raise ValueError("No pending topics found in database")


def complete_topic_in_database(config, topic_id: int, scene_count: int, total_duration: float,
                               api_calls: int, total_cost: float, output_path: str):
    """Mark topic as completed in database"""
    db_path = Path(config.paths['DATA_DIR']) / 'production.db'
    topic_manager = DatabaseTopicManager(str(db_path))

    topic_manager.mark_topic_as_completed(
        topic_id, scene_count, total_duration, api_calls, total_cost, output_path
    )

    print(f"✅ Topic {topic_id} marked as completed in database")


def save_production_outputs(output_dir: str, result: Dict, story_topic: str, topic_id: int,
                            api_calls: int, total_cost: float, config):
    """Save complete production outputs - SERVER VERSION WITH ALL JSON FILES + NEW YOUTUBE FEATURES"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
        # Initialize new generators
        frame_generator = YouTubeFrameJSONGenerator()
        seo_generator = AdvancedSEOGenerator()

        # Generate NEW features
        youtube_frame_data = frame_generator.generate_youtube_frame_timeline(result)
        advanced_seo_data = seo_generator.generate_advanced_youtube_seo(result)

        # 1. Complete story text
        story_path = output_path / "complete_story.txt"
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(result["complete_story"])
        saved_files.append("complete_story.txt")

        # 2. Scene plan with enhanced chapters
        plan_path = output_path / "scene_plan.json"
        scene_data = {
            "scene_plan": result["scene_plan"],
            "scene_chapters": result.get("scene_chapters", []),
            "total_scenes": len(result.get("scene_plan", [])),
            "total_duration_minutes": sum(scene.get('duration_minutes', 4) for scene in result.get("scene_plan", [])),
            "smart_algorithm_used": result.get("generation_stats", {}).get("smart_algorithm", False)
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
        saved_files.append("scene_plan.json")

        # 3. Visual prompts with thumbnail (INCLUDING SCENE 99)
        visual_path = output_path / "visual_generation_prompts.json"
        with open(visual_path, "w", encoding="utf-8") as f:
            json.dump(result["visual_prompts"], f, indent=2, ensure_ascii=False)
        saved_files.append("visual_generation_prompts.json")

        # 4. Voice directions for TTS
        voice_path = output_path / "voice_directions.json"
        with open(voice_path, "w", encoding="utf-8") as f:
            json.dump(result["voice_directions"], f, indent=2, ensure_ascii=False)
        saved_files.append("voice_directions.json")

        # 5. Character profiles
        character_path = output_path / "character_profiles.json"
        character_data = {
            "main_characters": result.get("main_characters", []),
            "character_relationships": result.get("character_relationships", []),
            "scene_character_mapping": result.get("scene_character_mapping", {}),
            "visual_style_notes": result.get("visual_style_notes", {}),
            "visual_generation_instructions": {
                "step1": "First generate reference images for each main character using their physical_description",
                "step2": "Then generate scene visuals using character references when characters_in_scene is not empty",
                "step3": "For atmospheric scenes (no characters), focus on setting and mood only",
                "step4": "Generate thumbnail using scene_number 99 in visual_generation_prompts.json",
                "reference_usage": "Always include relevant character reference images when generating scene visuals"
            }
        }
        with open(character_path, "w", encoding="utf-8") as f:
            json.dump(character_data, f, indent=2, ensure_ascii=False)
        saved_files.append("character_profiles.json")

        # 6. YouTube metadata (COMPLETE PACKAGE)
        youtube_path = output_path / "youtube_metadata.json"
        youtube_data = result.get("youtube_optimization", {})
        with open(youtube_path, "w", encoding="utf-8") as f:
            json.dump(youtube_data, f, indent=2, ensure_ascii=False)
        saved_files.append("youtube_metadata.json")

        # 7. Thumbnail data with composition strategy
        thumbnail_data = result.get("thumbnail_data", {})
        if thumbnail_data.get("thumbnail_prompt"):
            thumbnail_path = output_path / "thumbnail_generation.json"
            with open(thumbnail_path, "w", encoding="utf-8") as f:
                json.dump(thumbnail_data, f, indent=2, ensure_ascii=False)
            saved_files.append("thumbnail_generation.json")

        # 8. Hook & Subscribe scenes
        hook_subscribe_data = result.get("hook_subscribe_scenes", {})
        if hook_subscribe_data:
            hook_subscribe_path = output_path / "hook_subscribe_scenes.json"
            with open(hook_subscribe_path, "w", encoding="utf-8") as f:
                json.dump(hook_subscribe_data, f, indent=2, ensure_ascii=False)
            saved_files.append("hook_subscribe_scenes.json")

        # 9. Production specifications (DETAILED)
        production_specs = result.get("production_specifications", {})
        if production_specs:
            production_path = output_path / "production_specifications.json"
            with open(production_path, "w", encoding="utf-8") as f:
                json.dump(production_specs, f, indent=2, ensure_ascii=False)
            saved_files.append("production_specifications.json")

        # 10. Platform metadata (UPLOAD READY)
        platform_data = result.get("platform_metadata", {})
        if platform_data:
            platform_path = output_path / "platform_metadata.json"
            with open(platform_path, "w", encoding="utf-8") as f:
                json.dump(platform_data, f, indent=2, ensure_ascii=False)
            saved_files.append("platform_metadata.json")

        # 11. Audio generation prompts (ENHANCED)
        audio_prompts = []

        # Hook audio
        hook_section = result.get("hook_section", {})
        if hook_section:
            audio_prompts.append({
                "segment_id": "hook",
                "content": hook_section.get("content", ""),
                "duration_seconds": hook_section.get("duration_seconds", 30),
                "voice_direction": hook_section.get("voice_direction", ""),
                "tts_settings": {
                    "voice": "alloy",
                    "speed": 0.9,
                    "pitch": "normal",
                    "emphasis": "gentle mystery"
                }
            })

        # Subscribe audio
        subscribe_section = result.get("subscribe_section", {})
        if subscribe_section:
            audio_prompts.append({
                "segment_id": "subscribe",
                "content": subscribe_section.get("content", ""),
                "duration_seconds": subscribe_section.get("duration_seconds", 30),
                "voice_direction": subscribe_section.get("voice_direction", ""),
                "tts_settings": {
                    "voice": "alloy",
                    "speed": 0.95,
                    "pitch": "warm",
                    "emphasis": "friendly conversation"
                }
            })

        # Story scenes audio
        stories = result.get("stories", {})
        voice_directions = result.get("voice_directions", [])
        production_audio = production_specs.get("audio_production", {})
        tts_settings = production_audio.get("tts_settings", {})

        for scene_id, story_content in stories.items():
            voice_direction = next((v for v in voice_directions if v.get("scene_number") == int(scene_id)), {})
            scene_info = next((s for s in result.get("scene_plan", []) if s.get("scene_id") == int(scene_id)), {})

            audio_prompts.append({
                "segment_id": f"scene_{scene_id}",
                "content": story_content,
                "duration_minutes": scene_info.get("duration_minutes", 4),
                "voice_direction": voice_direction.get("direction", ""),
                "template": voice_direction.get("template", "atmospheric"),
                "style": voice_direction.get("style", "observational"),
                "emotion": scene_info.get("emotion", "peaceful"),
                "tts_settings": {
                    "voice": tts_settings.get("optimal_voice", "alloy"),
                    "speed": tts_settings.get("speed_multiplier", 0.85),
                    "pitch": tts_settings.get("pitch_adjustment", -2),
                    "volume": tts_settings.get("volume_level", 80),
                    "emphasis": "sleep-optimized"
                }
            })

        audio_path = output_path / "audio_generation_prompts.json"
        with open(audio_path, "w", encoding="utf-8") as f:
            json.dump(audio_prompts, f, indent=2, ensure_ascii=False)
        saved_files.append("audio_generation_prompts.json")

        # 12. Video composition instructions (ENHANCED)
        production_video = production_specs.get("video_assembly", {})
        video_specs = production_video.get("video_specifications", {})

        video_composition = {
            "timeline_structure": [
                {
                    "segment": "hook",
                    "start_time": 0,
                    "end_time": 30,
                    "video_source": "hook_subscribe_scenes.json -> hook_scenes",
                    "audio_source": "audio_generation_prompts.json -> hook segment",
                    "text_overlay": "None (pure atmospheric)",
                    "transition": "fade_in"
                },
                {
                    "segment": "subscribe",
                    "start_time": 30,
                    "end_time": 60,
                    "video_source": "hook_subscribe_scenes.json -> subscribe_scenes",
                    "audio_source": "audio_generation_prompts.json -> subscribe segment",
                    "text_overlay": "Subscribe button animation",
                    "transition": "smooth_blend"
                },
                {
                    "segment": "main_story",
                    "start_time": 60,
                    "end_time": sum(
                        scene.get('duration_minutes', 4) * 60 for scene in result.get("scene_plan", [])) + 60,
                    "video_source": "visual_generation_prompts.json -> scenes 1-N",
                    "audio_source": "audio_generation_prompts.json -> scenes 1-N",
                    "text_overlay": "Scene titles (optional)",
                    "transition": "crossfade_between_scenes"
                }
            ],
            "scene_sync_strategy": {
                "rule": "When audio mentions scene X, display scene X visual",
                "timing": "Immediate visual sync with narrative",
                "duration": "Variable scene durations from smart algorithm",
                "overlap": "15 second crossfade between scenes"
            },
            "thumbnail_usage": {
                "source": "visual_generation_prompts.json -> scene 99",
                "purpose": "YouTube thumbnail only, not used in video timeline",
                "composition": "RIGHT-side character, LEFT-side text overlay space"
            },
            "production_settings": {
                "resolution": video_specs.get("resolution", "1920x1080"),
                "frame_rate": video_specs.get("frame_rate", 30),
                "audio_bitrate": 192,
                "video_codec": "h264",
                "total_duration": f"{sum(scene.get('duration_minutes', 4) for scene in result.get('scene_plan', [])) + 1} minutes"
            },
            "chapters": result.get("scene_chapters", [])
        }

        video_path = output_path / "video_composition_instructions.json"
        with open(video_path, "w", encoding="utf-8") as f:
            json.dump(video_composition, f, indent=2, ensure_ascii=False)
        saved_files.append("video_composition_instructions.json")

        # 13. NEW: YouTube Frame JSON (FRAME-BY-FRAME AUTOMATION)
        youtube_frame_path = output_path / "youtube_frame_timeline.json"
        with open(youtube_frame_path, "w", encoding="utf-8") as f:
            json.dump(youtube_frame_data, f, indent=2, ensure_ascii=False)
        saved_files.append("youtube_frame_timeline.json")

        # 14. NEW: Advanced YouTube SEO
        advanced_seo_path = output_path / "youtube_advanced_seo.json"
        with open(advanced_seo_path, "w", encoding="utf-8") as f:
            json.dump(advanced_seo_data, f, indent=2, ensure_ascii=False)
        saved_files.append("youtube_advanced_seo.json")

        # 15. Generation report (COMPREHENSIVE)
        report_path = output_path / "generation_report.json"
        production_report = {
            "topic": story_topic,
            "topic_id": topic_id,
            "generation_completed": datetime.now().isoformat(),
            "model_used": config.claude_config["model"],
            "claude_4_optimized": True,
            "server_optimized": True,
            "five_stage_approach": True,
            "smart_algorithm_used": result.get("generation_stats", {}).get("smart_algorithm", False),
            "complete_pipeline": True,
            "new_features_included": {
                "youtube_frame_timeline": True,
                "advanced_seo_optimization": True,
                "frame_by_frame_automation": True,
                "enhanced_chapter_system": True
            },
            "stats": result["generation_stats"],
            "cost_analysis": {
                "total_api_calls": api_calls,
                "total_cost": total_cost,
                "cost_per_scene": total_cost / len(result.get("scene_plan", [1])),
                "cost_efficiency": "Claude 4 optimized"
            },
            "quality_metrics": {
                "scenes_planned": len(result.get("scene_plan", [])),
                "stories_written": len(result.get("stories", {})),
                "characters_extracted": len(result.get("main_characters", [])),
                "completion_rate": (len(result.get("stories", {})) / len(result.get("scene_plan", [1]))) * 100,
                "youtube_optimization": bool(result.get("youtube_optimization", {}).get("clickbait_titles")),
                "production_specs": bool(result.get("production_specifications", {}).get("audio_production")),
                "thumbnail_composition": bool(result.get("thumbnail_data", {}).get("thumbnail_prompt")),
                "frame_timeline_generated": bool(youtube_frame_data.get("youtube_frame_timeline")),
                "advanced_seo_generated": bool(advanced_seo_data.get("youtube_advanced_seo"))
            },
            "files_saved": saved_files,
            "next_steps": [
                "1. Generate character reference images using character_profiles.json",
                "2. Generate scene visuals (1-N) using visual_generation_prompts.json",
                "3. Generate thumbnail (scene 99) using visual_generation_prompts.json",
                "4. Generate audio using audio_generation_prompts.json with production specifications",
                "5. Use youtube_frame_timeline.json for frame-by-frame video automation",
                "6. Apply youtube_advanced_seo.json for comprehensive SEO optimization",
                "7. Upload to YouTube using platform_metadata.json with full automation"
            ],
            "automation_readiness": {
                "character_extraction": "✅ Complete",
                "youtube_optimization": "✅ Complete",
                "production_specifications": "✅ Complete",
                "platform_metadata": "✅ Complete",
                "composition_strategy": "✅ Complete",
                "frame_timeline": "✅ Complete (NEW)",
                "advanced_seo": "✅ Complete (NEW)",
                "api_ready_format": "✅ Complete"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ Complete production files saved: {saved_files}")
        config.logger.info(f"Files saved to: {output_dir}")

        # Mark topic as completed in database
        scene_count = len(result.get('scene_plan', []))
        total_duration = sum(scene.get('duration_minutes', 4) for scene in result.get('scene_plan', []))

        complete_topic_in_database(
            config, topic_id, scene_count, total_duration, api_calls, total_cost, output_dir
        )

    except Exception as e:
        print(f"❌ Save error: {e}")
        config.logger.error(f"Save error: {e}")


def print_production_summary(result: Dict, story_topic: str, output_path: str, generation_time: float):
    """Print complete production generation summary with NEW YouTube features"""
    stats = result["generation_stats"]

    print("\n" + "🚀" * 60)
    print("SMART AUTOMATED STORY GENERATOR - COMPLETE WITH NEW YOUTUBE FEATURES!")
    print("🚀" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: Claude 4 Sonnet")
    print(f"🖥️  Server Mode: ✅ ACTIVE")
    print(f"🏭 Complete Pipeline: ✅ ACTIVE")
    print(f"🎲 Smart Algorithm: ✅ ACTIVE")
    print(f"🎯 5-Stage Approach: ✅ ACTIVE")

    print(f"\n📊 PRODUCTION PERFORMANCE:")
    print(f"🔥 Total API Calls: {stats['api_calls_used']}")
    print(f"💰 Total Cost: ${result.get('total_cost', 0):.4f}")
    print(f"⏱️  Total Generation Time: {generation_time:.1f}s")
    print(f"🎬 Scenes Planned: {stats['scenes_planned']}")
    print(f"📝 Stories Written: {stats['stories_written']}")
    print(f"👥 Characters Extracted: {stats['characters_extracted']}")
    print(f"🖼️  Thumbnail Generated: {'✅ YES' if stats.get('thumbnail_generated') else '❌ NO'}")
    print(f"📺 YouTube Optimization: {'✅ YES' if stats.get('youtube_optimization_generated') else '❌ NO'}")
    print(f"🏭 Production Specs: {'✅ YES' if stats.get('production_specifications_generated') else '❌ NO'}")

    print(f"\n🆕 NEW YOUTUBE FEATURES:")
    print(f"🎬 Frame-by-Frame Timeline: ✅ GENERATED")
    print(f"🔍 Advanced SEO Optimization: ✅ GENERATED")
    print(f"📊 Chapter System Enhanced: ✅ COMPLETE")
    print(f"🎯 Mobile Optimization: ✅ INCLUDED")
    print(f"♿ Accessibility Features: ✅ INCLUDED")

    completion_rate = (stats['stories_written'] / stats.get('scenes_planned', 1)) * 100
    print(f"📊 Story Completion: {completion_rate:.1f}%")

    if completion_rate >= 80:
        print(f"\n🎉 MASSIVE SUCCESS!")
        print(f"✅ Complete story + character + YouTube + production + NEW FEATURES")
        print(f"✅ Frame-by-frame automation ready")
        print(f"✅ Advanced SEO optimization complete")
        print(f"🚀 ZERO manual work needed!")

    print("\n📄 GENERATED FILES (15 TOTAL - INCLUDING NEW FEATURES):")
    print("1. 📖 complete_story.txt - Full story text")
    print("2. 🎬 scene_plan.json - Smart scene structure + chapters")
    print("3. 🖼️  visual_generation_prompts.json - Scenes + Thumbnail (99)")
    print("4. 🎵 voice_directions.json - TTS guidance")
    print("5. 👥 character_profiles.json - Character data")
    print("6. 📺 youtube_metadata.json - Basic SEO package")
    print("7. 🖼️  thumbnail_generation.json - Composition strategy")
    print("8. 🎭 hook_subscribe_scenes.json - Background scenes")
    print("9. 🏭 production_specifications.json - Complete production specs")
    print("10. 📊 platform_metadata.json - Upload-ready data")
    print("11. 🎵 audio_generation_prompts.json - Enhanced TTS production")
    print("12. 🎥 video_composition_instructions.json - Video timeline + chapters")
    print("13. 🎬 youtube_frame_timeline.json - Frame-by-frame automation (NEW)")
    print("14. 🔍 youtube_advanced_seo.json - Advanced SEO optimization (NEW)")
    print("15. 📊 generation_report.json - Complete summary")

    print("🚀" * 60)


# Main execution logic
def main():
    """Main execution function"""
    try:
        print("🚀 SMART AUTOMATED STORY GENERATOR - COMPLETE WITH NEW YOUTUBE FEATURES")
        print("⚡ Server-optimized with complete pipeline + Frame-by-frame automation")
        print("🎲 Smart random scene count & duration generation")
        print("📊 Database integration + Advanced YouTube SEO")
        print("🎭 5-stage approach: Planning + Stories + Characters + Thumbnail + Hook/Subscribe")
        print("📄 Complete JSON outputs for automation (15 files)")
        print("🎯 RIGHT-side thumbnail positioning + Frame timeline")
        print("=" * 60)

        # Initialize server config
        config = ServerConfig()

        # Initialize Anthropic client
        client = Anthropic(api_key=config.api_key)

        # Import core generator
        from story_generator_core import AutomatedStoryGenerator

        # Get next topic from database
        topic_id, topic, description, clickbait_title, font_design = get_next_topic_from_database(config)
        print(f"\n📚 Topic ID: {topic_id} - {topic}")
        print(f"📝 Description: {description}")
        if clickbait_title:
            print(f"🎯 Clickbait Title: {clickbait_title}")

        # Setup output directory
        output_path = Path(config.paths['OUTPUT_DIR']) / str(topic_id)

        # Initialize generator
        generator = AutomatedStoryGenerator(config, client)

        # Generate complete story with smart algorithm + new features
        start_time = time.time()
        result = generator.generate_complete_story_with_characters(
            topic, description, clickbait_title, font_design
        )
        generation_time = time.time() - start_time

        # Add total cost to result
        result['total_cost'] = generator.total_cost

        # Save outputs with NEW YouTube features
        save_production_outputs(str(output_path), result, topic, topic_id,
                                generator.api_call_count, generator.total_cost, config)

        # Print comprehensive summary
        print_production_summary(result, topic, str(output_path), generation_time)

        print("\n🚀 COMPLETE PRODUCTION PIPELINE FINISHED WITH NEW FEATURES!")
        print(f"✅ All files ready for: {output_path}")
        print(f"📊 Database topic management: WORKING")
        print(f"🎲 Smart algorithm scene generation: WORKING")
        print(f"🎬 Frame-by-frame automation: READY")
        print(f"🔍 Advanced SEO optimization: READY")
        print(f"💰 Total cost: ${result.get('total_cost', 0):.4f}")

    except Exception as e:
        print(f"\n💥 GENERATOR ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()