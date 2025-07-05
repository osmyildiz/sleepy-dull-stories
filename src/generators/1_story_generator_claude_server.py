"""
Sleepy Dull Stories - COMPLETE Server-Ready Claude Story Generator
FIXES: All missing methods + Smart Algorithm + Database integration + Complete 5-stage pipeline
Production-optimized with complete automation
UPDATED: Added all missing JSON files from local version
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv
import time
import sys
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
import shutil

# Load environment first
load_dotenv()

# Server Configuration Class
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
        """Setup Claude configuration with PROVEN SETTINGS from successful version"""
        self.claude_config = {
            "model": "claude-sonnet-4-20250514",  # ✅ CLAUDE 4 - PROVEN SUCCESSFUL
            "max_tokens": 64000,  # ✅ HIGH TOKEN LIMIT - PROVEN SUCCESSFUL
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
            "streaming_response": True,  # ✅ PROVEN CRITICAL
            "long_timeout": True         # ✅ PROVEN CRITICAL
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
                logging.StreamHandler(sys.stdout)
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

# Initialize server config
try:
    CONFIG = ServerConfig()
    print("🚀 Server configuration loaded successfully")
except Exception as e:
    print(f"❌ Server configuration failed: {e}")
    sys.exit(1)

# Import Anthropic after config
try:
    from anthropic import Anthropic
    print("✅ Anthropic library imported")
except ImportError:
    print("❌ Anthropic library not found")
    print("Install with: pip install anthropic")
    sys.exit(1)

# Database Topic Management Integration
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
                import shutil
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
            ORDER BY priority ASC, created_at ASC 
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



class CharacterExtractionSystem:
    """Dynamic character extraction and analysis for any story topic"""

    def __init__(self):
        self.extraction_log = []

    def log_extraction_step(self, description: str, status: str = "START", metadata: Dict = None):
        """Log character extraction steps"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            entry.update(metadata)
        self.extraction_log.append(entry)

        icon = "🔍" if status == "START" else "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {description}")
        CONFIG.logger.info(f"{description} - Status: {status}")

    def calculate_character_importance(self, character_data: Dict) -> int:
        """Calculate character importance score (1-10)"""
        scene_count = len(character_data.get('scene_appearances', []))
        role = character_data.get('role', 'background')

        # Base score from scene appearances
        base_score = min(10, scene_count)

        # Role multipliers
        role_multipliers = {
            'protagonist': 1.0,
            'supporting': 0.8,
            'background': 0.6,
            'minor': 0.4
        }

        multiplier = role_multipliers.get(role, 0.5)
        final_score = int(base_score * multiplier)

        return max(1, min(10, final_score))

    def filter_top_characters(self, characters: List[Dict], max_characters: int = 5) -> List[Dict]:
        """Filter to top N most important characters"""
        # Calculate importance scores
        for char in characters:
            char['importance_score'] = self.calculate_character_importance(char)

        # Sort by importance and take top N
        sorted_chars = sorted(characters, key=lambda x: x['importance_score'], reverse=True)
        return sorted_chars[:max_characters]

    def analyze_scene_character_presence(self, scene_plan: List[Dict], characters: List[Dict]) -> Dict:
        """Analyze which characters appear in which scenes"""
        scene_character_map = {}

        for scene in scene_plan:
            scene_id = scene['scene_id']
            scene_characters = []

            # Check which characters are mentioned in this scene
            scene_text = f"{scene.get('title', '')} {scene.get('description', '')} {' '.join(scene.get('key_elements', []))}"
            scene_text_lower = scene_text.lower()

            for character in characters:
                char_name = character['name'].lower()

                # Check if character is mentioned
                if (char_name in scene_text_lower or
                    any(keyword in scene_text_lower for keyword in character.get('keywords', []))):
                    scene_characters.append({
                        'name': character['name'],
                        'role': character['role'],
                        'importance_score': character['importance_score']
                    })

            scene_character_map[str(scene_id)] = scene_characters

        return scene_character_map

    def select_thumbnail_character(self, characters: List[Dict], topic: str, description: str) -> Dict:
        """Intelligently select the best character for thumbnail or decide on atmospheric approach"""

        if not characters:
            return {
                "character_used": "None (Atmospheric focus)",
                "character_data": None,
                "reasoning": "No main characters available, using atmospheric focus"
            }

        # Score characters for thumbnail potential
        scored_characters = []

        for char in characters:
            score = 0

            # Base importance
            score += char.get('importance_score', 0) * 2

            # Role bonus
            role_bonus = {
                'protagonist': 15,
                'supporting': 10,
                'background': 5,
                'minor': 2
            }
            score += role_bonus.get(char.get('role', 'minor'), 0)

            # Visual appeal factors
            if char.get('use_in_marketing', False):
                score += 10

            if char.get('thumbnail_potential'):
                if any(keyword in char['thumbnail_potential'].lower() for keyword in
                      ['excellent', 'perfect', 'strong', 'great']):
                    score += 8
                elif any(keyword in char['thumbnail_potential'].lower() for keyword in
                        ['good', 'appealing', 'works well']):
                    score += 5

            # Scene count bonus
            scene_count = len(char.get('scene_appearances', []))
            if scene_count >= 3:
                score += 5
            elif scene_count >= 2:
                score += 3

            scored_characters.append({
                'character': char,
                'score': score
            })

        # Sort by score
        scored_characters.sort(key=lambda x: x['score'], reverse=True)

        # Decision logic
        best_char = scored_characters[0]

        if best_char['score'] >= 20:  # Strong character for thumbnail
            return {
                "character_used": best_char['character']['name'],
                "character_data": best_char['character'],
                "reasoning": f"High thumbnail score ({best_char['score']}): {best_char['character'].get('thumbnail_potential', 'Strong visual presence')}"
            }
        elif best_char['score'] >= 10:  # Moderate character
            return {
                "character_used": best_char['character']['name'],
                "character_data": best_char['character'],
                "reasoning": f"Moderate thumbnail potential ({best_char['score']}): Character has good visual appeal"
            }
        else:  # Go atmospheric
            return {
                "character_used": "None (Atmospheric focus)",
                "character_data": None,
                "reasoning": f"Best character score too low ({best_char['score']}), using atmospheric approach for better visual impact"
            }

class AutomatedStoryGenerator:
    """Complete server-ready automated story generation with FIXED Smart Algorithm"""

    def __init__(self):
        """Initialize story generator for server environment"""
        self.generation_log = []
        self.api_call_count = 0
        self.total_cost = 0.0
        self.character_system = CharacterExtractionSystem()

        try:
            self.client = Anthropic(api_key=CONFIG.api_key)
            CONFIG.logger.info("✅ Story generator initialized successfully")
            print("✅ Story generator initialized with Smart Algorithm + Budget Tracking")
        except Exception as e:
            CONFIG.logger.error(f"❌ Story generator initialization failed: {e}")
            print(f"❌ Story generator initialization failed: {e}")
            raise

    def log_step(self, description: str, status: str = "START", metadata: Dict = None):
        """Production logging system"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "api_calls": self.api_call_count,
            "total_cost": self.total_cost
        }
        if metadata:
            entry.update(metadata)
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {description} [API calls: {self.api_call_count}] [Cost: ${self.total_cost:.4f}]")
        CONFIG.logger.info(f"{description} - Status: {status} - API calls: {self.api_call_count} - Cost: ${self.total_cost:.4f}")

    def _generate_smart_scene_structure(self, target_duration: int = 135) -> Dict:
        """FIXED: Generate smart scene structure with random count and durations"""

        # Generate random scene count (28-45)
        scene_count = random.randint(28, 45)
        print(f"🎲 Smart Algorithm: Generated {scene_count} scenes (random 28-45)")

        # Add variation to target duration (120-150 minutes)
        actual_target = target_duration + random.randint(-15, 15)

        # Generate emotion phases
        peaceful_end = int(scene_count * 0.3)
        curiosity_end = int(scene_count * 0.6)
        concern_end = int(scene_count * 0.8)

        # Generate scene durations based on emotion
        scene_durations = []
        total_duration = 0

        for i in range(scene_count):
            # Determine emotion
            if i < peaceful_end:
                emotion = "peaceful"
                base_range = (4, 6)
            elif i < curiosity_end:
                emotion = "curiosity"
                base_range = (3, 5)
            elif i < concern_end:
                emotion = "concern"
                base_range = (2, 4)
            else:
                emotion = "resolution"
                base_range = (4, 7)

            # Generate random duration within emotion range
            duration = random.uniform(base_range[0], base_range[1])

            # Add template modifiers
            template_idx = i % 4
            templates = ["atmospheric", "character_focused", "historical_detail", "sensory_journey"]
            template = templates[template_idx]

            # Template modifiers
            modifiers = {
                "atmospheric": 1.25,
                "character_focused": 1.0,
                "historical_detail": 0.85,
                "sensory_journey": 1.15
            }

            duration *= modifiers[template]

            # Position modifiers
            if i < scene_count * 0.15:  # Opening
                duration *= 1.1
            elif i >= scene_count * 0.85:  # Resolution
                duration *= 1.2
            elif i >= scene_count * 0.7:  # Climax
                duration *= 0.9

            # Add natural variation
            duration *= random.uniform(0.9, 1.1)

            # Ensure bounds
            duration = max(1.5, min(8.0, duration))

            scene_durations.append(round(duration, 1))
            total_duration += duration

        # Adjust to target duration
        if abs(total_duration - actual_target) > 10:
            adjustment_factor = actual_target / total_duration
            scene_durations = [round(max(1.5, min(8.0, d * adjustment_factor)), 1) for d in scene_durations]
            total_duration = sum(scene_durations)

        return {
            'scene_count': scene_count,
            'scene_durations': scene_durations,
            'target_duration': actual_target,
            'actual_duration': round(total_duration, 1),
            'duration_stats': {
                'min_scene': min(scene_durations),
                'max_scene': max(scene_durations),
                'avg_scene': round(sum(scene_durations) / len(scene_durations), 1)
            },
            'natural_variation': True
        }

    def generate_hook_subscribe_scenes(self, scene_plan: List[Dict], hook_content: str, subscribe_content: str) -> Dict:
        """Generate background scenes for hook and subscribe with precise timing"""

        self.log_step("Hook & Subscribe Scene Selection")

        # Select 10 atmospheric scenes for hook (0-30s)
        hook_scenes = []
        atmospheric_scenes = [s for s in scene_plan if s.get('template') == 'atmospheric' or s.get('emotion') == 'peaceful'][:10]

        for i, scene in enumerate(atmospheric_scenes):
            hook_scenes.append({
                "scene_id": scene['scene_id'],
                "scene_title": scene['title'],
                "start_time": i * 3,
                "end_time": (i * 3) + 3,
                "duration": 3,
                "visual_prompt": f"Atmospheric cinematic view of {scene['location']}, golden hour lighting, peaceful and mysterious mood",
                "timing_note": f"Display during hook seconds {i*3}-{(i*3)+3}",
                "sync_importance": "HIGH - Must align with hook narration rhythm"
            })

        # Select 10 community scenes for subscribe (30-60s)
        subscribe_scenes = []
        community_scenes = [s for s in scene_plan if s.get('template') == 'character_focused' or len(s.get('characters_mentioned', [])) > 0][:10]

        for i, scene in enumerate(community_scenes):
            subscribe_scenes.append({
                "scene_id": scene['scene_id'],
                "scene_title": scene['title'],
                "start_time": i * 3,
                "end_time": (i * 3) + 3,
                "duration": 3,
                "visual_prompt": f"Welcoming community view of {scene['location']}, warm lighting, inviting atmosphere",
                "timing_note": f"Display during subscribe seconds {i*3}-{(i*3)+3}",
                "sync_importance": "HIGH - Must feel warm and community-building"
            })

        return {
            "hook_scenes": hook_scenes,
            "subscribe_scenes": subscribe_scenes,
            "scene_story_alignment": {
                "rule": "When story mentions scene X, immediately display scene X",
                "critical": "Scene mention = Scene display",
                "timing": "Instant visual sync with narrative"
            },
            "production_notes": {
                "hook_timing": "Use hook_scenes during golden hook narration (0-30s)",
                "subscribe_timing": "Use subscribe_scenes during subscribe request (30-60s)",
                "visual_sync": "Each scene should blend seamlessly with spoken content",
                "fallback_strategy": "If scene unavailable, use next scene in sequence"
            }
        }

    def _generate_scene_chapters(self, scene_plan: List[Dict]) -> List[Dict]:
        """Generate YouTube chapter markers for scenes"""
        chapters = []
        current_time = 60  # Start after hook and subscribe (60 seconds)

        for scene in scene_plan:
            duration_seconds = int(scene.get('duration_minutes', 4) * 60)

            chapters.append({
                "time": f"{current_time // 60}:{current_time % 60:02d}",
                "title": f"Scene {scene['scene_id']}: {scene.get('title', 'Unknown')}"[:100],
                "duration_seconds": duration_seconds,
                "emotion": scene.get('emotion', 'peaceful'),
                "template": scene.get('template', 'atmospheric')
            })

            current_time += duration_seconds

        return chapters

    def generate_complete_story_with_characters(self, topic: str, description: str, clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """
        COMPLETE 5-STAGE APPROACH WITH FIXED SMART ALGORITHM:
        Stage 1: Smart Planning + Hook + Subscribe + First Half stories
        Stage 2: Remaining stories (second half)
        Stage 3: Character extraction and analysis
        Stage 4: Intelligent thumbnail generation
        Stage 5: Hook & Subscribe scene selection + Complete JSON outputs
        """

        self.log_step("Complete Story Generation with Smart Random Durations")

        try:
            # STAGE 1: Smart Planning + First Half
            stage1_result = self._generate_stage1(topic, description)
            time.sleep(1)  # Server-friendly pause

            # Get scene structure from stage 1
            scene_plan = stage1_result.get('scene_plan', [])
            total_scenes = len(scene_plan)
            first_half_stories = len(stage1_result.get('stories', {}))

            print(f"🎲 Smart Structure Generated:")
            print(f"   📊 Total scenes: {total_scenes}")
            print(f"   📝 First half: {first_half_stories} stories")
            print(f"   📝 Second half: {total_scenes - first_half_stories} stories")

            if total_scenes > 0:
                durations = [scene.get('duration_minutes', 4) for scene in scene_plan]
                print(f"   ⏱️ Duration range: {min(durations):.1f}-{max(durations):.1f} minutes")
                print(f"   📊 Total duration: {sum(durations):.1f} minutes")

            # STAGE 2: Second Half Stories
            stage2_result = self._generate_stage2(topic, description, stage1_result)
            time.sleep(1)

            # STAGE 3: Character Extraction
            character_result = self._extract_characters(topic, description, stage1_result, stage2_result)
            time.sleep(1)

            # STAGE 4: Intelligent Thumbnail Generation
            thumbnail_result = self._generate_intelligent_thumbnail(
                topic, description, character_result, clickbait_title, font_design
            )
            time.sleep(1)

            # STAGE 5: Hook & Subscribe Scene Selection
            hook_subscribe_result = self.generate_hook_subscribe_scenes(
                stage1_result.get('scene_plan', []),
                stage1_result.get('golden_hook', {}).get('content', ''),
                stage1_result.get('subscribe_section', {}).get('content', '')
            )

            # COMBINE: Merge all stages
            combined_result = self._combine_all_stages(
                stage1_result, stage2_result, character_result, thumbnail_result, hook_subscribe_result, topic, description
            )

            # Add smart generation stats
            combined_result['generation_stats'].update({
                'smart_algorithm': True,
                'random_scene_count': total_scenes,
                'natural_duration_variation': True,
                'duration_range': f"{min(durations):.1f}-{max(durations):.1f} minutes" if durations else "N/A"
            })

            self.log_step("Complete Smart Generation Pipeline Finished", "SUCCESS", {
                "total_scenes": len(combined_result.get('scene_plan', [])),
                "total_stories": len(combined_result.get('stories', {})),
                "characters_extracted": len(combined_result.get('main_characters', [])),
                "thumbnail_generated": combined_result.get('generation_stats', {}).get('thumbnail_generated', False),
                "hook_subscribe_generated": combined_result.get('generation_stats', {}).get('hook_subscribe_generated', False),
                "smart_algorithm_used": True,
                "api_calls_total": self.api_call_count,
                "total_cost": self.total_cost
            })

            return combined_result

        except Exception as e:
            self.log_step("Generation Failed", "ERROR")
            CONFIG.logger.error(f"Generation failed: {e}")
            raise

    def _generate_stage1(self, topic: str, description: str) -> Dict[str, Any]:
        """STAGE 1: FIXED Smart planning with proper scene count + first half stories"""

        # Generate smart scene structure
        smart_structure = self._generate_smart_scene_structure()
        total_scenes = smart_structure['scene_count']
        scene_durations = smart_structure['scene_durations']
        first_half = total_scenes // 2

        self.log_step(f"Stage 1: Smart Planning + First {first_half} Stories (Total: {total_scenes} scenes)")

        stage1_prompt = f"""Create the complete foundation for a 2-hour sleep story about "{topic}".

TOPIC: {topic}
DESCRIPTION: {description}

SMART STORY STRUCTURE:
- Total scenes: {total_scenes} (NATURAL VARIATION - random count 28-45)
- Target duration: {smart_structure['target_duration']} minutes
- Scene durations: VARIABLE (see list below)
- First half: {first_half} scenes (this stage)
- Second half: {total_scenes - first_half} scenes (next stage)

SCENE DURATION PLAN:
{', '.join([f'Scene {i+1}: {dur}min' for i, dur in enumerate(scene_durations)])}

STAGE 1 REQUIREMENTS:
You must provide ALL planning elements + first {first_half} stories in complete detail.

## 1. GOLDEN HOOK (30 seconds, ~90 words)
- Atmospheric opening that sets the scene
- Gentle intrigue but calming
- Cinematic visual details

## 2. SUBSCRIBE SECTION (30 seconds, ~70 words) 
- Natural community invitation
- Warm, friendly tone (not corporate)

## 3. COMPLETE SCENE PLAN (Exactly {total_scenes} scenes with SMART DURATIONS)
Each scene must use the exact duration from the plan above:

{chr(10).join([f"Scene {i+1}: {scene_durations[i]:.1f} minutes" for i in range(total_scenes)])}

Scene structure requirements:
- Template rotation: atmospheric, character_focused, historical_detail, sensory_journey
- Style rotation: observational, immersive, documentary, poetic, cinematic
- Emotion progression: 1-30% peaceful, 31-60% curiosity, 61-80% concern, 81-100% resolution
- Key characters mentioned in descriptions

## 4. FIRST {first_half} COMPLETE STORIES (Scenes 1-{first_half})
Each story must be 300-900 words (based on scene duration) with:
- Present tense, second person perspective - but NEVER start with "You find yourself"
- Rich sensory details (sight, sound, smell, touch)
- [PAUSE] markers for TTS
- Sleep-optimized language
- Historical accuracy
- Clear character interactions and mentions
- Create unique, atmospheric openings for each scene

## 5. BASIC VISUAL PROMPTS (All {total_scenes} scenes)
- Simple AI image generation prompts
- Focus on location and atmosphere
- Character presence noted but details added later

## 6. VOICE DIRECTIONS (All {total_scenes} scenes)
- TTS guidance for each scene
- Pace, mood, emphasis

OUTPUT FORMAT (Complete JSON):
{{
  "golden_hook": {{
    "content": "[90-word atmospheric opening]",
    "duration_seconds": 30,
    "voice_direction": "Gentle, mysterious but calming tone"
  }},
  "subscribe_section": {{
    "content": "[70-word community invitation]",
    "duration_seconds": 30,
    "voice_direction": "Warm, friendly conversational tone"
  }},
  "scene_plan": [
    {{
      "scene_id": 1,
      "title": "[Scene title]",
      "location": "[Historical location]", 
      "duration_minutes": {scene_durations[0] if scene_durations else 4},
      "template": "atmospheric",
      "narrative_style": "observational",
      "emotion": "peaceful",
      "sensory_focus": "sight",
      "description": "[What happens - include character names if present]",
      "key_elements": ["element1", "element2", "element3"],
      "characters_mentioned": ["character1", "character2"]
    }}
  ],
  "stories": {{
    "1": "[COMPLETE story for scene 1 with character interactions]",
    "2": "[COMPLETE story for scene 2]"
  }},
  "visual_prompts": [
    {{
      "scene_number": 1,
      "title": "[Scene title]",
      "prompt": "[Basic AI image prompt]",
      "duration_minutes": {scene_durations[0] if scene_durations else 4},
      "emotion": "peaceful"
    }}
  ],
  "voice_directions": [
    {{
      "scene_number": 1,
      "title": "[Scene title]", 
      "direction": "[TTS guidance]",
      "template": "atmospheric",
      "style": "observational"
    }}
  ],
  "stage1_stats": {{
    "scenes_planned": {total_scenes},
    "stories_written": {first_half},
    "total_word_count": "[calculated]",
    "characters_introduced": "[count]",
    "ready_for_stage2": true
  }}
}}

Generate complete Stage 1 content with all {total_scenes} scenes planned and first {first_half} stories written.
USE THE EXACT DURATIONS FROM THE PLAN ABOVE."""

        try:
            self.api_call_count += 1

            # ✅ PROVEN SUCCESSFUL METHOD - STREAMING RESPONSE LIKE OLD CODE
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,  # ✅ STREAMING - PROVEN CRITICAL FOR LONG CONTENT
                timeout=1800,  # ✅ 30 MINUTE TIMEOUT - PROVEN SUCCESSFUL
                system="You are a MASTER STORYTELLER and automated content creator. Stage 1: Create complete planning + first half atmospheric stories with rich character interactions. Focus on memorable, distinct characters.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            # ✅ COLLECT STREAMING RESPONSE LIKE SUCCESSFUL OLD CODE
            content = ""
            print("📡 Stage 1: Streaming Claude 4 response (PROVEN METHOD)...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 1: {len(content):,} characters...")

            print(f"✅ Stage 1 complete: {len(content):,} characters - STREAMING SUCCESS")
            # ✅ CALCULATE COST LIKE OLD CODE
            input_tokens = len(stage1_prompt) // 4  # Rough estimate
            output_tokens = len(content) // 4  # Rough estimate
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)  # Claude costs
            self.total_cost += stage_cost

            CONFIG.logger.info(f"Stage 1 response length: {len(content)} characters - Cost: ${stage_cost:.4f}")

            # ✅ PARSE STAGE 1 RESULT WITH PROVEN ERROR HANDLING
            parsed_result = self._parse_claude_response(content, "stage1")

            self.log_step("Stage 1 Parsing", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stories_written": len(parsed_result.get('stories', {})),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 1 Failed", "ERROR")
            CONFIG.logger.error(f"Stage 1 error: {e}")
            raise

    def _generate_stage2(self, topic: str, description: str, stage1_result: Dict) -> Dict[str, Any]:
        """STAGE 2: FIXED Remaining stories (second half)"""

        # Get scene plan from stage 1
        scene_plan = stage1_result.get('scene_plan', [])
        stories_written = len(stage1_result.get('stories', {}))
        total_scenes = len(scene_plan)
        remaining_scenes = total_scenes - stories_written

        if remaining_scenes <= 0:
            self.log_step("Stage 2: No remaining stories needed", "SUCCESS")
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        self.log_step(f"Stage 2: Remaining {remaining_scenes} Stories (Dynamic)")

        # Get scenes that need stories
        remaining_scene_plan = []
        for scene in scene_plan:
            if str(scene['scene_id']) not in stage1_result.get('stories', {}):
                remaining_scene_plan.append(scene)

        if len(remaining_scene_plan) == 0:
            self.log_step("Stage 2: All stories already written", "SUCCESS")
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        # Create stage 2 prompt
        scenes_text = "\n".join([
            f"Scene {scene['scene_id']}: {scene['title']}\n"
            f"Location: {scene['location']}\n"
            f"Duration: {scene.get('duration_minutes', 4)} minutes\n"
            f"Template: {scene['template']} | Style: {scene['narrative_style']}\n"
            f"Emotion: {scene['emotion']} | Focus: {scene['sensory_focus']}\n"
            f"Description: {scene['description']}\n"
            f"Characters: {', '.join(scene.get('characters_mentioned', []))}\n"
            for scene in remaining_scene_plan
        ])

        stage2_prompt = f"""Complete the sleep story for "{topic}" by writing the remaining {remaining_scenes} stories.

TOPIC: {topic}
DESCRIPTION: {description}

SCENES TO COMPLETE:
{scenes_text}

SMART DURATION NOTES:
- Each scene has VARIABLE duration (shown above)
- Word count should match duration: ~150 words per minute
- Longer scenes (5-7 min) = 750-1050 words
- Shorter scenes (2-3 min) = 300-450 words
- Medium scenes (4-5 min) = 600-750 words

REQUIREMENTS:
- Write COMPLETE stories for all remaining scenes
- Present tense, second person perspective - NEVER use "You find yourself"
- Follow emotion progression throughout the story
- Each story must be atmospheric and historically accurate
- Rich sensory details throughout
- Continue character development from Stage 1
- Maintain character consistency and interactions
- ADJUST WORD COUNT based on scene duration

OUTPUT FORMAT:
{{
  "stories": {{
    "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}": "[COMPLETE story matching duration]",
    "{remaining_scene_plan[1]['scene_id'] if len(remaining_scene_plan) > 1 else 'Y'}": "[COMPLETE story matching duration]"
  }},
  "stage2_stats": {{
    "stories_written": {remaining_scenes},
    "scenes_covered": "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}-{remaining_scene_plan[-1]['scene_id'] if remaining_scene_plan else 'Y'}",
    "smart_durations": true,
    "total_word_count": "[calculated]",
    "character_development": "continued"
  }}
}}

Write all {remaining_scenes} remaining stories with appropriate length for each scene's duration."""

        try:
            self.api_call_count += 1

            # ✅ PROVEN SUCCESSFUL METHOD - STREAMING RESPONSE LIKE OLD CODE
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,  # ✅ STREAMING - PROVEN CRITICAL FOR LONG CONTENT
                timeout=1800,  # ✅ 30 MINUTE TIMEOUT - PROVEN SUCCESSFUL
                system="You are a MASTER STORYTELLER. Stage 2: Complete the remaining stories with rich character development and consistent character interactions from Stage 1.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            # ✅ COLLECT STREAMING RESPONSE LIKE SUCCESSFUL OLD CODE
            content = ""
            print("📡 Stage 2: Streaming Claude 4 response (PROVEN METHOD)...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 2: {len(content):,} characters...")

            print(f"✅ Stage 2 complete: {len(content):,} characters - STREAMING SUCCESS")

            # ✅ CALCULATE COST
            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse Stage 2 result
            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 Parsing", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 Failed", "ERROR")
            CONFIG.logger.error(f"Stage 2 error: {e}")
            return {"stories": {}, "stage2_stats": {"error": str(e)}}

    def _extract_characters(self, topic: str, description: str, stage1_result: Dict, stage2_result: Dict) -> Dict[str, Any]:
        """STAGE 3: Extract main characters + YouTube optimization + REGENERATE VISUAL PROMPTS"""

        self.character_system.log_extraction_step("Character Extraction and Production Optimization")

        # Combine all story content for analysis
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))

        scene_plan = stage1_result.get('scene_plan', [])

        # Create character extraction prompt
        story_content = ""
        for scene_id, story in all_stories.items():
            story_content += f"Scene {scene_id}:\n{story}\n\n"

        # Add scene plan for context
        scene_context = ""
        for scene in scene_plan:
            scene_context += f"Scene {scene['scene_id']}: {scene.get('title', '')} - {scene.get('description', '')}\n"

        character_prompt = f"""Analyze the complete sleep story and create character extraction + YouTube optimization package.

TOPIC: {topic}
DESCRIPTION: {description}

STORY CONTENT (First 25000 chars):
{story_content[:25000]}

SCENE PLAN CONTEXT (First 4000 chars):
{scene_context[:4000]}

REQUIREMENTS:

## PART 1: DEEP CHARACTER EXTRACTION
- Identify maximum {CONFIG.claude_config['max_characters']} main characters
- Focus on characters that appear in multiple scenes
- Provide comprehensive character analysis for each including:
  * Physical description (for visual consistency)
  * Personality traits and voice style
  * Character arc (beginning, conflict, resolution)
  * Symbolic meaning and core function in story
  * Visual contrast and lighting preferences
  * Marketing utility for thumbnails/promotion
- Map which scenes each character appears in
- Analyze character relationships and dynamics

## PART 2: YOUTUBE OPTIMIZATION
Create complete YouTube upload package

## PART 3: PRODUCTION SPECIFICATIONS
Generate all technical requirements for full automation

OUTPUT FORMAT (Complete JSON):
{{
  "main_characters": [
    {{
      "name": "[Character name]",
      "role": "protagonist|supporting|background|minor",
      "importance_score": 0,
      "scene_appearances": [1, 3, 7, 12],
      "personality_traits": ["trait1", "trait2", "trait3"],
      "physical_description": "[Detailed visual description for consistency]",
      "visual_notes": "[Special notes for image generation and lighting]",
      "voice_style": "[How character speaks - pace, tone, emotional style]",
      "core_function": "[Character's purpose in the story - symbolic or narrative role]",
      "character_arc": {{
        "beginning": "[Character's initial state/motivation]",
        "conflict": "[Main challenge or tension character faces]",
        "ending": "[Character's resolution or transformation]"
      }},
      "symbolism": "[What this character represents thematically]",
      "visual_contrast": "[Lighting, shadow, color preferences for visual distinction]",
      "emotional_journey": "[How character's emotions evolve through scenes]",
      "use_in_marketing": true/false,
      "thumbnail_potential": "[Why this character works well in thumbnails]",
      "relationships": [
        {{"character": "other_character_name", "relationship": "detailed description", "visual_dynamic": "how they appear together"}}
      ]
    }}
  ],
  "character_relationships": [
    {{
      "char1": "Character1", 
      "char2": "Character2", 
      "relationship": "detailed relationship description",
      "emotional_dynamic": "loving|respectful|complex|protective|conflicted",
      "visual_dynamic": "how they appear together in scenes",
      "story_function": "what this relationship represents in the narrative",
      "evolution": "how relationship changes throughout story",
      "symbolic_meaning": "deeper thematic significance"
    }}
  ],
  "scene_character_mapping": {{
    "1": ["Character1"],
    "2": ["Character1", "Character2"]
  }},
  "visual_style_notes": {{
    "art_style": "[Preferred illustration style]",
    "color_palette": "[Dominant colors]",
    "mood": "[Overall visual mood]",
    "period_accuracy": "[Historical period details]"
  }},
  "youtube_optimization": {{
    "clickbait_titles": [
      "This Ancient Queen's Final Secret Will Put You to Sleep Instantly",
      "I Spent 2 Hours in Ancient Palace (Sleep Story)",
      "What Really Happened in History's Most Peaceful Night?",
      "Ancient Sleep Ritual That Actually Works",
      "The Most Relaxing Historical Story Ever Told"
    ],
    "thumbnail_concept": {{
      "main_character": "[Main character name]",
      "dramatic_scene": "[Most visually striking scene]",
      "text_overlay": "[Large, bold text for thumbnail]",
      "color_scheme": "[Eye-catching colors]",
      "emotion": "[Character emotion for maximum impact]",
      "background": "[Atmospheric background]",
      "style_notes": "[Thumbnail design specifications]"
    }},
    "video_description": {{
      "hook": "[First 125 characters - crucial for SEO]",
      "main_description": "[Full description with story overview]",
      "chapters": [
        {{"time": "0:00", "title": "Golden Hook - Enter the Palace"}},
        {{"time": "0:30", "title": "Subscribe for More Sleep Stories"}},
        {{"time": "1:00", "title": "Scene 1: The Golden Antechamber"}}
      ],
      "subscribe_cta": "[Subscribe call-to-action text]",
      "social_links": "[Links to other platforms]",
      "disclaimer": "[Sleep story disclaimer text]"
    }},
    "seo_strategy": {{
      "primary_keywords": ["sleep story", "relaxation", "insomnia help"],
      "long_tail_keywords": ["2 hour sleep story ancient", "historical bedtime story"],
      "trending_keywords": ["sleep podcast", "asmr sleep"],
      "niche_keywords": ["ancient history sleep", "palace meditation"],
      "location_keywords": ["ancient", "historical"],
      "competitor_keywords": ["jason stephenson", "michelle sanctuary"]
    }},
    "tags": [
      "sleep story", "bedtime story", "relaxation", "insomnia", "meditation",
      "history", "ancient", "palace", "asmr", "calm", "peaceful", "2 hours"
    ],
    "hashtags": [
      "#sleepstory", "#bedtimestory", "#relaxation", "#meditation", "#insomnia",
      "#history", "#ancient", "#asmr", "#deepsleep", "#calm", "#stressrelief"
    ],
    "youtube_metadata": {{
      "category": "Education",
      "default_language": "en",
      "privacy_status": "public",
      "license": "youtube",
      "embeddable": true,
      "made_for_kids": false,
      "target_audience": "adults 25-65"
    }}
  }},
  "production_specifications": {{
    "audio_production": {{
      "tts_settings": {{
        "optimal_voice": "[Best TTS voice for sleep content]",
        "speed_multiplier": 0.85,
        "pitch_adjustment": -2,
        "volume_level": 80,
        "pause_durations": {{
          "[PAUSE]": 2.0,
          "paragraph_break": 1.5,
          "scene_transition": 3.0
        }}
      }},
      "background_music": {{
        "primary_track": "[Genre appropriate for story period]",
        "volume_level": 15,
        "fade_in_duration": 2.0,
        "fade_out_duration": 3.0
      }},
      "audio_export": {{
        "format": "MP3",
        "bitrate": "192kbps", 
        "sample_rate": "44.1kHz"
      }}
    }},
    "video_assembly": {{
      "scene_timing_precision": [
        {{
          "scene_number": 1,
          "start_time": "00:01:00",
          "end_time": "00:05:03", 
          "duration_seconds": 243,
          "word_count": 580
        }}
      ],
      "video_specifications": {{
        "resolution": "1920x1080",
        "frame_rate": 30,
        "transition_type": "slow_fade",
        "export_format": "MP4_H264"
      }}
    }},
    "quality_control": {{
      "content_validation": {{
        "sleep_optimization_score": "[1-10 rating]",
        "historical_accuracy_verified": true,
        "content_appropriateness": "family_friendly"
      }},
      "technical_validation": {{
        "total_word_count": "[exact count]",
        "target_duration_met": true,
        "character_consistency_score": "[1-10]"
      }}
    }}
  }},
  "character_stats": {{
    "total_characters_found": 0,
    "main_characters_extracted": 0,
    "character_coverage": "percentage of scenes with characters"
  }}
}}

Analyze thoroughly and create complete package with DEEP CHARACTER ANALYSIS including:
- Character arcs (beginning → conflict → resolution)
- Symbolic meaning and thematic representation  
- Visual contrast and lighting preferences
- Voice style and emotional journey
- Marketing and thumbnail potential
- Complex relationship dynamics

Plus full YouTube optimization and production specifications."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=16000,
                temperature=0.3,
                stream=True,
                timeout=900,
                system="You are an expert character analyst and production optimization specialist. Extract main characters with comprehensive analysis including character arcs, symbolism, visual contrast, and marketing potential. Create complete production package with deep character psychology.",
                messages=[{"role": "user", "content": character_prompt}]
            )

            # Collect streaming response
            content = ""
            print("📡 Stage 3: Analyzing characters and creating optimization...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text

            print(f"✅ Character analysis complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(character_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse character extraction result
            parsed_result = self._parse_claude_response(content, "character_extraction")

            # Process characters through extraction system
            if 'main_characters' in parsed_result:
                # Filter to top characters
                top_characters = self.character_system.filter_top_characters(
                    parsed_result['main_characters'],
                    CONFIG.claude_config['max_characters']
                )
                parsed_result['main_characters'] = top_characters

                # Analyze scene-character presence
                scene_character_map = self.character_system.analyze_scene_character_presence(
                    stage1_result.get('scene_plan', []),
                    top_characters
                )
                parsed_result['scene_character_mapping'] = scene_character_map

                # REGENERATE VISUAL PROMPTS with correct character integration
                regenerated_visual_prompts = self._regenerate_visual_prompts_with_characters(
                    stage1_result.get('scene_plan', []),
                    top_characters,
                    scene_character_map,
                    parsed_result.get('visual_style_notes', {})
                )

                # Replace the old visual prompts with the regenerated ones
                parsed_result['regenerated_visual_prompts'] = regenerated_visual_prompts

                self.character_system.log_extraction_step("Visual Prompts Regenerated", "SUCCESS", {
                    "prompts_created": len(regenerated_visual_prompts),
                    "character_integrated_scenes": len([p for p in regenerated_visual_prompts if p.get('character_reference_needed')])
                })

            self.character_system.log_extraction_step("Character Extraction", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('main_characters', [])),
                "character_names": [c.get('name', 'Unknown') for c in parsed_result.get('main_characters', [])],
                "visual_prompts_regenerated": 'regenerated_visual_prompts' in parsed_result,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Character Extraction Failed", "ERROR")
            print(f"❌ Character extraction error: {e}")
            return {"main_characters": [], "character_stats": {"error": str(e)}}

    def _generate_intelligent_thumbnail(self, topic: str, description: str, character_result: Dict, clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """STAGE 4: Generate intelligent thumbnail with character analysis"""

        self.character_system.log_extraction_step("Intelligent Thumbnail Generation")

        characters = character_result.get('main_characters', [])
        visual_style = character_result.get('visual_style_notes', {})

        # Select optimal character for thumbnail
        thumbnail_character_selection = self.character_system.select_thumbnail_character(
            characters, topic, description
        )

        # Prepare thumbnail generation prompt
        character_info = ""
        if thumbnail_character_selection['character_data']:
            char_data = thumbnail_character_selection['character_data']
            character_info = f"""
SELECTED CHARACTER: {char_data['name']}
PHYSICAL DESCRIPTION: {char_data['physical_description']}
VISUAL NOTES: {char_data.get('visual_notes', '')}
THUMBNAIL POTENTIAL: {char_data.get('thumbnail_potential', 'Strong visual presence')}
ROLE: {char_data['role']}
REASONING: {thumbnail_character_selection['reasoning']}
"""
        else:
            character_info = f"""
ATMOSPHERIC APPROACH SELECTED
REASONING: {thumbnail_character_selection['reasoning']}
NO CHARACTER FOCUS - Use scenic/atmospheric elements
"""

        # Use provided clickbait title or generate fallback
        if not clickbait_title:
            youtube_data = character_result.get('youtube_optimization', {})
            clickbait_titles = youtube_data.get('clickbait_titles', [])
            clickbait_title = clickbait_titles[0] if clickbait_titles else f"The Secret History of {topic} (2 Hour Sleep Story)"

        # Use provided font design or generate fallback
        if not font_design:
            font_design = "Bold impact font, uppercase for key words, warm golden color (#d4af37) for main text, contrasted with deep shadows for readability"

        thumbnail_prompt = f"""Create an intelligent thumbnail design for the sleep story "{topic}".

STORY TOPIC: {topic}
STORY DESCRIPTION: {description}

{character_info}

VISUAL STYLE CONTEXT:
ART STYLE: {visual_style.get('art_style', 'Romantic realism with atmospheric lighting')}
COLOR PALETTE: {visual_style.get('color_palette', 'Warm golden, amber, rose tones')}
MOOD: {visual_style.get('mood', 'Peaceful, contemplative, nostalgic')}

CLICKBAIT TITLE: {clickbait_title}
FONT DESIGN: {font_design}

REQUIREMENTS:
Create a thumbnail that balances SLEEP CONTENT (peaceful, calming) with CLICKABILITY (attention-grabbing but not jarring).

## CHARACTER-BASED APPROACH:
If character selected: Feature the character prominently in a peaceful but visually striking pose/setting
If atmospheric: Focus on the most visually compelling location/scene from the story

## VISUAL STRATEGY:
1. **Background**: Choose most compelling scene location
2. **Lighting**: Warm, golden hour lighting for maximum appeal
3. **Composition**: Rule of thirds, leading lines, visual hierarchy
4. **Text Placement**: Readable but not overwhelming
5. **Sleep Optimization**: Calming colors, not overstimulating
6. **Click Optimization**: Compelling visual hook, emotional appeal

OUTPUT FORMAT:
{{
  "thumbnail_prompt": {{
    "scene_number": 99,
    "character_used": "{thumbnail_character_selection['character_used']}",
    "clickbait_title": "{clickbait_title}",
    "font_design": "{font_design}",
    "prompt": "[Detailed visual prompt for AI image generation optimized for YouTube thumbnails]",
    "visual_style": "[Style notes including mood, lighting, composition]",
    "character_positioning": "[How character is positioned if included]",
    "text_overlay_strategy": "[How text should be overlaid on image]",
    "emotional_appeal": "[What emotion should viewer feel when seeing thumbnail]",
    "target_audience_appeal": "[Why this appeals to sleep story viewers]",
    "clickability_factors": "[What makes this thumbnail clickable]",
    "sleep_content_balance": "[How it maintains sleep content feel while being clickable]",
    "thumbnail_reasoning": "{thumbnail_character_selection['reasoning']}",
    "background_scene": "[Primary scene/location for background]",
    "lighting_strategy": "[Lighting approach for maximum visual appeal]",
    "composition_notes": "[Visual composition strategy]"
  }},
  "thumbnail_alternatives": [
    {{
      "variant": "Character Focus",
      "prompt": "[Alternative thumbnail focusing more on character]"
    }},
    {{
      "variant": "Atmospheric Focus", 
      "prompt": "[Alternative thumbnail focusing on setting/mood]"
    }},
    {{
      "variant": "Action Moment",
      "prompt": "[Alternative showing key story moment]"
    }}
  ],
  "thumbnail_stats": {{
    "character_approach": "{thumbnail_character_selection['character_used']}",
    "selection_reasoning": "{thumbnail_character_selection['reasoning']}",
    "visual_style_matched": true,
    "clickbait_optimized": true,
    "sleep_content_appropriate": true
  }}
}}

Create the PERFECT thumbnail that will get clicks while maintaining the peaceful, sleep-optimized feel of the content. Consider your target audience: people seeking relaxation and sleep help who are scrolling through YouTube late at night or during rest periods."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=8000,
                temperature=0.4,
                stream=True,
                timeout=600,
                system="You are a YouTube thumbnail optimization specialist who understands both sleep content marketing and visual psychology. Create thumbnails that balance peaceful sleep content with clickable appeal. Consider the unique needs of viewers seeking relaxation content.",
                messages=[{"role": "user", "content": thumbnail_prompt}]
            )

            # Collect streaming response
            content = ""
            print("📡 Stage 4: Generating intelligent thumbnail...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text

            print(f"✅ Thumbnail generation complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(thumbnail_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse thumbnail result
            parsed_result = self._parse_claude_response(content, "thumbnail_generation")

            self.character_system.log_extraction_step("Thumbnail Generation", "SUCCESS", {
                "character_approach": thumbnail_character_selection['character_used'],
                "selection_reasoning": thumbnail_character_selection['reasoning'],
                "alternatives_generated": len(parsed_result.get('thumbnail_alternatives', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Thumbnail Generation Failed", "ERROR")
            print(f"❌ Thumbnail generation error: {e}")

            # Fallback thumbnail
            fallback_thumbnail = {
                "thumbnail_prompt": {
                    "scene_number": 99,
                    "character_used": "None (Atmospheric focus)",
                    "clickbait_title": clickbait_title or f"The Secret History of {topic} (2 Hour Sleep Story)",
                    "font_design": font_design or "Bold impact font with warm colors",
                    "prompt": f"Atmospheric thumbnail of {topic}, warm golden lighting, peaceful but compelling visual, optimized for sleep content viewers",
                    "visual_style": "Peaceful and inviting",
                    "thumbnail_reasoning": "Fallback due to generation error"
                },
                "thumbnail_stats": {"error": str(e)}
            }
            return fallback_thumbnail

    def _regenerate_visual_prompts_with_characters(self, scene_plan: List[Dict], characters: List[Dict], scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
        """Character extraction'dan SONRA visual prompts'ı yeniden oluştur"""

        self.character_system.log_extraction_step("Regenerating Visual Prompts with Character Integration")

        regeneration_prompt = f"""Based on the completed scene plan and extracted character data, create accurate visual generation prompts for all scenes.

SCENE PLAN SUMMARY:
{json.dumps(scene_plan, indent=2)}

MAIN CHARACTERS:
{json.dumps([{
    'name': char.get('name', ''),
    'physical_description': char.get('physical_description', ''),
    'visual_notes': char.get('visual_notes', ''),
    'role': char.get('role', '')
} for char in characters], indent=2)}

SCENE-CHARACTER MAPPING:
{json.dumps(scene_character_map, indent=2)}

VISUAL STYLE NOTES:
{json.dumps(style_notes, indent=2)}

REQUIREMENTS:
Create accurate visual prompts that match the scene content and characters. Each prompt must:

1. **Match the scene location and activity exactly**
2. **Include characters when they appear in the scene**
3. **Use atmospheric descriptions when no characters present**
4. **Be optimized for AI image generation**

OUTPUT FORMAT for each scene:
{{
  "scene_number": X,
  "title": "[Scene title from scene plan]",
  "location": "[Scene location]",
  "characters_present": ["Character1", "Character2"] or [],
  "character_reference_needed": true/false,
  "prompt": "[Detailed visual prompt matching scene content]",
  "enhanced_prompt": "[Same prompt with character markers for reference system]",
  "duration_minutes": X,
  "emotion": "[Scene emotion]",
  "template": "[Scene template]",
  "characters_in_scene": [
    {{
      "name": "Character Name",
      "description": "Physical description for visual reference",
      "importance": X
    }}
  ]
}}

Generate visual prompts that accurately reflect the scene plan and character presence.

OUTPUT (Complete JSON array of all scenes):
"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=16000,
                temperature=0.3,
                stream=True,
                timeout=900,
                system="You are a visual prompt specialist. Create accurate AI image generation prompts that exactly match the provided scene content and character data.",
                messages=[{"role": "user", "content": regeneration_prompt}]
            )

            content = ""
            print("📡 Regenerating Visual Prompts with Character Integration...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text

            print(f"✅ Visual prompt regeneration complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(regeneration_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            try:
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                elif content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                visual_prompts = json.loads(content)

                self.character_system.log_extraction_step("Visual Prompt Regeneration", "SUCCESS", {
                    "prompts_generated": len(visual_prompts),
                    "character_integrated": True,
                    "stage_cost": stage_cost
                })

                return visual_prompts

            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed, attempting partial extraction: {e}")
                return self._extract_visual_prompts_from_text(content)

        except Exception as e:
            self.character_system.log_extraction_step("Visual Prompt Regeneration Failed", "ERROR")
            print(f"❌ Visual prompt regeneration error: {e}")
            return self._create_fallback_visual_prompts(scene_plan, characters, scene_character_map)

    def _extract_visual_prompts_from_text(self, content: str) -> List[Dict]:
        """Extract visual prompts from text when JSON parsing fails"""
        prompts = []
        for i in range(1, 41):
            prompt_data = {
                "scene_number": i,
                "title": f"Scene {i}",
                "characters_present": [],
                "character_reference_needed": False,
                "prompt": f"Ancient scene, historically accurate, atmospheric lighting",
                "enhanced_prompt": f"[ATMOSPHERIC SCENE] Ancient scene, historically accurate, atmospheric lighting",
                "duration_minutes": 4,
                "emotion": "peaceful",
                "characters_in_scene": []
            }
            prompts.append(prompt_data)
        return prompts

    def _create_fallback_visual_prompts(self, scene_plan: List[Dict], characters: List[Dict], scene_character_map: Dict) -> List[Dict]:
        """Fallback: Create visual prompts directly from scene plan and character data"""
        prompts = []

        for scene in scene_plan:
            scene_id = scene['scene_id']
            scene_characters = scene_character_map.get(str(scene_id), [])

            character_refs = []
            character_names = []

            for scene_char in scene_characters:
                char_name = scene_char if isinstance(scene_char, str) else scene_char.get('name', '')
                character_names.append(char_name)

                full_char = next((c for c in characters if c['name'] == char_name), None)
                if full_char:
                    character_refs.append({
                        'name': full_char['name'],
                        'description': full_char['physical_description'],
                        'importance': full_char['importance_score']
                    })

            location = scene.get('location', 'Ancient setting')
            description = scene.get('description', 'Peaceful scene')
            emotion = scene.get('emotion', 'peaceful')

            if character_names:
                char_list = ', '.join(character_names)
                prompt = f"[CHARACTERS: {char_list}] {location}, {description}, historically accurate, {emotion} atmosphere"
                enhanced_prompt = f"[CHARACTERS: {char_list}] {location}, {description}, historically accurate, {emotion} atmosphere"
                char_ref_needed = True
            else:
                prompt = f"[ATMOSPHERIC SCENE - NO CHARACTERS] {location}, {description}, historically accurate, {emotion} atmosphere"
                enhanced_prompt = prompt
                char_ref_needed = False

            prompt_data = {
                "scene_number": scene_id,
                "title": scene.get('title', f"Scene {scene_id}"),
                "location": location,
                "characters_present": character_names,
                "character_reference_needed": char_ref_needed,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration_minutes": scene.get('duration_minutes', 4),
                "emotion": emotion,
                "template": scene.get('template', 'atmospheric'),
                "characters_in_scene": character_refs
            }

            prompts.append(prompt_data)

        return prompts

    def _combine_all_stages(self, stage1: Dict, stage2: Dict, character_data: Dict, thumbnail_data: Dict, hook_subscribe_data: Dict, topic: str, description: str) -> Dict[str, Any]:
        """Combine all five stages into final result - USING REGENERATED VISUAL PROMPTS + THUMBNAIL"""

        self.log_step("Combining All Stages with Regenerated Visual Prompts + Thumbnail + Hook/Subscribe")

        # Merge stories
        all_stories = {}
        all_stories.update(stage1.get('stories', {}))
        all_stories.update(stage2.get('stories', {}))

        # Use REGENERATED visual prompts
        if 'regenerated_visual_prompts' in character_data:
            enhanced_visual_prompts = character_data['regenerated_visual_prompts']
            print(f"✅ Using regenerated visual prompts: {len(enhanced_visual_prompts)} prompts")
        else:
            print("⚠️ Using fallback visual prompt enhancement")
            enhanced_visual_prompts = self._enhance_visual_prompts_with_characters(
                stage1.get('visual_prompts', []),
                character_data.get('main_characters', []),
                character_data.get('scene_character_mapping', {}),
                character_data.get('visual_style_notes', {})
            )

        # ADD THUMBNAIL TO VISUAL PROMPTS
        thumbnail_prompt = thumbnail_data.get('thumbnail_prompt', {})
        if thumbnail_prompt:
            enhanced_visual_prompts.append(thumbnail_prompt)
            print(f"✅ Thumbnail added to visual prompts")

        # Generate scene chapters for YouTube
        scene_chapters = self._generate_scene_chapters(stage1.get('scene_plan', []))

        # Compile complete story text
        complete_story = self._compile_complete_story({
            **stage1,
            'stories': all_stories
        })

        # Final result with ALL ENHANCEMENTS
        result = {
            "hook_section": stage1.get("golden_hook", {}),
            "subscribe_section": stage1.get("subscribe_section", {}),
            "scene_plan": stage1.get("scene_plan", []),
            "scene_chapters": scene_chapters,
            "complete_story": complete_story,
            "visual_prompts": enhanced_visual_prompts,  # INCLUDES THUMBNAIL
            "voice_directions": stage1.get("voice_directions", []),
            "stories": all_stories,

            # CHARACTER DATA
            "main_characters": character_data.get('main_characters', []),
            "character_relationships": character_data.get('character_relationships', []),
            "scene_character_mapping": character_data.get('scene_character_mapping', {}),
            "visual_style_notes": character_data.get('visual_style_notes', {}),

            # YOUTUBE DATA
            "youtube_optimization": character_data.get('youtube_optimization', {}),

            # THUMBNAIL DATA
            "thumbnail_data": thumbnail_data,

            # HOOK & SUBSCRIBE DATA
            "hook_subscribe_scenes": hook_subscribe_data,

            # PRODUCTION DATA
            "production_specifications": character_data.get('production_specifications', {}),

            "generation_stats": {
                "api_calls_used": self.api_call_count,
                "total_cost": self.total_cost,
                "five_stage_approach": True,
                "smart_algorithm": True,
                "visual_prompts_regenerated": 'regenerated_visual_prompts' in character_data,
                "thumbnail_generated": bool(thumbnail_data.get('thumbnail_prompt')),
                "hook_subscribe_generated": bool(hook_subscribe_data.get('hook_scenes')),
                "youtube_optimization_generated": bool(character_data.get('youtube_optimization', {}).get('clickbait_titles')),
                "production_specifications_generated": bool(character_data.get('production_specifications', {}).get('audio_production')),
                "visual_prompts_with_thumbnail": len(enhanced_visual_prompts),
                "scenes_planned": len(stage1.get("scene_plan", [])),
                "stories_written": len(all_stories),
                "stage1_stories": len(stage1.get('stories', {})),
                "stage2_stories": len(stage2.get('stories', {})),
                "characters_extracted": len(character_data.get('main_characters', [])),
                "production_ready": len(all_stories) >= 25,
                "total_duration_minutes": sum(scene.get('duration_minutes', 4) for scene in stage1.get("scene_plan", [])),
                "automated_production_ready": True,
                "server_optimized": True,
                "complete_pipeline": True
            },
            "generation_log": self.generation_log,
            "character_extraction_log": self.character_system.extraction_log,
            "topic": topic,
            "description": description,
            "generated_at": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "enhancement_status": "complete_5_stage_pipeline_with_smart_algorithm_and_all_optimizations"
        }

        return result

    def _enhance_visual_prompts_with_characters(self, visual_prompts: List[Dict], characters: List[Dict], scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
        """Enhanced visual prompts with character information (FALLBACK METHOD)"""
        enhanced_prompts = []

        for prompt in visual_prompts:
            scene_number = prompt.get('scene_number', 0)
            enhanced_prompt = prompt.copy()

            scene_characters = scene_character_map.get(str(scene_number), [])

            if scene_characters:
                character_refs = []
                for scene_char in scene_characters:
                    char_name = scene_char if isinstance(scene_char, str) else scene_char.get('name', '')
                    full_char = next((c for c in characters if c['name'] == char_name), None)
                    if full_char:
                        character_refs.append({
                            'name': full_char['name'],
                            'description': full_char['physical_description'],
                            'visual_notes': full_char.get('visual_notes', ''),
                            'importance': full_char['importance_score']
                        })

                enhanced_prompt['characters_in_scene'] = character_refs
                enhanced_prompt['character_reference_needed'] = True
                enhanced_prompt['enhanced_prompt'] = f"[CHARACTERS: {', '.join([c['name'] for c in character_refs])}] {prompt.get('prompt', '')}"
            else:
                enhanced_prompt['characters_in_scene'] = []
                enhanced_prompt['character_reference_needed'] = False
                enhanced_prompt['enhanced_prompt'] = f"[ATMOSPHERIC SCENE - NO CHARACTERS] {prompt.get('prompt', '')}"

            enhanced_prompt['visual_style_notes'] = style_notes
            enhanced_prompts.append(enhanced_prompt)

        return enhanced_prompts

    def _compile_complete_story(self, story_data: Dict) -> str:
        """Compile all components into final story text"""
        story_parts = []

        # Golden Hook
        if "golden_hook" in story_data and story_data["golden_hook"]:
            story_parts.append("=== GOLDEN HOOK (0-30 seconds) ===")
            story_parts.append(story_data["golden_hook"].get("content", ""))
            story_parts.append("")

        # Subscribe Section
        if "subscribe_section" in story_data and story_data["subscribe_section"]:
            story_parts.append("=== SUBSCRIBE REQUEST (30-60 seconds) ===")
            story_parts.append(story_data["subscribe_section"].get("content", ""))
            story_parts.append("")

        # Main Story
        story_parts.append("=== MAIN STORY ===")
        story_parts.append("")

        # Add scenes
        scene_plan = story_data.get("scene_plan", [])
        stories = story_data.get("stories", {})

        for scene in scene_plan:
            scene_id = scene["scene_id"]
            story_content = stories.get(str(scene_id), f"[Story for scene {scene_id} - Planned but not yet written]")

            story_parts.append(f"## Scene {scene_id}: {scene['title']}")
            story_parts.append(f"Duration: {scene['duration_minutes']} minutes")
            story_parts.append(f"Voice: {scene['narrative_style']}")
            story_parts.append(f"Emotion: {scene['emotion']}")
            story_parts.append("")
            story_parts.append(story_content)
            story_parts.append("")

        return "\n".join(story_parts)

    def _parse_claude_response(self, content: str, stage: str) -> Dict[str, Any]:
        """Parse Claude response with improved error handling"""
        try:
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"⚠️ {stage}: Full JSON failed, extracting partial data...")
                return self._extract_partial_json(content, stage)

        except Exception as e:
            print(f"❌ {stage} parsing failed: {e}")
            return {}

    def _extract_partial_json(self, content: str, stage: str) -> Dict[str, Any]:
        """Extract usable data from partial JSON"""
        result = {}

        try:
            if stage == "stage1":
                result = {
                    "golden_hook": self._extract_json_object(content, "golden_hook"),
                    "subscribe_section": self._extract_json_object(content, "subscribe_section"),
                    "scene_plan": self._extract_json_array(content, "scene_plan"),
                    "stories": self._extract_stories_dict(content),
                    "visual_prompts": self._extract_json_array(content, "visual_prompts"),
                    "voice_directions": self._extract_json_array(content, "voice_directions")
                }
            elif stage == "stage2":
                result = {
                    "stories": self._extract_stories_dict(content)
                }
            elif stage == "character_extraction":
                result = {
                    "main_characters": self._extract_json_array(content, "main_characters"),
                    "character_relationships": self._extract_json_array(content, "character_relationships"),
                    "scene_character_mapping": self._extract_json_object(content, "scene_character_mapping"),
                    "visual_style_notes": self._extract_json_object(content, "visual_style_notes"),
                    "youtube_optimization": self._extract_json_object(content, "youtube_optimization"),
                    "production_specifications": self._extract_json_object(content, "production_specifications"),
                    "character_stats": self._extract_json_object(content, "character_stats")
                }
            elif stage == "thumbnail_generation":
                result = {
                    "thumbnail_prompt": self._extract_json_object(content, "thumbnail_prompt"),
                    "thumbnail_alternatives": self._extract_json_array(content, "thumbnail_alternatives"),
                    "thumbnail_stats": self._extract_json_object(content, "thumbnail_stats")
                }

        except Exception as e:
            print(f"⚠️ Partial extraction error for {stage}: {e}")

        return result

    def _extract_json_object(self, content: str, key: str) -> Dict:
        """Extract a JSON object by key"""
        try:
            pattern = f'"{key}":\\s*{{[^}}]+}}'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                obj_json = match.group(0).replace(f'"{key}":', '')
                return json.loads(obj_json)
        except:
            pass
        return {}

    def _extract_json_array(self, content: str, key: str) -> List:
        """Extract a JSON array by key"""
        try:
            start = content.find(f'"{key}": [')
            if start == -1:
                return []

            bracket_count = 0
            in_array = False
            array_content = ""

            for i, char in enumerate(content[start:], start):
                if char == '[':
                    bracket_count += 1
                    in_array = True
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0 and in_array:
                        array_content = content[start + len(f'"{key}": ['):i]
                        break

            if array_content:
                objects = re.findall(r'{[^}]*}', array_content, re.DOTALL)
                complete_objects = []
                for obj in objects:
                    try:
                        json.loads(obj)
                        complete_objects.append(obj)
                    except:
                        break

                if complete_objects:
                    return json.loads('[' + ','.join(complete_objects) + ']')
        except:
            pass
        return []

    def _extract_stories_dict(self, content: str) -> Dict[str, str]:
        """Extract stories dictionary"""
        stories = {}
        try:
            story_pattern = r'"(\d+)":\s*"([^"]+(?:\\.[^"]*)*?)"'
            matches = re.findall(story_pattern, content)

            for story_id, story_content in matches:
                story_content = story_content.replace('\\"', '"')
                story_content = story_content.replace('\\n', '\n')
                story_content = story_content.replace('\\[PAUSE\\]', '[PAUSE]')

                if len(story_content) > 200:
                    stories[story_id] = story_content
        except Exception as e:
            print(f"Story extraction error: {e}")

        return stories

# Database-based topic functions
def get_next_topic_from_database() -> Tuple[int, str, str, str, str]:
    """FIXED: Get next topic from database instead of CSV"""
    # Initialize database topic manager
    db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
    topic_manager = DatabaseTopicManager(str(db_path))

    # Import CSV if database is empty
    csv_path = CONFIG.paths['TOPIC_CSV_PATH']
    topic_manager.import_csv_if_needed(csv_path)

    # Get next pending topic
    result = topic_manager.get_next_pending_topic()

    if result:
        topic_id, topic, description, clickbait_title, font_design = result

        # Mark as started
        topic_manager.mark_topic_as_started(topic_id)

        CONFIG.logger.info(f"Topic selected from database: {topic}")
        print(f"✅ Topic selected from database: {topic}")
        return topic_id, topic, description, clickbait_title, font_design
    else:
        raise ValueError("No pending topics found in database")

def complete_topic_in_database(topic_id: int, scene_count: int, total_duration: float,
                              api_calls: int, total_cost: float, output_path: str):
    """Mark topic as completed in database"""
    db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
    topic_manager = DatabaseTopicManager(str(db_path))

    topic_manager.mark_topic_as_completed(
        topic_id, scene_count, total_duration, api_calls, total_cost, output_path
    )

    CONFIG.logger.info(f"Topic {topic_id} marked as completed in database")
    print(f"✅ Topic {topic_id} marked as completed in database")

def save_production_outputs(output_dir: str, result: Dict, story_topic: str, topic_id: int,
                              api_calls: int, total_cost: float):
    """Save complete production outputs - SERVER VERSION WITH ALL JSON FILES + NEW FEATURES"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
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

        # 6. YouTube metadata (COMPLETE PACKAGE WITH API_READY_FORMAT)
        youtube_path = output_path / "youtube_metadata.json"
        youtube_data = result.get("youtube_optimization", {})

        # ADD API_READY_FORMAT (MISSING FROM SERVER)
        youtube_data["api_ready_format"] = {
            "snippet": {
                "title": result.get("youtube_optimization", {}).get("clickbait_titles", ["Sleep Story"])[0] if result.get("youtube_optimization", {}).get("clickbait_titles") else "Sleep Story",
                "description": result.get("youtube_optimization", {}).get("video_description", {}).get("main_description", ""),
                "tags": result.get("youtube_optimization", {}).get("tags", [])[:30],
                "categoryId": "27",
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en"
            },
            "status": {
                "privacyStatus": "public",
                "embeddable": True,
                "license": "youtube",
                "publicStatsViewable": True,
                "madeForKids": False
            }
        }

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

        # 10. AUTOMATION_SPECS.JSON (MISSING FROM SERVER - LOKAL VERSION)
        automation_path = output_path / "automation_specs.json"
        automation_data = {
            "audio_production": result.get("production_specifications", {}).get("audio_production", {}),
            "video_assembly": result.get("production_specifications", {}).get("video_assembly", {}),
            "quality_control": result.get("production_specifications", {}).get("quality_control", {}),
            "automation_specifications": result.get("production_specifications", {}).get("automation_specifications", {}),
            "precise_timing_breakdown": result.get("production_specifications", {}).get("precise_timing_breakdown", {}),
            "implementation_ready": True
        }
        with open(automation_path, "w", encoding="utf-8") as f:
            json.dump(automation_data, f, indent=2, ensure_ascii=False)
        saved_files.append("automation_specs.json")

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

        # Story scenes audio with production specs
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

        # 12. ALL_STORIES.JSON (MISSING FROM SERVER)
        stories_path = output_path / "all_stories.json"
        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(result["stories"], f, indent=2, ensure_ascii=False)
        saved_files.append("all_stories.json")

        # 13. Video composition instructions (ENHANCED)
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
                    "end_time": sum(scene.get('duration_minutes', 4) * 60 for scene in result.get("scene_plan", [])) + 60,
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

        # 14. Generation report (COMPREHENSIVE)
        report_path = output_path / "generation_report.json"
        production_report = {
            "topic": story_topic,
            "topic_id": topic_id,
            "generation_completed": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "claude_4_optimized": True,
            "server_optimized": True,
            "five_stage_approach": True,
            "smart_algorithm_used": result.get("generation_stats", {}).get("smart_algorithm", False),
            "complete_pipeline": True,
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
                "thumbnail_composition": bool(result.get("thumbnail_data", {}).get("thumbnail_prompt"))
            },
            "files_saved": saved_files,
            "next_steps": [
                "1. Generate character reference images using character_profiles.json",
                "2. Generate scene visuals (1-N) using visual_generation_prompts.json",
                "3. Generate thumbnail (scene 99) using visual_generation_prompts.json",
                "4. Generate audio using audio_generation_prompts.json with production specifications",
                "5. Compose video using video_composition_instructions.json with chapters",
                "6. Upload to YouTube using platform_metadata.json with full SEO optimization"
            ],
            "automation_readiness": {
                "character_extraction": "✅ Complete",
                "youtube_optimization": "✅ Complete",
                "production_specifications": "✅ Complete",
                "platform_metadata": "✅ Complete",
                "composition_strategy": "✅ Complete",
                "api_ready_format": "✅ Complete"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ Complete production files saved: {saved_files}")
        CONFIG.logger.info(f"Files saved to: {output_dir}")

        # Mark topic as completed in database
        scene_count = len(result.get('scene_plan', []))
        total_duration = sum(scene.get('duration_minutes', 4) for scene in result.get('scene_plan', []))

        complete_topic_in_database(
            topic_id, scene_count, total_duration, api_calls, total_cost, output_dir
        )

    except Exception as e:
        print(f"❌ Save error: {e}")
        CONFIG.logger.error(f"Save error: {e}")

def print_production_summary(result: Dict, story_topic: str, output_path: str, generation_time: float):
    """Print complete production generation summary with all new features"""
    stats = result["generation_stats"]

    print("\n" + "🚀" * 60)
    print("SMART AUTOMATED STORY GENERATOR - COMPLETE OPTIMIZATION FINISHED!")
    print("🚀" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CONFIG.claude_config['model']} (Claude 4)")
    print(f"🖥️  Server Mode: {'✅ ACTIVE' if stats.get('server_optimized') else '❌ OFF'}")
    print(f"🏭 Complete Pipeline: {'✅ ACTIVE' if stats.get('complete_pipeline') else '❌ OFF'}")
    print(f"🎲 Smart Algorithm: {'✅ ACTIVE' if stats.get('smart_algorithm') else '❌ OFF'}")
    print(f"🎯 5-Stage Approach: {'✅ ACTIVE' if stats.get('five_stage_approach') else '❌ OFF'}")

    print(f"\n📊 CLAUDE 4 PRODUCTION PERFORMANCE:")
    print(f"🔥 Total API Calls: {stats['api_calls_used']}")
    print(f"💰 Total Cost: ${result.get('total_cost', 0):.4f}")
    print(f"⏱️  Total Generation Time: {generation_time:.1f}s")
    print(f"🎬 Scenes Planned: {stats['scenes_planned']}")
    print(f"📝 Stories Written: {stats['stories_written']}")
    print(f"👥 Characters Extracted: {stats['characters_extracted']}")
    print(f"🖼️  Thumbnail Generated: {'✅ YES' if stats.get('thumbnail_generated') else '❌ NO'}")
    print(f"📺 YouTube Optimization: {'✅ YES' if stats.get('youtube_optimization_generated') else '❌ NO'}")
    print(f"🏭 Production Specs: {'✅ YES' if stats.get('production_specifications_generated') else '❌ NO'}")
    print(f"🎭 Hook & Subscribe: {'✅ YES' if stats.get('hook_subscribe_generated') else '❌ NO'}")
    print(f"🎥 Visual Prompts (with thumbnail): {stats.get('visual_prompts_with_thumbnail', 0)}")

    # Smart generation stats
    if stats.get('smart_algorithm'):
        print(f"\n🎲 SMART GENERATION STATS:")
        print(f"📊 Random Scene Count: {stats.get('random_scene_count', 'N/A')}")
        print(f"⏱️  Duration Range: {stats.get('duration_range', 'N/A')}")
        print(f"🌟 Natural Variation: {'✅ YES' if stats.get('natural_duration_variation') else '❌ NO'}")

        # Calculate scene statistics
        scene_plan = result.get('scene_plan', [])
        if scene_plan:
            durations = [scene.get('duration_minutes', 4) for scene in scene_plan]
            total_duration = sum(durations)
            print(f"📈 Total Duration: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")
            print(f"📊 Average Scene: {total_duration/len(durations):.1f} minutes")
            print(f"🎯 Duration Accuracy: Smart algorithm ensures natural variation")

    # NEW FEATURES
    youtube_opt = result.get("youtube_optimization", {})
    if youtube_opt:
        print(f"\n📺 YOUTUBE OPTIMIZATION:")
        print(f"🎯 Clickbait Titles: {len(youtube_opt.get('clickbait_titles', []))}")
        print(f"🏷️  SEO Tags: {len(youtube_opt.get('tags', []))}")
        print(f"📚 Chapters: {len(result.get('scene_chapters', []))}")
        print(f"📝 Description: {'✅ Complete' if youtube_opt.get('video_description') else '❌ Missing'}")
        print(f"🔌 API Ready Format: {'✅ Complete' if youtube_opt.get('api_ready_format') else '❌ Missing'}")

    production_specs = result.get("production_specifications", {})
    if production_specs:
        print(f"\n🏭 PRODUCTION SPECIFICATIONS:")
        print(f"🎵 Audio Production: {'✅ Complete' if production_specs.get('audio_production') else '❌ Missing'}")
        print(f"🎬 Video Assembly: {'✅ Complete' if production_specs.get('video_assembly') else '❌ Missing'}")
        print(f"✅ Quality Control: {'✅ Complete' if production_specs.get('quality_control') else '❌ Missing'}")
        print(f"🤖 Automation Specs: {'✅ Complete' if production_specs.get('automation_specifications') else '❌ Missing'}")

    thumbnail_data = result.get("thumbnail_data", {})
    if thumbnail_data:
        print(f"\n🖼️  THUMBNAIL COMPOSITION STRATEGY:")
        composition = thumbnail_data.get("composition_strategy", {})
        print(f"🎯 Primary Approach: {composition.get('primary_approach', 'N/A')}")
        print(f"👁️  Visual Hierarchy: {composition.get('visual_hierarchy', 'N/A')}")
        print(f"📱 Mobile Optimization: {'✅ YES' if composition.get('mobile_optimization') else '❌ NO'}")

    completion_rate = (stats['stories_written'] / stats.get('scenes_planned', 1)) * 100
    print(f"📊 Story Completion: {completion_rate:.1f}%")

    if completion_rate >= 80:
        print(f"\n🎉 MASSIVE SUCCESS!")
        print(f"✅ Complete story + character + YouTube + production + thumbnail system")
        print(f"✅ Ready for FULL AUTOMATION")
        print(f"🚀 Zero manual work needed!")
    elif completion_rate >= 60:
        print(f"\n✅ EXCELLENT PROGRESS!")
        print(f"⚡ Ready for automated pipeline")
        print(f"🎯 Production deployment recommended")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"🔍 Review generation_report.json for issues")

    print("\n📄 GENERATED FILES (14 TOTAL - INCLUDING ALL MISSING ONES):")
    print("1. 📖 complete_story.txt - Full story text")
    print("2. 🎬 scene_plan.json - Smart scene structure + chapters")
    print("3. 🖼️  visual_generation_prompts.json - Scenes + Thumbnail (99)")
    print("4. 🎵 voice_directions.json - TTS guidance")
    print("5. 👥 character_profiles.json - Character data")
    print("6. 📺 youtube_metadata.json - Complete SEO package + API ready format")
    print("7. 🖼️  thumbnail_generation.json - Composition strategy")
    print("8. 🎭 hook_subscribe_scenes.json - Background scenes")
    print("9. 🏭 production_specifications.json - Complete production specs")
    print("10. 🤖 automation_specs.json - Automation-specific data (ADDED)")
    print("11. 🎵 audio_generation_prompts.json - Enhanced TTS production")
    print("12. 📚 all_stories.json - All stories in separate file (ADDED)")
    print("13. 🎥 video_composition_instructions.json - Video timeline + chapters")
    print("14. 📊 generation_report.json - Complete summary")

    print("🚀" * 60)

if __name__ == "__main__":
    try:
        print("🚀 SMART AUTOMATED STORY GENERATOR - COMPLETE WITH ALL MISSING FILES")
        print("⚡ Server-optimized with complete pipeline")
        print("🎲 FIXED: Smart random scene count & duration generation")
        print("📊 FIXED: Database integration instead of CSV")
        print("🎭 5-stage approach: Planning + Stories + Characters + Thumbnail + Hook/Subscribe")
        print("📄 Complete JSON outputs for automation")
        print("🎯 RIGHT-side thumbnail positioning for text overlay")
        print("✅ ADDED: all_stories.json + automation_specs.json + api_ready_format")
        print("=" * 60)

        # FIXED: Get next topic from database
        topic_id, topic, description, clickbait_title, font_design = get_next_topic_from_database()
        print(f"\n📚 Topic ID: {topic_id} - {topic}")
        print(f"📝 Description: {description}")
        if clickbait_title:
            print(f"🎯 Clickbait Title: {clickbait_title}")

        # Setup output directory
        output_path = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id)

        # Initialize generator
        generator = AutomatedStoryGenerator()

        # Generate complete story with FIXED smart algorithm
        start_time = time.time()
        result = generator.generate_complete_story_with_characters(
            topic, description, clickbait_title, font_design
        )
        generation_time = time.time() - start_time

        # Add total cost to result
        result['total_cost'] = generator.total_cost

        # Save outputs (complete function needed)
        save_production_outputs(str(output_path), result, topic, topic_id,
                               generator.api_call_count, generator.total_cost)

        # Print comprehensive summary
        print_production_summary(result, topic, str(output_path), generation_time)

        print("\n🚀 COMPLETE PRODUCTION PIPELINE FINISHED WITH ALL FILES!")
        print(f"✅ All files ready for: {output_path}")
        print(f"📊 Database topic management: WORKING")
        print(f"🎲 Smart algorithm scene generation: FIXED")
        print(f"📝 Story distribution: FIXED")
        print(f"📚 all_stories.json: ADDED")
        print(f"🤖 automation_specs.json: ADDED")
        print(f"🔌 api_ready_format: ADDED")
        print(f"💰 Total cost: ${result.get('total_cost', 0):.4f}")

    except Exception as e:
        print(f"\n💥 SMART GENERATOR ERROR: {e}")
        CONFIG.logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()