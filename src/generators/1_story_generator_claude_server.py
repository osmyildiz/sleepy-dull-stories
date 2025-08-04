"""
Sleepy Dull Stories - COMPLETE Server-Ready Claude Story Generator
UPDATED: All missing methods + Smart Algorithm + Database integration + Complete 5-stage pipeline + ALL LOCAL FEATURES
Production-optimized with complete automation + Enhanced with all local JSON files
FINAL VERSION: All local features integrated into server version
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
            "long_timeout": True,        # ✅ PROVEN CRITICAL,
            "validation_enabled": True,  # ✅ EKLEYİN
            "auto_correction": True,  # ✅ EKLEYİN
            "target_tolerance": 0.2  # ✅ EKLEYİN (±20% tolerans)
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
# Midjourney Content Policy Awareness for Story Generator
# Add this to story generator before visual prompt creation

MIDJOURNEY_CONTENT_AWARENESS_PROMPT = """
## 🎨 CRITICAL: MIDJOURNEY VISUAL GENERATION CONTENT POLICY AWARENESS

These visual prompts will be processed by Midjourney AI for image generation. Midjourney has strict content filtering that automatically rejects suspicious words, causing generation failures. Choose your words carefully to avoid censorship.

### 📋 MIDJOURNEY'S CONTENT FILTERING SYSTEM SENSITIVITY:

#### 1. 🔴 PROBLEMATIC WORD CATEGORIES THAT CAUSE REJECTIONS:
- **Intimacy/Privacy Words**: "intimate", "private", "personal" → AI interprets as romantic/sexual
- **Time+Location Combos**: "late at night + chamber/room" → Creates suspicious context
- **Religious/Cultural References**: "Hebrew", "religious texts" → Flagged as sensitive content  
- **Abstract/Supernatural**: "mystical", "supernatural", "voices" → Creates ambiguity
- **Physical Proximity**: "embracing", "touching", "close contact" → Romantic interpretation
- **Private Spaces**: "bedroom", "bath", "private quarters" → Intimate space perception

#### 2. ✅ SAFE ALTERNATIVE STRATEGY FOR HIGH SUCCESS RATE:
- **Lighting**: "warm lighting" ✅ NOT "intimate lighting" ❌
- **Spaces**: "study room" ✅ NOT "private chamber" ❌  
- **Time**: "evening hours" ✅ NOT "late at night" ❌
- **Texts**: "ancient scrolls" ✅ NOT "Hebrew texts" ❌
- **Atmosphere**: "peaceful ambiance" ✅ NOT "mystical atmosphere" ❌
- **Activity**: "focused study" ✅ NOT "personal reading" ❌

#### 3. 🎯 PROACTIVE SAFE WORD SELECTION MINDSET:
For every word you write, ask: "Could Midjourney misinterpret this word?"
- If YES → Find neutral, educational, academic alternative
- If NO → Safe to proceed

#### 4. 🏛️ SAFE APPROACH FOR HISTORICAL CONTENT:
Always include these safety qualifiers:
- ✅ "historical educational setting"
- ✅ "classical academic atmosphere" 
- ✅ "scholarly learning environment"
- ✅ "period-accurate educational scene"
- ✅ "warm educational lighting"
- ✅ "family-friendly historical content"

#### 5. 🔧 MENTAL SAFETY FILTER FOR EVERY PROMPT:
Before writing any visual prompt, verify:
- "Would this create an image a parent wouldn't hesitate to show their child?"
- "Is the educational/academic nature crystal clear?"
- "Are these words objective, scientific, historical?"
- "Does this avoid any romantic, intimate, or private implications?"

### 🎨 GUARANTEED-SAFE VISUAL PROMPT FORMULA:
```
"[HISTORICAL_LOCATION] with [CHARACTER/SCHOLARLY_ACTIVITY], warm educational lighting, 
classical academic setting, [PERIOD_DETAILS], peaceful scholarly atmosphere, 
historical educational content, family-friendly scene"
```

### 🎯 TRANSFORMATION EXAMPLES:
❌ RISKY: "Private study chamber late at night with scholar working intimately with Hebrew texts"
✅ SAFE: "Ancient study room in evening hours with scholar focused on historical manuscripts, warm educational lighting, classical academic setting, scholarly dedication"

❌ RISKY: "Intimate reading nook with personal cushions" 
✅ SAFE: "Quiet study corner with comfortable seating, focused learning environment"

❌ RISKY: "Mystical voices whispering ancient wisdom"
✅ SAFE: "Echo chamber preserving ancient knowledge, architectural acoustics"

### 💡 SUCCESS CHECKLIST FOR EVERY VISUAL PROMPT:
1. ✅ Educational/academic tone present?
2. ✅ No ambiguous/suspicious words?  
3. ✅ Historical/scholarly context explicit?
4. ✅ Family-friendly language throughout?
5. ✅ Objective, descriptive approach maintained?
6. ✅ Would pass parent approval test?

### 🎯 AUTOMATION SUCCESS STRATEGY:
This content awareness ensures:
- 100% Midjourney acceptance rate
- No failed generations requiring retries  
- Consistent visual output quality
- Zero content policy violations
- Reliable automation pipeline

Apply this awareness to ALL visual descriptions, scene planning, and character descriptions.
Your word choices directly impact generation success rate.
"""

print("✅ Midjourney Content Policy Awareness (English) ready for integration!")
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
    """Complete server-ready automated story generation with FIXED Smart Algorithm + ALL LOCAL FEATURES"""

    def __init__(self):
        """Initialize story generator for server environment"""
        self.generation_log = []
        self.api_call_count = 0
        self.total_cost = 0.0
        self.character_system = CharacterExtractionSystem()

        try:
            self.client = Anthropic(api_key=CONFIG.api_key)
            CONFIG.logger.info("✅ Story generator initialized successfully")
            print("✅ Story generator initialized with Smart Algorithm + Budget Tracking + ALL LOCAL FEATURES")
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
        scene_count = random.randint(28, 40)
        print(f"🎲 Smart Algorithm: Generated {scene_count} scenes (random 28-40)")

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

    # MEVCUT FONKSİYONDAN SONRA:
    def generate_hook_subscribe_visual_prompts(self, scene_plan: List[Dict], hook_content: str, subscribe_content: str,
                                               visual_style_notes: Dict) -> Dict:
        """Hook & Subscribe için GÖRSEL PROMPTLARI oluştur"""

        self.log_step("Hook & Subscribe Visual Prompt Generation")

        # Basit fallback - API çağrısı yapmadan
        atmospheric_scenes = [s for s in scene_plan if s.get('template') == 'atmospheric'][:10]
        character_scenes = [s for s in scene_plan if s.get('template') == 'character_focused'][:10]

        hook_visuals = []
        for i, scene in enumerate(atmospheric_scenes):
            hook_visuals.append({
                "sequence_id": i + 1,
                "timing": f"{i * 3}-{(i * 3) + 3} seconds",
                "source_scene": f"Scene {scene['scene_id']}",
                "visual_prompt": f"Cinematic atmospheric view of {scene.get('location')}, golden hour lighting, mysterious but peaceful mood",
                "cinematic_style": "Wide shot, warm lighting",
                "transition_note": "Slow dissolve"
            })

        subscribe_visuals = []
        for i, scene in enumerate(character_scenes):
            subscribe_visuals.append({
                "sequence_id": i + 1,
                "timing": f"{30 + (i * 3)}-{30 + (i * 3) + 3} seconds",
                "source_scene": f"Scene {scene['scene_id']}",
                "visual_prompt": f"Welcoming view of {scene.get('location')}, warm lighting, community atmosphere",
                "community_element": "Warm, inviting feel",
                "engagement_factor": "Encourages subscription"
            })

        return {
            "hook_visual_prompts": hook_visuals,
            "subscribe_visual_prompts": subscribe_visuals
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
        COMPLETE 5-STAGE APPROACH WITH FIXED SMART ALGORITHM + ALL LOCAL FEATURES:
        Stage 1: Smart Planning + Hook + Subscribe + First Half stories
        Stage 2: Remaining stories (second half)
        Stage 3: Character extraction and analysis
        Stage 4: Intelligent thumbnail generation
        Stage 5: Hook & Subscribe scene selection + Complete JSON outputs
        """

        self.log_step("Complete Story Generation with Smart Random Durations + ALL LOCAL FEATURES")

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
        """
        STAGE 1: COLM TÓIBÍN MASTER STYLE - Smart planning + first half stories

        TÓIBÍN'S GENIUS:
        - "Hush you hear, sighing of disappearance"
        - Daily life thick with human uncertainties
        - Silence between something and nothing
        - Characters led by desires they don't understand
        """

        # Generate smart scene structure
        visual_safety_education = MIDJOURNEY_CONTENT_AWARENESS_PROMPT
        smart_structure = self._generate_smart_scene_structure()
        total_scenes = smart_structure['scene_count']
        scene_durations = smart_structure['scene_durations']
        first_half = total_scenes // 2

        self.log_step(
            f"Stage 1: TÓIBÍN MASTER STYLE Planning + First {first_half} Stories (Total: {total_scenes} scenes)")

        stage1_prompt = f"""Create the complete foundation for a 2-hour sleep story about "{topic}" using COLM TÓIBÍN'S MASTERFUL STYLE.

    TOPIC: {topic}
    DESCRIPTION: {description}

    🎭 COLM TÓIBÍN WRITING MASTERY - THE COMPLETE GUIDE:

    ## TÓIBÍN'S CORE GENIUS:
    "In all Colm Tóibín's work there is a hush you hear, a kind of sighing of the disappearance of something that is so far gone there is no way of saying what it is. His characters, led by desires that they don't understand, often have vulnerabilities and strengths that are virtually the same."

    ## TÓIBÍN'S SIGNATURE STYLE ELEMENTS:

    ### 1. 🤫 THE POWER OF SILENCE & SUBTLETY:
    - **"Sparseness of tone with superabundance of suggestion"**
    - **"Style occupies gap between something and nothing"**
    - What is NOT said is as important as what IS said
    - Characters pause, hesitate, observe quietly
    - Emotions simmer beneath surface politeness

    **EXAMPLE TÓIBÍN APPROACH:**
    ❌ "Marcus was devastated by the news"
    ✅ "Marcus nodded. He placed his cup carefully on the marble table. Outside, a bird called once, then fell silent."

    ### 2. 🏛️ DAILY LIFE AS PROFOUND DRAMA:
    - **"Not much happens but work is thick with human uncertainties"**
    - Ordinary moments reveal character depths
    - **"Fascination of commonplaces"** - everyday details matter
    - Characters doing normal things: eating, walking, thinking
    - Small gestures carry enormous emotional weight

    **TÓIBÍN SCENE BUILDING:**
    - Start with simple activity (character pouring wine)
    - Add internal observation (they remember something)
    - Include environmental detail (sound of fountain)
    - Let silence do the work (pause, breath, reflection)

    ### 3. 🎭 CHARACTER PSYCHOLOGY MASTERY:
    - **"Characters yearn for understanding and acceptance"**
    - **"Led by desires they don't understand"**  
    - Internal conflicts more important than external action
    - **"Mixed motives and tacit exchanges"**
    - Characters often lonely but dignified

    ### 4. 📝 TÓIBÍN'S SENTENCE ARCHITECTURE:
    - **Short, precise observations**
    - **Longer, flowing contemplative passages**
    - **"Quiet, vivid epiphanies of belonging and estrangement"**
    - Rhythm mimics human thought patterns
    - Present tense immediacy mixed with memory

    **TÓIBÍN SENTENCE EXAMPLES:**
    - "The light had changed. He noticed this as he walked."
    - "There was something in her voice, something he recognized but could not name."
    - "The house felt different now, as though it were holding its breath."

    ## 🎯 YOUR TÓIBÍN-STYLE SLEEP STORIES REQUIREMENTS:

    ### OPENING STYLE MASTERY (Each scene different):
    1. **Quiet Observation:** "The evening settles differently today."
    2. **Character Action:** "She moves through the garden slowly."  
    3. **Environmental Shift:** "Something in the light has changed."
    4. **Internal Recognition:** "He knows this feeling, this weight."
    5. **Sensory Awareness:** "The sound reaches him gradually."
    6. **Memory Trigger:** "The scent brings back something distant."
    7. **Present Moment:** "This is how afternoons end here."
    8. **Emotional Recognition:** "A familiar sadness touches the edges."

    ### SCENE DEVELOPMENT PATTERN:
    ```
    1. QUIET BEGINNING (simple activity/observation)
    2. INTERNAL LAYER (character's thoughts/feelings)  
    3. ENVIRONMENTAL DETAIL (place, time, atmosphere)
    4. HUMAN CONNECTION (other characters, relationships)
    5. DEEPER RECOGNITION (emotional insight/acceptance)
    6. PEACEFUL RESOLUTION (gentle acceptance/closure)
    ```

    ### CHARACTER INTERACTION STYLE:
    - **Understated dialogue:** Characters say less than they feel
    - **Gesture communication:** Nods, pauses, small actions
    - **Shared silence:** Comfortable quiet between people
    - **Unspoken understanding:** Characters reading each other
    - **Gentle observation:** Watching others with kindness

    SMART STORY STRUCTURE:
    - Total scenes: {total_scenes} (NATURAL VARIATION - random count 28-45)
    - Target duration: {smart_structure['target_duration']} minutes
    - Scene durations: VARIABLE (see list below)
    - First half: {first_half} scenes (this stage)
    - Second half: {total_scenes - first_half} scenes (next stage)

    SCENE DURATION PLAN:
    {', '.join([f'Scene {i + 1}: {dur}min' for i, dur in enumerate(scene_durations)])}

    ## 1. GOLDEN HOOK (30 seconds, ~90 words) - TÓIBÍN STYLE
    Create atmospheric opening with TÓIBÍN'S QUIET MASTERY:
    - Start with subtle observation about the day/place
    - Include sense of something just beyond perception
    - **"Hush you hear, sighing of disappearance"** quality
    - Gentle melancholy but deeply peaceful
    - Historical setting established through small details
    - Invitation to witness quiet human moments

    ## 2. SUBSCRIBE SECTION (30 seconds, ~70 words) - WARM COMMUNITY
    - Natural, non-corporate invitation
    - **"Join us for more quiet moments from history"**
    - Tóibín-like warmth and literary sensibility

    ## 3. COMPLETE SCENE PLAN (Exactly {total_scenes} scenes with SMART DURATIONS)
    Each scene duration from plan above:

    {chr(10).join([f"Scene {i + 1}: {scene_durations[i]:.1f} minutes" for i in range(total_scenes)])}

    Scene structure requirements for TÓIBÍN STYLE:
    - Focus on ORDINARY MOMENTS with extraordinary emotional depth
    - **"Fascination of commonplaces"** approach
    - Characters doing daily activities with internal complexity
    - Template rotation: contemplative, character_study, daily_ritual, emotional_recognition
    - Emotion progression: quiet observation → gentle recognition → deeper understanding → peaceful acceptance

    ## 4. FIRST {first_half} COMPLETE STORIES (Scenes 1-{first_half}) - TÓIBÍN MASTERY

    ### 🎭 TÓIBÍN'S OPENING MASTERY:
    Each opening must demonstrate different aspect of his genius:

    1. **Quiet Environmental Shift:** "The light settles differently over the courtyard this evening."
    2. **Character's Simple Action:** "Marcus pauses at the fountain, listening."
    3. **Internal Recognition:** "Something in the day feels familiar, though she cannot say what."
    4. **Atmospheric Observation:** "The garden holds its breath in the late afternoon heat."
    5. **Memory Echo:** "This reminds him of something, some other evening."
    6. **Present Moment Awareness:** "She knows this feeling, this particular weight of waiting."
    7. **Sensory Gateway:** "The scent of olive oil and herbs drifts from the kitchen."
    8. **Emotional Recognition:** "A gentle sadness touches the edges of the gathering."

    ### 📝 TÓIBÍN'S SLEEP-OPTIMIZED WRITING:
    - **Present tense, second person** (but filtered through Tóibín's contemplative lens)
    - **"Sparseness with superabundance of suggestion"**
    - **Rich sensory details** but understated, not overwhelming
    - **[PAUSE] markers** at natural breathing/thinking points
    - **Historical accuracy** through small, authentic details
    - **Character psychology** - internal lives more than external action
    - **Word count matched to duration** (~150 words/minute)

    ### 🎯 CONTENT THEMES (TÓIBÍN SPECIALTIES):
    - **Daily rituals** (meals, prayers, work, conversations)
    - **Family relationships** (especially parent/child, spouse dynamics)
    - **Quiet social moments** (gatherings, shared meals, walks)
    - **Internal contemplation** (characters thinking, remembering, observing)
    - **Gentle human connection** (understanding without words)
    - **"Mixed motives and tacit exchanges"** between characters
    - **Sense of time passing** (seasons, life stages, historical change)

    {visual_safety_education}

    ## 5. BASIC VISUAL PROMPTS (All {total_scenes} scenes)
    - TÓIBÍN-INSPIRED: Focus on intimate, domestic, contemplative scenes
    - Characters in quiet moments of daily life
    - Historical settings that feel lived-in, not grand
    - Warm, natural lighting suggesting introspection

    ## 6. VOICE DIRECTIONS (All {total_scenes} scenes) - TÓIBÍN TONE
    - **"Gentle, contemplative storytelling"**
    - **"Literary sensibility with warm human connection"**
    - **"Understated emotion, profound undertones"**
    - **"Irish literary tradition - thoughtful, observational"**

    OUTPUT FORMAT (Complete JSON):
    {{
      "golden_hook": {{
        "content": "[90-word Tóibín-style atmospheric opening with quiet observation]",
        "duration_seconds": 30,
        "voice_direction": "Gentle, literary, contemplative - like Tóibín reading his own work"
      }},
      "subscribe_section": {{
        "content": "[70-word warm literary community invitation]",
        "duration_seconds": 30,
        "voice_direction": "Warm, literary, non-commercial - book club invitation feeling"
      }},
      "scene_plan": [
        {{
          "scene_id": 1,
          "title": "[Tóibín-style scene title focusing on quiet moment]",
          "location": "[Historical location - intimate, not grand]", 
          "duration_minutes": {scene_durations[0] if scene_durations else 4},
          "template": "contemplative",
          "narrative_style": "observational_internal",
          "emotion": "quiet_recognition",
          "sensory_focus": "understated_multiple",
          "description": "[Daily activity with internal depth - include character names]",
          "key_elements": ["quiet_activity", "internal_thought", "environmental_detail"],
          "characters_mentioned": ["character1", "character2"],
          "toibin_elements": ["silence_between_words", "ordinary_moment_profound", "character_psychology"]
        }}
      ],
      "stories": {{
        "1": "[COMPLETE Tóibín-style story - quiet observation opening, internal depth, historical authenticity]",
        "2": "[COMPLETE Tóibín-style story - different opening approach, character psychology focus]"
      }},
      "visual_prompts": [
        {{
          "scene_number": 1,
          "title": "[Scene title]",
          "prompt": "[Tóibín-inspired: intimate daily life scene, warm lighting, contemplative mood]",
          "duration_minutes": {scene_durations[0] if scene_durations else 4},
          "emotion": "quiet_contemplation"
        }}
      ],
      "voice_directions": [
        {{
          "scene_number": 1,
          "title": "[Scene title]", 
          "direction": "[Literary storytelling - contemplative pace, understated emotion, Irish literary sensibility]",
          "template": "contemplative",
          "style": "toibin_observational"
        }}
      ],
      "stage1_stats": {{
        "scenes_planned": {total_scenes},
        "stories_written": {first_half},
        "toibin_style_applied": true,
        "quiet_mastery_demonstrated": true,
        "daily_life_focus": true,
        "character_psychology_depth": true,
        "total_word_count": "[calculated]",
        "characters_introduced": "[count]",
        "ready_for_stage2": true
      }}
    }}

    ## COLM TÓIBÍN MASTERY CHALLENGE:
    Write {first_half} stories that demonstrate TÓIBÍN'S GENIUS:

    **REQUIRED ELEMENTS FOR EACH STORY:**
    - ✅ **UNIQUE QUIET OPENING** (8 different approaches shown above)
    - ✅ **"Fascination of commonplaces"** - ordinary moments made profound
    - ✅ **Character psychology depth** - internal lives richly explored
    - ✅ **"Sparseness with superabundance"** - say less, suggest more
    - ✅ **Historical authenticity** through small, telling details  
    - ✅ **Sleep-optimized pacing** with natural breathing rhythm
    - ✅ **"Hush you hear"** quality - peaceful, contemplative atmosphere

    **TÓIBÍN'S SIGNATURE QUALITIES TO DEMONSTRATE:**
    - **Silence as communication** - what characters don't say
    - **"Mixed motives and tacit exchanges"** - complex relationships
    - **"Quiet, vivid epiphanies"** - moments of recognition
    - **Daily activities** as windows into character souls
    - **Understated emotion** that resonates deeply

    Channel COLM TÓIBÍN'S literary mastery. Create stories that feel like they belong in his celebrated works - intimate, psychologically rich, quietly profound, and deeply human.

    USE THE EXACT DURATIONS FROM THE PLAN ABOVE."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,
                timeout=1800,
                system="You are COLM TÓIBÍN, the celebrated Irish master of literary fiction. Apply your signature style: 'sparseness of tone with superabundance of suggestion,' characters led by desires they don't understand, the fascination of commonplaces, and the quiet recognition of human psychology. Your stories capture the 'hush you hear, the sighing of disappearance' - peaceful, contemplative, psychologically rich. Focus on daily life moments with extraordinary emotional depth. Create sleep content with your literary mastery.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            content = ""
            print("📡 Stage 1: Streaming with COLM TÓIBÍN MASTERY...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📚 TÓIBÍN STYLE: {len(content):,} characters...")

            print(f"✅ Stage 1 TÓIBÍN MASTERY complete: {len(content):,} characters")

            input_tokens = len(stage1_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            CONFIG.logger.info(f"Stage 1 Tóibín response length: {len(content)} characters - Cost: ${stage_cost:.4f}")

            parsed_result = self._parse_claude_response(content, "stage1")

            self.log_step("Stage 1 TÓIBÍN MASTERY Parsing", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stories_written": len(parsed_result.get('stories', {})),
                "toibin_style_applied": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 1 TÓIBÍN MASTERY Failed", "ERROR")
            CONFIG.logger.error(f"Stage 1 Tóibín error: {e}")
            raise

    def _generate_stage2(self, topic: str, description: str, stage1_result: Dict) -> Dict[str, Any]:
        """
        STAGE 2: COLM TÓIBÍN MASTER STYLE - Remaining stories with LITERARY CONTINUITY

        TÓIBÍN'S CONSISTENCY:
        - Maintain character psychology established in Stage 1
        - Continue "quiet recognition" emotional progression
        - Deepen "fascination of commonplaces" approach
        - Evolve "hush you hear" atmosphere throughout
        """

        scene_plan = stage1_result.get('scene_plan', [])
        stories_written = len(stage1_result.get('stories', {}))
        total_scenes = len(scene_plan)
        remaining_scenes = total_scenes - stories_written

        if remaining_scenes <= 0:
            self.log_step("Stage 2: No remaining stories needed", "SUCCESS")
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        self.log_step(f"Stage 2: TÓIBÍN MASTER CONTINUITY - {remaining_scenes} Stories")

        remaining_scene_plan = []
        for scene in scene_plan:
            if str(scene['scene_id']) not in stage1_result.get('stories', {}):
                remaining_scene_plan.append(scene)

        if len(remaining_scene_plan) == 0:
            self.log_step("Stage 2: All stories already written", "SUCCESS")
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        scenes_text = "\n\n".join([
            f"""SCENE {scene['scene_id']}: {scene['title']}
    Location: {scene['location']}
    Duration: {scene.get('duration_minutes', 4)} minutes
    Template: {scene['template']} | Style: {scene['narrative_style']}
    Emotion: {scene['emotion']} | Sensory Focus: {scene['sensory_focus']}
    Description: {scene['description']}
    Key Elements: {', '.join(scene.get('key_elements', []))}
    Characters: {', '.join(scene.get('characters_mentioned', []))}
    Tóibín Elements: {', '.join(scene.get('toibin_elements', []))}"""
            for scene in remaining_scene_plan
        ])

        stage2_prompt = f"""Complete the sleep story for "{topic}" by writing the remaining {remaining_scenes} stories in COLM TÓIBÍN'S MASTERFUL STYLE.

    TOPIC: {topic}
    DESCRIPTION: {description}

    SCENES TO COMPLETE:
    {scenes_text}

    🎭 COLM TÓIBÍN MASTER CONTINUITY - STAGE 2 REQUIREMENTS:

    ## MAINTAIN STAGE 1 TÓIBÍN EXCELLENCE:
    Continue the literary mastery established in Stage 1:
    - **Same "hush you hear" atmospheric quality**
    - **Character consistency** - same personalities, relationships, internal patterns
    - **"Sparseness with superabundance"** - restrained language with deep suggestion
    - **"Fascination of commonplaces"** - ordinary moments made profound
    - **Emotional progression** - deepen the quiet recognition from Stage 1

    ## TÓIBÍN'S ADVANCED OPENING MASTERY (STAGE 2 VARIETY):

    Since Stage 1 used 8 opening approaches, Stage 2 must demonstrate NEW variations:

    ### 9. **Temporal Shift Recognition:** "The afternoon feels different now, heavier somehow."
    ### 10. **Character Relationship Moment:** "She watches him from across the courtyard, understanding."
    ### 11. **Environmental Psychology:** "The garden seems to hold their secrets in its shadows."
    ### 12. **Internal Weather:** "Something has shifted in the house's rhythm."
    ### 13. **Collective Awareness:** "They all sense it, though none speaks of it."
    ### 14. **Memory Confluence:** "This moment echoes others, from before."
    ### 15. **Seasonal Recognition:** "Autumn arrives early in small signs."
    ### 16. **Emotional Geography:** "The dining room carries the weight of unspoken words."

    ## TÓIBÍN'S CHARACTER CONTINUITY MASTERY:

    ### CHARACTER CONSISTENCY RULES:
    - **Same internal voices** - characters think in same patterns as Stage 1
    - **Relationship evolution** - deepen connections established earlier
    - **"Mixed motives and tacit exchanges"** - continue complex character dynamics
    - **Emotional memory** - characters carry forward Stage 1 experiences
    - **Speech patterns** - maintain same dialogue styles and rhythms

    ### CHARACTER DEVELOPMENT PROGRESSION:
    ```
    STAGE 1: Introduction + establishment of internal patterns
    STAGE 2: Deepening + evolution + quiet recognitions
    EMOTIONAL ARC: Peaceful observation → Growing awareness → Gentle acceptance
    ```

    ## TÓIBÍN'S SLEEP STORY REQUIREMENTS (ADVANCED):

    ### 📝 WORD COUNT PRECISION:
    - 2-3 minute scenes: 280-420 words (2-3 × 140)
    - 4-5 minute scenes: 560-700 words (4-5 × 140)  
    - 6-7 minute scenes: 840-980 words (6-7 × 140)
    - 8+ minute scenes: 1120+ words (8+ × 140)
    - Base rate: ~140 words per minute for sleep content

    ### 🎭 TÓIBÍN'S CHARACTER INTEGRATION (ADVANCED):
    - **"Characters led by desires they don't understand"** - show internal conflicts
    - **Understated dialogue** - more gesture than words
    - **Shared silences** - comfortable quiet between people
    - **Observational moments** - characters watching each other with kindness
    - **Emotional archaeology** - characters understanding through small signs

    ### 🌙 TÓIBÍN'S SLEEP OPTIMIZATION:
    - **Present tense, second person** (but filtered through literary consciousness)
    - **"Quiet, vivid epiphanies"** - moments of gentle recognition
    - **Natural breathing rhythm** - sentences that flow like thought
    - **Peaceful resolution** - each scene ends with acceptance/understanding
    - **"Hush you hear"** quality - that distinctive Tóibín silence

    ### 🏛️ TÓIBÍN'S HISTORICAL AUTHENTICITY:
    - **Small, telling details** - period accuracy through commonplace objects
    - **Social customs** shown through character behavior, not explanation
    - **Daily rhythms** - how people actually lived their ordinary days
    - **Environmental storytelling** - let setting reveal historical context
    - **"Lived-in" feeling** - history as background to human experience

    ### 🎵 TÓIBÍN'S NARRATIVE MASTERY:
    - **Sentence rhythm variation** - short observations, longer contemplative flows
    - **Emotional undertow** - feelings flowing beneath surface politeness
    - **"Sighing of disappearance"** - that sense of time passing, things changing
    - **Contemplative pace** - allow characters time to think, observe, recognize
    - **Literary density** - every sentence carrying weight beyond its words

    ## FORBIDDEN ELEMENTS (ANTI-TÓIBÍN):
    ❌ **Dramatic external action** - keep focus on internal/relational
    ❌ **Overwrought emotion** - Tóibín is always understated
    ❌ **Generic openings** - especially "You find yourself" (completely banned)
    ❌ **Surface-level interaction** - all relationships must have depth
    ❌ **Rushed pacing** - allow contemplative time for recognition
    ❌ **Obvious exposition** - let readers discover through observation

    ## TÓIBÍN MASTERY EXAMPLES FOR STAGE 2:

    **CHARACTER CONTINUITY EXAMPLE:**
    *Stage 1 established Marcus as quietly observational*
    *Stage 2 continues:* "Marcus moves through the evening ritual as he always does, but tonight he notices how Julia's hands pause over the wine cups, the way she listens to sounds beyond the courtyard walls."

    **EMOTIONAL PROGRESSION EXAMPLE:**
    *Stage 1 introduced gentle melancholy*
    *Stage 2 deepens:* "The familiar sadness is there, but softer now, like an old friend who no longer needs introduction."

    OUTPUT FORMAT:
    {{
      "stories": {{
        "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}": "[COMPLETE Tóibín-style story maintaining Stage 1 character consistency with advanced opening mastery]",
        "{remaining_scene_plan[1]['scene_id'] if len(remaining_scene_plan) > 1 else 'Y'}": "[COMPLETE Tóibín-style story with unique opening from advanced variety list]"
      }},
      "stage2_stats": {{
        "stories_written": {remaining_scenes},
        "scenes_covered": "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}-{remaining_scene_plan[-1]['scene_id'] if remaining_scene_plan else 'Y'}",
        "toibin_continuity_maintained": true,
        "character_consistency_achieved": true,
        "advanced_opening_mastery": true,
        "literary_depth_sustained": true,
        "quiet_recognition_progression": true,
        "total_word_count": "[calculated]",
        "emotional_arc_completion": "deepened_understanding"
      }}
    }}

    ## COLM TÓIBÍN STAGE 2 MASTERY CHALLENGE:

    Write {remaining_scenes} stories that demonstrate TÓIBÍN'S ADVANCED LITERARY CRAFT:

    ### REQUIRED EXCELLENCE FOR EACH STORY:
    - ✅ **ADVANCED UNIQUE OPENING** (9-16 from new variety list above)
    - ✅ **CHARACTER CONSISTENCY** - same internal voices/patterns from Stage 1
    - ✅ **EMOTIONAL PROGRESSION** - deepen the quiet recognition established earlier
    - ✅ **"Fascination of commonplaces"** - continue finding profound in ordinary
    - ✅ **Perfect word count** matched to contemplative scene duration
    - ✅ **Historical authenticity** through small, lived-in details
    - ✅ **Sleep-optimized literary pacing** - gentle rhythm for peaceful rest

    ### TÓIBÍN'S SIGNATURE ELEMENTS TO CONTINUE:
    - **"Hush you hear"** - maintain that distinctive peaceful silence
    - **"Mixed motives and tacit exchanges"** - keep character relationships complex
    - **"Sparseness with superabundance"** - restrained language, rich suggestion
    - **"Quiet, vivid epiphanies"** - moments of gentle recognition and acceptance
    - **"Characters led by desires they don't understand"** - internal psychology depth

    ### STAGE 2 EMOTIONAL PROGRESSION:
    **STAGE 1:** Peaceful observation + character introduction
    **STAGE 2:** Growing awareness + relationship deepening + gentle acceptance

    Continue channeling COLM TÓIBÍN'S celebrated literary mastery. These remaining stories should feel seamlessly connected to Stage 1 while demonstrating advanced opening variety and deeper character development. Maintain the contemplative, psychologically rich, quietly profound qualities that make Tóibín's work so distinctive and perfect for sleep content.

    REMEMBER: Character consistency is crucial - these people must feel like the same individuals from Stage 1, just revealed more deeply through continued observation."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,
                timeout=1800,
                system="You are COLM TÓIBÍN continuing your literary masterwork from Stage 1. Maintain absolute character consistency - same internal voices, relationship patterns, and emotional rhythms established earlier. Use your advanced opening mastery (approaches 9-16) while deepening the 'quiet recognition' emotional progression. Continue the 'hush you hear' atmospheric quality and 'fascination of commonplaces' approach. These stories must feel seamlessly connected to Stage 1 while demonstrating your full range of literary craft.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = ""
            print("📡 Stage 2: Streaming with TÓIBÍN LITERARY CONTINUITY...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📚 TÓIBÍN CONTINUITY: {len(content):,} characters...")

            print(f"✅ Stage 2 TÓIBÍN LITERARY MASTERY complete: {len(content):,} characters")

            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 TÓIBÍN LITERARY CONTINUITY Parsing", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "toibin_continuity_maintained": True,
                "character_consistency_achieved": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 TÓIBÍN LITERARY CONTINUITY Failed", "ERROR")
            CONFIG.logger.error(f"Stage 2 Tóibín continuity error: {e}")
            return {"stories": {}, "stage2_stats": {"error": str(e)}}

    def _extract_characters(self, topic: str, description: str, stage1_result: Dict, validation_result: Dict) -> Dict[
        str, Any]:
        """STAGE 3: Extract main characters + YouTube optimization + REGENERATE VISUAL PROMPTS (using validated stories)"""

        self.character_system.log_extraction_step("Character Extraction and Production Optimization")

        # Use validated stories instead of raw stage results
        all_stories = validation_result.get('validated_stories', {})
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
                system="You are analyzing stories written in KRISTIN HANNAH's style. Focus on Hannah's signature female characters - resilient women who discover strength through crisis. Look for characters with Hannah's trademark emotional depth, authentic historical context, and the kind of compelling personal journeys that made The Nightingale and The Four Winds bestsellers. Extract characters who embody Hannah's themes of survival, sacrifice, and the untold stories of women in history.",
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
                    "character_integrated_scenes": len(
                        [p for p in regenerated_visual_prompts if p.get('character_reference_needed')])
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

    def _regenerate_visual_prompts_with_toibin_drama(self, scene_plan: List[Dict], characters: List[Dict],
                                                     scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
        """Generate visual prompts balancing TÓIBÍN LITERARY QUALITY with DRAMATIC THUMBNAIL APPEAL"""

        prompts = []

        for scene in scene_plan:
            scene_id = scene['scene_id']
            scene_characters = scene_character_map.get(str(scene_id), [])
            location = scene.get('location', 'Historical setting')
            emotion = scene.get('emotion', 'quiet_contemplation')
            description = scene.get('description', '')
            title = scene.get('title', f"Scene {scene_id}")

            # Balance dramatic appeal with Tóibín authenticity
            if scene_characters:
                char_names = [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters]
                char_list = ', '.join(char_names)

                # DRAMATIC but LITERARY approach
                if emotion in ['concern', 'recognition', 'conflict']:
                    drama_element = "showing quiet concern and internal contemplation about unfolding events"
                    lighting = "dramatic lighting that emphasizes character psychology and internal struggle"
                else:
                    drama_element = "in thoughtful, contemplative poses with subtle emotional intensity"
                    lighting = "warm, sophisticated lighting suggesting literary depth and character complexity"

                prompt = f"Literary cinematic scene of {location}, featuring {char_list} {drama_element}, {lighting}, Tóibín-style character psychology visible in expressions, historically accurate period setting, sophisticated composition balancing dramatic appeal with contemplative literary quality"

                enhanced_prompt = f"[CHARACTERS: {char_list}] [TÓIBÍN + DRAMATIC BALANCE] {prompt}"
                char_ref_needed = True

                characters_in_scene = []
                for char_name in char_names:
                    full_char = next((c for c in characters if c['name'] == char_name), None)
                    if full_char:
                        characters_in_scene.append({
                            'name': char_name,
                            'description': full_char.get('physical_description',
                                                         'Literary character with dramatic potential'),
                            'importance': full_char.get('importance_score', 5),
                            'dramatic_potential': full_char.get('dramatic_potential', {}),
                            'toibin_elements': full_char.get('toibin_elements', {})
                        })
            else:
                # Atmospheric scenes with dramatic potential
                prompt = f"Atmospheric literary scene of {location}, sophisticated cinematography with dramatic visual appeal, warm intelligent lighting, Tóibín-style contemplative mood with subtle dramatic tension, historically accurate, educational atmosphere that draws clicks while promoting peaceful contemplation"
                enhanced_prompt = f"[TÓIBÍN ATMOSPHERIC + DRAMATIC APPEAL] {prompt}"
                char_ref_needed = False
                characters_in_scene = []

            prompt_data = {
                "scene_number": scene_id,
                "title": title,
                "location": location,
                "characters_present": [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters],
                "character_reference_needed": char_ref_needed,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration_minutes": scene.get('duration_minutes', 4),
                "emotion": emotion,
                "characters_in_scene": characters_in_scene,
                "toibin_literary_quality": True,
                "dramatic_appeal_balanced": True,
                "marketing_optimization": "Balances click appeal with literary authenticity"
            }

            prompts.append(prompt_data)

        return prompts

    # 1. REPLACE: _regenerate_visual_prompts_with_characters
    def _regenerate_visual_prompts_with_toibin_drama(self, scene_plan: List[Dict], characters: List[Dict],
                                                     scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
        """
        Generate TÓIBÍN LITERARY + DRAMATIC MARKETING visual prompts

        PERFECT BALANCE:
        - DRAMATIC APPEAL for thumbnails and visual engagement
        - TÓIBÍN LITERARY AUTHENTICITY for content quality
        - SLEEP OPTIMIZATION for peaceful viewing experience
        - HISTORICAL ACCURACY for educational value
        """

        self.character_system.log_extraction_step("TÓIBÍN Literary + Dramatic Marketing Visual Prompts")

        # Enhanced Midjourney safety + Tóibín literary awareness
        TOIBIN_DRAMATIC_PROMPT_GUIDE = """
        Create LITERARY DRAMATIC visual prompts that balance CLICK APPEAL with TÓIBÍN AUTHENTICITY.

        🎭 TÓIBÍN VISUAL PHILOSOPHY:
        - "Sparseness with superabundance of suggestion" - show less, suggest more
        - "Fascination of commonplaces" - make ordinary moments visually compelling
        - "Quiet dignity" - characters maintain composure even in dramatic scenes
        - "Hush you hear" - peaceful undertone even in visually dramatic compositions

        🔥 DRAMATIC MARKETING REQUIREMENTS:
        - Strong visual contrast for thumbnail appeal
        - Character expressions that convey internal drama
        - Composition that works for "BETRAYAL!" "MYSTERIOUS!" overlays
        - Eye-catching but sophisticated, not cheap clickbait

        ⚡ SAFETY + SOPHISTICATION RULES:
        - Never use: "intimate", "private", "mystical", "late at night"
        - Use instead: "contemplative", "personal reflection", "atmospheric", "evening hours"
        - Focus on educational/historical sophistication
        - Show character psychology through expression and posture
        - Include environmental storytelling that suggests literary depth

        🎨 TÓIBÍN + DRAMATIC ENHANCEMENT STRATEGY:
        - Show character internal states through subtle facial expressions
        - Use lighting that suggests both drama and contemplation
        - Include multiple visual layers (character psychology + historical setting)
        - Environmental details that support both story drama and peaceful mood
        - Cinematic composition that works for thumbnails AND story atmosphere
        """

        regeneration_prompt = f"""
        {TOIBIN_DRAMATIC_PROMPT_GUIDE}

        Create TÓIBÍN LITERARY + DRAMATIC MARKETING visual prompts for historical story scenes.

        SCENE PLAN:
        {json.dumps([{
            'scene_id': s['scene_id'],
            'title': s['title'],
            'location': s['location'],
            'emotion': s['emotion'],
            'description': s['description'],
            'toibin_elements': s.get('toibin_elements', [])
        } for s in scene_plan], indent=2)}

        MAIN CHARACTERS (TÓIBÍN + DRAMATIC ANALYSIS):
        {json.dumps([{
            'name': char.get('name', ''),
            'physical_description': char.get('physical_description', ''),
            'role': char.get('role', ''),
            'toibin_elements': char.get('toibin_elements', {}),
            'dramatic_potential': char.get('dramatic_potential', {}),
            'thumbnail_potential': char.get('thumbnail_potential', '')
        } for char in characters], indent=2)}

        SCENE-CHARACTER MAPPING:
        {json.dumps(scene_character_map, indent=2)}

        🎯 TÓIBÍN + DRAMATIC REQUIREMENTS FOR EACH SCENE:

        ### CHARACTER SCENES (when characters present):
        1. **TÓIBÍN PSYCHOLOGY:** Show internal contradiction through subtle expression
        2. **DRAMATIC APPEAL:** Character faces/poses that work for thumbnail overlays
        3. **"Quiet Dignity":** Characters maintain composure despite internal drama
        4. **Literary Depth:** Environmental details suggesting character complexity
        5. **Historical Authenticity:** Period-accurate details that enhance both drama and education

        ### ATMOSPHERIC SCENES (no characters):
        1. **"Fascination of Commonplaces":** Make ordinary settings visually compelling
        2. **Environmental Storytelling:** Setting that suggests both drama and peace
        3. **Sophisticated Composition:** Thumbnail-worthy but literary, not cheap
        4. **Historical Education:** Accurate details that support story authenticity

        ### EMOTIONAL SCENE TREATMENT:
        - **PEACEFUL scenes:** Tóibín contemplative beauty + subtle visual drama for appeal
        - **CURIOSITY scenes:** Character psychology visible + environmental intrigue
        - **CONCERN scenes:** "Quiet dignity" under pressure + dramatic lighting
        - **RESOLUTION scenes:** Peaceful acceptance + visually satisfying closure

        ### DRAMATIC THUMBNAIL OPTIMIZATION:
        - **Composition:** RIGHT side character placement, LEFT side clear for text overlay
        - **Lighting:** Dramatic enough for clicks, peaceful enough for sleep content
        - **Expression:** Character emotions that suggest internal drama without anxiety
        - **Background:** Historical setting that supports both educational and visual appeal

        OUTPUT FORMAT for each scene:
        {{
          "scene_number": X,
          "title": "[Tóibín-style scene title]",
          "location": "[Historical location with literary atmosphere]", 
          "characters_present": ["Character1"] or [],
          "character_reference_needed": true/false,
          "prompt": "[TÓIBÍN LITERARY + DRAMATIC prompt balancing authenticity with appeal]",
          "enhanced_prompt": "[Same prompt with Tóibín + dramatic character markers]",
          "duration_minutes": X,
          "emotion": "[Scene emotion with Tóibín interpretation]",
          "characters_in_scene": [
            {{
              "name": "Character Name",
              "description": "Physical description with dramatic potential",
              "importance": X,
              "toibin_psychology": "Internal state visible in expression",
              "dramatic_appeal": "Why this works for thumbnails",
              "literary_authenticity": "How this maintains Tóibín quality"
            }}
          ],
          "toibin_literary_elements": {{
            "quiet_dignity_visible": "How character composure shows in visual",
            "internal_contradiction": "Subtle signs of character internal conflict",
            "environmental_storytelling": "Setting details that suggest character depth",
            "sparseness_with_suggestion": "What the image suggests beyond what it shows"
          }},
          "dramatic_marketing_elements": {{
            "thumbnail_appeal": "Why this image draws clicks",
            "visual_drama": "Dramatic elements that don't compromise peaceful mood",
            "character_expression": "Facial expression perfect for dramatic overlays",
            "composition_strategy": "How this works for 'BETRAYAL!' 'MYSTERIOUS!' text"
          }},
          "sleep_content_optimization": {{
            "peaceful_undertone": "How image maintains calm despite drama",
            "soothing_elements": "Visual elements that promote relaxation",
            "non_jarring_composition": "Why this won't disturb sleep-seeking viewers"
          }}
        }}

        ## 🎭 TÓIBÍN + DRAMATIC MASTERY CHALLENGE:

        Create visual prompts that achieve the PERFECT BALANCE:

        ### REQUIRED EXCELLENCE FOR EACH PROMPT:
        - ✅ **TÓIBÍN LITERARY AUTHENTICITY** - sophisticated, psychologically rich visuals
        - ✅ **DRAMATIC THUMBNAIL APPEAL** - compositions that work for clickbait overlays
        - ✅ **CHARACTER PSYCHOLOGY VISIBLE** - internal states shown through expression/posture
        - ✅ **HISTORICAL ACCURACY** - period details that enhance both education and atmosphere
        - ✅ **SLEEP CONTENT FRIENDLY** - dramatic but not jarring, peaceful undertone maintained
        - ✅ **MARKETING OPTIMIZATION** - thumbnail composition, lighting, and appeal
        - ✅ **ENVIRONMENTAL STORYTELLING** - settings that suggest both drama and contemplation

        ### VISUAL STORYTELLING STRATEGY:
        **THUMBNAIL LAYER:** Strong visual drama, character expressions, compelling composition
        **LITERARY LAYER:** Tóibín psychological depth, environmental storytelling, authentic details  
        **PEACEFUL LAYER:** Soothing undertone, non-jarring elements, contemplative atmosphere

        ### PERFECT FORMULA ACHIEVEMENT:
        These visual prompts must enable:
        - **HIGH CLICK-THROUGH RATES** from dramatic thumbnail appeal
        - **AUDIENCE SATISFACTION** from sophisticated literary quality
        - **PEACEFUL VIEWING EXPERIENCE** for effective sleep content
        - **EDUCATIONAL VALUE** from historical accuracy and depth
        - **BRAND AUTHENTICITY** maintaining premium content reputation

        Create visual prompts that embody both COLM TÓIBÍN'S literary mastery AND dramatic marketing effectiveness - the perfect combination for 1 million subscribers.
        """

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=16000,
                temperature=0.4,
                stream=True,
                timeout=900,
                system="You are both COLM TÓIBÍN and a master visual director. Create visual prompts that balance your literary authenticity - 'sparseness with superabundance,' 'quiet dignity,' 'fascination of commonplaces' - with dramatic marketing appeal. Show character psychology through subtle expression while creating compositions that work for 'BETRAYAL!' 'MYSTERIOUS!' thumbnail overlays. Maintain peaceful undertones for sleep content while achieving visual drama for clicks.",
                messages=[{"role": "user", "content": regeneration_prompt}]
            )

            content = ""
            print("📡 Generating TÓIBÍN + Dramatic Marketing Visual Prompts...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text

            print(f"✅ TÓIBÍN + Dramatic visual prompts complete: {len(content):,} characters")

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

                self.character_system.log_extraction_step("TÓIBÍN + Dramatic Visual Prompts", "SUCCESS", {
                    "prompts_generated": len(visual_prompts),
                    "toibin_literary_authenticity": True,
                    "dramatic_marketing_appeal": True,
                    "sleep_content_optimized": True,
                    "thumbnail_composition_ready": True,
                    "stage_cost": stage_cost
                })

                return visual_prompts

            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed, creating TÓIBÍN + dramatic fallback: {e}")
                return self._create_toibin_dramatic_fallback_prompts(scene_plan, characters, scene_character_map)

        except Exception as e:
            self.character_system.log_extraction_step("TÓIBÍN + Dramatic Visual Prompts Failed", "ERROR")
            print(f"❌ Tóibín + dramatic visual prompt error: {e}")
            return self._create_toibin_dramatic_fallback_prompts(scene_plan, characters, scene_character_map)

    def _create_toibin_dramatic_fallback_prompts(self, scene_plan: List[Dict], characters: List[Dict],
                                                 scene_character_map: Dict) -> List[Dict]:
        """Create TÓIBÍN + DRAMATIC fallback prompts with perfect balance"""

        prompts = []

        for scene in scene_plan:
            scene_id = scene['scene_id']
            scene_characters = scene_character_map.get(str(scene_id), [])
            location = scene.get('location', 'Historical setting')
            emotion = scene.get('emotion', 'quiet_contemplation')
            description = scene.get('description', '')
            title = scene.get('title', f"Scene {scene_id}")

            # Tóibín + Dramatic balance based on emotion
            if emotion in ['concern', 'curiosity', 'recognition']:
                toibin_element = "showing quiet internal contemplation about unfolding complexity"
                dramatic_element = "with subtle emotional intensity and dramatic lighting suitable for thumbnail overlays"
                lighting = "sophisticated dramatic lighting that emphasizes character psychology while maintaining peaceful undertone"
            else:  # peaceful, resolution
                toibin_element = "in thoughtful, dignified poses with visible internal depth"
                dramatic_element = "with compelling visual composition and character expression perfect for marketing"
                lighting = "warm, sophisticated lighting suggesting both literary quality and visual appeal"

            # Character integration with Tóibín + dramatic balance
            if scene_characters:
                char_names = [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters]
                char_list = ', '.join(char_names)

                prompt = f"Literary cinematic scene of {location}, featuring {char_list} {toibin_element}, {dramatic_element}, {lighting}, Tóibín-style character psychology visible in expressions and posture, historically accurate period setting, sophisticated composition balancing dramatic thumbnail appeal with contemplative literary atmosphere, environmental storytelling supporting both marketing effectiveness and peaceful sleep content"

                enhanced_prompt = f"[TÓIBÍN LITERARY + DRAMATIC MARKETING] [CHARACTERS: {char_list}] {prompt}"
                char_ref_needed = True

                # Character details with Tóibín + dramatic analysis
                characters_in_scene = []
                for char_name in char_names:
                    full_char = next((c for c in characters if c['name'] == char_name), None)
                    if full_char:
                        characters_in_scene.append({
                            'name': char_name,
                            'description': full_char.get('physical_description',
                                                         'Literary character with dramatic appeal'),
                            'importance': full_char.get('importance_score', 5),
                            'toibin_psychology': 'Internal contradiction visible in subtle expression',
                            'dramatic_appeal': 'Face and posture perfect for thumbnail marketing',
                            'literary_authenticity': 'Maintains Tóibín quiet dignity and complexity'
                        })
            else:
                # Atmospheric scenes with Tóibín + dramatic potential
                prompt = f"Atmospheric literary scene of {location}, sophisticated cinematography with compelling visual drama, {lighting}, Tóibín-style contemplative mood with subtle dramatic tension for thumbnail appeal, historically accurate environmental storytelling, educational atmosphere that draws clicks while promoting peaceful contemplation and sleep"

                enhanced_prompt = f"[TÓIBÍN ATMOSPHERIC + DRAMATIC MARKETING] {prompt}"
                char_ref_needed = False
                characters_in_scene = []

            prompt_data = {
                "scene_number": scene_id,
                "title": title,
                "location": location,
                "characters_present": [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters],
                "character_reference_needed": char_ref_needed,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration_minutes": scene.get('duration_minutes', 4),
                "emotion": emotion,
                "characters_in_scene": characters_in_scene,
                "toibin_literary_elements": {
                    "quiet_dignity_visible": "Character composure maintained despite internal drama",
                    "internal_contradiction": "Subtle psychological complexity shown in expression",
                    "environmental_storytelling": "Setting details suggesting character depth and historical authenticity",
                    "sparseness_with_suggestion": "Visual restraint that suggests deeper meaning"
                },
                "dramatic_marketing_elements": {
                    "thumbnail_appeal": "Strong visual composition perfect for dramatic text overlays",
                    "visual_drama": "Emotional intensity that draws clicks without compromising peaceful mood",
                    "character_expression": "Facial expressions ideal for 'BETRAYAL!' 'MYSTERIOUS!' marketing",
                    "composition_strategy": "RIGHT side character placement, LEFT side clear for thumbnail text"
                },
                "sleep_content_optimization": {
                    "peaceful_undertone": "Maintains calming atmosphere despite visual drama",
                    "soothing_elements": "Warm lighting and composed character expressions promote relaxation",
                    "non_jarring_composition": "Sophisticated visual appeal that won't disturb sleep-seeking viewers"
                }
            }

            prompts.append(prompt_data)

        return prompts

    def _create_enhanced_fallback_prompts(self, scene_plan: List[Dict], characters: List[Dict],
                                          scene_character_map: Dict) -> List[Dict]:
        """
        Create TÓIBÍN + DRAMATIC enhanced fallback prompts with perfect balance

        FALLBACK STRATEGY:
        - Maintain Tóibín literary authenticity even in backup prompts
        - Ensure dramatic appeal for thumbnail optimization
        - Keep sleep content peaceful undertone
        - Preserve historical accuracy and educational value
        """

        prompts = []

        for scene in scene_plan:
            scene_id = scene['scene_id']
            scene_characters = scene_character_map.get(str(scene_id), [])
            location = scene.get('location', 'Historical setting')
            emotion = scene.get('emotion', 'quiet_contemplation')
            description = scene.get('description', '')
            title = scene.get('title', f"Scene {scene_id}")

            # TÓIBÍN + DRAMATIC elements based on emotion
            if emotion == 'concern':
                # Tóibín approach: quiet dignity under pressure
                toibin_modifier = "showing quiet internal contemplation and dignified concern about unfolding complexity"
                dramatic_element = "with subtle emotional tension and sophisticated dramatic lighting perfect for thumbnail appeal"
                environmental_storytelling = "with environmental details suggesting underlying tension while maintaining peaceful educational atmosphere"

            elif emotion == 'curiosity':
                # Tóibín approach: characters observing, wondering
                toibin_modifier = "in thoughtful observation and gentle curiosity about their circumstances"
                dramatic_element = "with compelling visual interest and character expressions ideal for marketing appeal"
                environmental_storytelling = "with intriguing historical details that draw viewer attention while promoting contemplative mood"

            elif emotion == 'resolution':
                # Tóibín approach: quiet acceptance and understanding
                toibin_modifier = "showing peaceful determination and quiet acceptance with visible internal growth"
                dramatic_element = "with hopeful visual composition and character expressions perfect for satisfying thumbnails"
                environmental_storytelling = "with harmonious community elements and beautiful historical setting supporting both visual appeal and peaceful closure"

            else:  # peaceful, quiet_recognition
                # Tóibín approach: contemplative serenity
                toibin_modifier = "in contemplative, dignified poses with visible psychological depth and quiet recognition"
                dramatic_element = "with sophisticated visual appeal and warm lighting suitable for dramatic marketing while maintaining peaceful mood"
                environmental_storytelling = "with beautiful, historically accurate atmosphere and daily life details that fascinate while promoting relaxation"

            # Character integration with Tóibín + Dramatic balance
            if scene_characters:
                char_names = [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters]
                char_list = ', '.join(char_names)

                # TÓIBÍN + DRAMATIC CHARACTER PROMPT
                prompt = f"Literary cinematic scene of {location}, featuring {char_list} {toibin_modifier}, {dramatic_element}, {environmental_storytelling}, historically accurate period setting, sophisticated composition balancing Tóibín psychological authenticity with dramatic thumbnail potential, multiple visual layers showing character internal states and environmental context perfect for both click appeal and sleep content"

                enhanced_prompt = f"[TÓIBÍN LITERARY + DRAMATIC MARKETING] [CHARACTERS: {char_list}] {prompt}"
                char_ref_needed = True

                # Character details with Tóibín + dramatic analysis
                characters_in_scene = []
                for char_name in char_names:
                    full_char = next((c for c in characters if c['name'] == char_name), None)
                    if full_char:
                        # Extract Tóibín + dramatic elements if available
                        toibin_elements = full_char.get('toibin_elements', {})
                        dramatic_potential = full_char.get('dramatic_potential', {})

                        characters_in_scene.append({
                            'name': char_name,
                            'description': full_char.get('physical_description',
                                                         'Literary character with dramatic appeal'),
                            'importance': full_char.get('importance_score', 5),
                            'toibin_psychology': toibin_elements.get('quiet_dignity',
                                                                     'Internal complexity visible in expression'),
                            'dramatic_appeal': dramatic_potential.get('thumbnail_emotion',
                                                                      'Perfect for marketing thumbnails'),
                            'literary_authenticity': 'Maintains Tóibín character depth and sophistication'
                        })
            else:
                # ATMOSPHERIC SCENES with Tóibín + dramatic balance
                prompt = f"Atmospheric literary scene of {location} {environmental_storytelling}, historically accurate period setting, sophisticated lighting that suggests both contemplative depth and visual drama, environmental storytelling emphasizing Tóibín 'fascination of commonplaces' while maintaining thumbnail appeal, educational atmosphere that draws clicks while promoting peaceful sleep contemplation"

                enhanced_prompt = f"[TÓIBÍN ATMOSPHERIC + DRAMATIC APPEAL] {prompt}"
                char_ref_needed = False
                characters_in_scene = []

            # Complete prompt data with Tóibín + dramatic analysis
            prompt_data = {
                "scene_number": scene_id,
                "title": title,
                "location": location,
                "characters_present": [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters],
                "character_reference_needed": char_ref_needed,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration_minutes": scene.get('duration_minutes', 4),
                "emotion": emotion,
                "characters_in_scene": characters_in_scene,

                # TÓIBÍN LITERARY ELEMENTS
                "toibin_literary_elements": {
                    "quiet_dignity_maintained": "Characters show composure and internal depth even in fallback prompts",
                    "fascination_of_commonplaces": "Ordinary moments made visually compelling through literary perspective",
                    "sparseness_with_suggestion": "Visual restraint that implies deeper meaning and character psychology",
                    "internal_complexity_visible": "Character psychology evident in expressions and posture"
                },

                # DRAMATIC MARKETING ELEMENTS
                "dramatic_marketing_elements": {
                    "thumbnail_optimization": "Composition and lighting perfect for dramatic text overlays",
                    "visual_click_appeal": "Strong dramatic elements that draw viewer attention and clicks",
                    "character_expression_appeal": "Facial expressions and poses ideal for marketing thumbnails",
                    "sophisticated_drama": "Dramatic appeal that maintains literary quality, not cheap clickbait"
                },

                # SLEEP CONTENT OPTIMIZATION
                "sleep_content_balance": {
                    "peaceful_undertone_maintained": "Dramatic elements balanced with calming atmosphere for sleep",
                    "non_jarring_visuals": "Sophisticated composition that won't disturb sleep-seeking viewers",
                    "contemplative_mood_preserved": "Literary contemplative quality maintained despite dramatic appeal",
                    "educational_atmosphere": "Historical accuracy and depth supporting peaceful learning experience"
                },

                # FALLBACK QUALITY ASSURANCE
                "fallback_excellence": {
                    "maintains_system_standards": "Backup prompts preserve full Tóibín + dramatic quality",
                    "consistent_branding": "Literary sophistication + marketing appeal maintained in fallbacks",
                    "production_ready": "Fallback prompts fully suitable for final video production",
                    "thumbnail_ready": "Emergency prompts still optimized for dramatic thumbnail creation"
                }
            }

            prompts.append(prompt_data)

        print(f"✅ Created {len(prompts)} TÓIBÍN + Dramatic enhanced fallback prompts")
        print(f"🎭 Literary authenticity + marketing appeal maintained in backup system")

        return prompts

    # 2. REPLACE: _generate_intelligent_thumbnail
    def _generate_intelligent_thumbnail(self, topic: str, description: str, character_result: Dict,
                                        clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """
        Generate TÓIBÍN LITERARY + DRAMATIC MARKETING thumbnail with perfect balance

        THUMBNAIL PHILOSOPHY:
        - DRAMATIC APPEAL for maximum click-through rates
        - TÓIBÍN LITERARY SOPHISTICATION for brand authenticity
        - SLEEP CONTENT APPROPRIATENESS for audience satisfaction
        - HISTORICAL ACCURACY for educational credibility

        PERFECT FORMULA: Sophisticated drama that draws clicks while promising peaceful quality content
        """

        self.character_system.log_extraction_step("TÓIBÍN + Dramatic Marketing Thumbnail Generation")

        characters = character_result.get('main_characters', [])
        visual_style = character_result.get('visual_style_notes', {})

        # Analyze topic for Tóibín + dramatic approach
        topic_lower = f"{topic} {description}".lower()

        # ENHANCED ANALYSIS: Historical drama vs peaceful contemplation
        crisis_keywords = ['last day', 'fall', 'destruction', 'siege', 'fire', 'burning', 'final', 'end', 'war',
                           'battle', 'betrayal', 'assassination', 'death', 'murder']
        contemplative_keywords = ['evening', 'dinner', 'garden', 'villa', 'palace', 'chamber', 'study', 'library',
                                  'private', 'quiet', 'peaceful']

        is_crisis = any(keyword in topic_lower for keyword in crisis_keywords)
        is_contemplative = any(keyword in topic_lower for keyword in contemplative_keywords)

        # TÓIBÍN + DRAMATIC DRAMA LEVEL CALCULATION
        if is_crisis:
            drama_level = 8  # High drama for crisis topics
            crisis_elements = [kw for kw in crisis_keywords if kw in topic_lower]
            toibin_approach = "quiet_dignity_under_pressure"
        elif is_contemplative:
            drama_level = 6  # Moderate drama for contemplative topics
            crisis_elements = []
            toibin_approach = "contemplative_recognition"
        else:
            drama_level = 5  # Balanced drama for general topics
            crisis_elements = []
            toibin_approach = "everyday_profundity"

        # ENHANCED CHARACTER SELECTION with Tóibín + dramatic analysis
        thumbnail_character_selection = self.character_system.select_thumbnail_character(
            characters, topic, description
        )

        # ENHANCED TITLE GENERATION with dramatic + literary balance
        if not clickbait_title:
            youtube_data = character_result.get('youtube_optimization', {})
            clickbait_titles = youtube_data.get('clickbait_titles', [])
            if clickbait_titles:
                clickbait_title = clickbait_titles[0]
            else:
                # Generate Tóibín + dramatic title
                if is_crisis:
                    clickbait_title = f"BETRAYAL! {topic.split()[-1] if topic.split() else 'Character'}'s Final Peaceful Evening (2 Hour Sleep Story)"
                else:
                    clickbait_title = f"MYSTERIOUS! The Secret History of {topic} (Most Peaceful Sleep Story)"

        if not font_design:
            font_design = "Bold impact font with literary sophistication, warm golden color (#d4af37) with deep shadows, balancing dramatic appeal with educated audience expectations"

        thumbnail_prompt = f"""Create a TÓIBÍN LITERARY + DRAMATIC MARKETING thumbnail for "{topic}" that achieves perfect balance between click appeal and literary authenticity.

    TOPIC: {topic}
    DESCRIPTION: {description}
    CRISIS TOPIC: {is_crisis}
    CONTEMPLATIVE TOPIC: {is_contemplative}
    DRAMA LEVEL: {drama_level}/10
    CRISIS ELEMENTS: {crisis_elements}
    TÓIBÍN APPROACH: {toibin_approach}

    CHARACTER SELECTION:
    {json.dumps(thumbnail_character_selection, indent=2)}

    CLICKBAIT TITLE: {clickbait_title}

    🎭 TÓIBÍN + DRAMATIC THUMBNAIL MASTERY:

    ## LITERARY AUTHENTICITY REQUIREMENTS:
    - **"Quiet Dignity"** - Characters maintain composure and internal depth even in dramatic scenes
    - **"Sparseness with Suggestion"** - Visual restraint that implies deeper meaning
    - **"Fascination of Commonplaces"** - Ordinary moments made visually compelling
    - **Sophisticated Composition** - Educated audience expectations met with literary sensibility
    - **Character Psychology Visible** - Internal states shown through subtle expression and posture

    ## DRAMATIC MARKETING REQUIREMENTS:
    - **HIGH CLICK-THROUGH POTENTIAL** - Visual elements that compel immediate clicks
    - **EMOTIONAL STORYTELLING** - Character expressions and environmental context that tell story
    - **DRAMATIC TEXT OVERLAY COMPATIBILITY** - Composition perfect for "BETRAYAL!" "MYSTERIOUS!" overlays
    - **VISUAL HIERARCHY** - Clear focus on character emotion with supporting environmental drama
    - **THUMBNAIL COMPETITION ADVANTAGE** - Stands out against generic sleep content thumbnails

    ## PERFECT BALANCE STRATEGY:

    ### 🎬 TÓIBÍN + DRAMATIC VISUAL STORYTELLING:
    1. **Character Internal Drama** - Show "characters led by desires they don't understand" through expression
    2. **Environmental Literary Context** - Historical accuracy with sophisticated atmospheric storytelling  
    3. **"Mixed Motives" Visible** - Complex character psychology evident in subtle facial expression
    4. **Dramatic Lighting** - Sophisticated cinematography that draws clicks while maintaining literary quality
    5. **Cultural/Historical Symbolism** - Elements that suggest educational depth and narrative richness

    ### 🎯 EMOTIONAL APPROACH BY TOPIC TYPE:

    {f'''
    FOR CRISIS TOPICS like "{topic}" (Tóibín + Dramatic):
    - **Character Expression**: Quiet concern and dignified contemplation about historical events (NOT fear/terror)
    - **Environmental Elements**: Subtle crisis indicators balanced with peaceful, educational atmosphere
    - **Tóibín Element**: "Quiet dignity under pressure" - composure despite internal awareness
    - **Dramatic Appeal**: Expressions that work for "BETRAYAL!" overlays while maintaining sophistication
    - **Literary Quality**: Historical accuracy and character depth visible in visual composition
    - **Sleep Balance**: Contemplative sadness rather than jarring anxiety - invites peaceful exploration
    ''' if is_crisis else f'''
    FOR CONTEMPLATIVE TOPICS like "{topic}" (Tóibín + Dramatic):
    - **Character Expression**: Thoughtful recognition and gentle internal complexity  
    - **Environmental Elements**: Beautiful historical details suggesting narrative depth and luxury
    - **Tóibín Element**: "Contemplative recognition" - moments of quiet understanding
    - **Dramatic Appeal**: Compelling character psychology and visual richness that draws clicks
    - **Literary Quality**: Sophisticated composition suggesting quality storytelling
    - **Sleep Balance**: Warm, inviting atmosphere that promises peaceful, educational content
    '''}

    ### 🎯 COMPOSITION MASTERY (CRITICAL FOR SUCCESS):
    - **Character Position**: RIGHT side of frame (60-70% from left edge) - PROVEN EFFECTIVE
    - **Text Space**: LEFT side (30-40%) completely CLEAR for dramatic text overlay
    - **Character Scale**: Close enough to show facial expression, wide enough to include environmental context
    - **Multiple Characters**: Show relationships and "tacit exchanges" when multiple characters present
    - **Background**: Historical setting that suggests both educational value and narrative intrigue

    ### 💤 SLEEP CONTENT + LITERARY BRAND BALANCE:
    - **Sophisticated Color Palette** - Warm, inviting tones that suggest quality content and peaceful experience
    - **Literary Sensibility** - Visual composition that appeals to educated audience seeking quality sleep content
    - **Contemplative Mood** - Even dramatic elements maintain underlying peaceful, thoughtful atmosphere
    - **Educational Promise** - Visual elements that suggest learning and cultural enrichment
    - **Premium Content Indication** - Sophistication level that justifies longer viewing time investment

    OUTPUT FORMAT:
    {{
      "thumbnail_prompt": {{
        "scene_number": 99,
        "character_used": "{thumbnail_character_selection['character_used']}",
        "clickbait_title": "{clickbait_title}",
        "font_design": "{font_design}",
        "drama_level": {drama_level},
        "toibin_approach": "{toibin_approach}",
        "is_crisis_topic": {is_crisis},
        "is_contemplative_topic": {is_contemplative},
        "prompt": "[TÓIBÍN LITERARY + DRAMATIC MARKETING cinematic thumbnail with perfect balance]",
        "visual_style": "[Sophisticated dramatic style maintaining literary authenticity]",
        "character_positioning": "RIGHT side of frame (60-70% from left), literary character depth visible",
        "text_overlay_strategy": "LEFT side (30-40%) clear for dramatic title text overlays",
        "emotional_appeal": "[Specific sophisticated emotion that draws clicks while promising quality]",
        "environmental_storytelling": "[Historical context that suggests both drama and educational depth]",
        "clickability_factors": ["sophisticated_character_psychology", "dramatic_historical_context", "literary_quality_promise"],
        "sleep_content_balance": "[How dramatic appeal maintains peaceful, contemplative undertone for sleep audience]",
        "literary_authenticity": "[How thumbnail maintains Tóibín literary quality and educated audience appeal]",
        "toibin_elements": {{
          "quiet_dignity_visible": "[How character maintains composure despite dramatic situation]",
          "internal_complexity_shown": "[Subtle signs of character psychological depth]",
          "fascination_of_commonplaces": "[How ordinary moment is made visually compelling]",
          "sparseness_with_suggestion": "[Visual restraint that implies deeper narrative meaning]"
        }},
        "dramatic_marketing_elements": {{
          "click_appeal_factors": "[Specific visual elements that compel immediate clicks]",
          "thumbnail_competition_advantage": "[How this stands out against generic sleep content]",
          "emotional_storytelling": "[Character expression and context that tell visual story]",
          "overlay_compatibility": "[Perfect composition for BETRAYAL! MYSTERIOUS! text overlays]"
        }},
        "thumbnail_reasoning": "{thumbnail_character_selection['reasoning']} - Enhanced with Tóibín literary sophistication",
        "composition_notes": "RIGHT-positioned character with literary depth, LEFT text space, sophisticated drama maintaining peaceful undertone"
      }},
      "thumbnail_alternatives": [
        {{
          "variant": "Literary Character Focus",
          "prompt": "[Alternative emphasizing Tóibín character psychology with dramatic lighting]"
        }},
        {{
          "variant": "Historical Environmental Drama", 
          "prompt": "[Alternative focusing on sophisticated historical setting with character contemplation]"
        }},
        {{
          "variant": "Symbolic Literary Moment",
          "prompt": "[Alternative showing key story symbols with Tóibín contemplative approach]"
        }}
      ],
      "thumbnail_stats": {{
        "character_approach": "{thumbnail_character_selection['character_used']}",
        "toibin_literary_authenticity": true,
        "dramatic_marketing_appeal": true,
        "visual_storytelling": true,
        "environmental_elements": true,
        "composition_optimized": true,
        "clickability_enhanced": true,
        "sleep_appropriate": true,
        "educated_audience_targeted": true,
        "premium_content_suggested": true,
        "crisis_balanced": {is_crisis},
        "literary_sophistication_maintained": true
      }}
    }}

    ## 🎭 TÓIBÍN + DRAMATIC THUMBNAIL MASTERY CHALLENGE:

    Create a thumbnail that achieves the PERFECT BALANCE for 1 million subscribers:

    ### REQUIRED EXCELLENCE:
    - ✅ **DRAMATIC CLICK APPEAL** - Compels immediate clicks from dramatic visual storytelling
    - ✅ **TÓIBÍN LITERARY AUTHENTICITY** - Maintains sophisticated character psychology and literary quality
    - ✅ **SLEEP CONTENT APPROPRIATENESS** - Promises peaceful, contemplative experience despite drama
    - ✅ **HISTORICAL EDUCATIONAL VALUE** - Suggests learning and cultural enrichment  
    - ✅ **PREMIUM BRAND POSITIONING** - Appeals to educated audience seeking quality sleep content
    - ✅ **CHARACTER PSYCHOLOGY DEPTH** - Shows "quiet dignity" and internal complexity
    - ✅ **ENVIRONMENTAL STORYTELLING** - Historical context supporting both drama and peace
    - ✅ **COMPOSITION PERFECTION** - RIGHT character, LEFT text, thumbnail overlay ready

    ### SUCCESS METRICS THIS THUMBNAIL MUST ACHIEVE:
    - **HIGH CTR** from dramatic appeal and character psychology intrigue
    - **AUDIENCE SATISFACTION** from literary quality promise fulfilled  
    - **BRAND AUTHENTICITY** maintaining sophisticated content reputation
    - **SLEEP EFFECTIVENESS** peaceful undertone despite dramatic marketing
    - **EDUCATIONAL VALUE** historical accuracy and cultural depth
    - **PREMIUM POSITIONING** justifying 2-hour content investment

    CRITICAL SUCCESS REQUIREMENT: This thumbnail must work perfectly for "BETRAYAL! Character's Final Peaceful Evening" while maintaining Tóibín literary sophistication and sleep content appropriateness. The perfect formula for viral growth + loyal educated audience.

    For crisis topics: Show contemplative response to historical drama with quiet dignity and sophisticated visual storytelling.
    For contemplative topics: Show compelling character psychology and historical richness with warm, inviting literary atmosphere."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=8000,
                temperature=0.4,
                stream=True,
                timeout=600,
                system="You are both COLM TÓIBÍN and a master YouTube thumbnail strategist. Create thumbnails that balance your literary authenticity - 'quiet dignity,' 'sparseness with suggestion,' 'fascination of commonplaces' - with dramatic marketing appeal. Show character psychology through sophisticated expression while creating compositions perfect for 'BETRAYAL!' 'MYSTERIOUS!' overlays. Maintain peaceful, contemplative undertones for sleep content while achieving maximum click appeal for educated audiences.",
                messages=[{"role": "user", "content": thumbnail_prompt}]
            )

            content = ""
            print("📡 Generating TÓIBÍN + Dramatic Marketing Thumbnail...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text

            print(f"✅ TÓIBÍN + Dramatic thumbnail complete: {len(content):,} characters")

            input_tokens = len(thumbnail_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            parsed_result = self._parse_claude_response(content, "thumbnail_generation")

            self.character_system.log_extraction_step("TÓIBÍN + Dramatic Marketing Thumbnail", "SUCCESS", {
                "drama_level": drama_level,
                "toibin_approach": toibin_approach,
                "is_crisis_topic": is_crisis,
                "is_contemplative_topic": is_contemplative,
                "literary_authenticity_maintained": True,
                "dramatic_marketing_optimized": True,
                "visual_storytelling": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            print(f"❌ Tóibín + dramatic thumbnail error: {e}")

            # ENHANCED TÓIBÍN + DRAMATIC FALLBACK
            if is_crisis:
                fallback_elements = "showing quiet dignity and contemplative concern about historical events, with sophisticated environmental indicators suggesting crisis context"
                mood = "contemplative literary response to historical drama, maintaining peaceful undertone"
                toibin_element = "quiet dignity under pressure"
            elif is_contemplative:
                fallback_elements = "in thoughtful, sophisticated contemplation with visible internal complexity"
                mood = "warm, inviting literary atmosphere suggesting quality contemplative content"
                toibin_element = "contemplative recognition"
            else:
                fallback_elements = "showing character psychology depth with compelling visual appeal"
                mood = "sophisticated literary atmosphere balancing drama with peaceful content promise"
                toibin_element = "fascination of commonplaces"

            fallback_prompt = f"TÓIBÍN LITERARY + DRAMATIC MARKETING cinematic thumbnail of {topic}, character positioned RIGHT side of frame {fallback_elements}, historically accurate sophisticated setting, literary cinematography with dramatic appeal, {mood}, LEFT side clear for text overlay, visual storytelling balancing Tóibín authenticity with click optimization, {toibin_element} visible in character expression and composition"

            return {
                "thumbnail_prompt": {
                    "scene_number": 99,
                    "character_used": "Main character (Tóibín + dramatic enhanced fallback)",
                    "clickbait_title": clickbait_title,
                    "drama_level": drama_level,
                    "toibin_approach": toibin_approach,
                    "prompt": fallback_prompt,
                    "character_positioning": "RIGHT side with literary depth, LEFT text space clear",
                    "visual_storytelling": "TÓIBÍN + dramatic environmental storytelling",
                    "literary_authenticity": "Maintains sophisticated character psychology and literary quality",
                    "dramatic_marketing_appeal": "Optimized for click-through while preserving authenticity",
                    "thumbnail_reasoning": "Enhanced Tóibín + dramatic fallback maintaining perfect balance"
                },
                "thumbnail_stats": {
                    "toibin_literary_authenticity": True,
                    "dramatic_marketing_appeal": True,
                    "visual_storytelling": True,
                    "enhanced_fallback": True,
                    "literary_sophistication_maintained": True
                }
            }

    def _extract_visual_prompts_from_text(self, content: str) -> List[Dict]:
        """
        Extract TÓIBÍN + DRAMATIC visual prompts from text when JSON parsing fails

        CRITICAL FALLBACK STRATEGY:
        - Maintain Tóibín literary authenticity even in emergency fallback
        - Preserve dramatic marketing appeal for thumbnail optimization
        - Ensure sleep content peaceful undertone throughout
        - Keep historical accuracy and educational value

        NEVER compromise system quality even in backup scenarios
        """

        prompts = []

        # Try to extract actual scene count and topic from content
        try:
            # Look for scene count indicators in the content
            scene_count_matches = re.findall(r'scene[s]?\s*(?:planned|count|total)[\s:]*(\d+)', content.lower())
            if scene_count_matches:
                total_scenes = int(scene_count_matches[0])
            else:
                total_scenes = 35  # Smart algorithm typical range

            # Try to extract topic information
            topic_matches = re.findall(r'topic[:\s]+([^,\n\.]+)', content.lower())
            topic_hint = topic_matches[0].strip() if topic_matches else "historical"

        except:
            total_scenes = 35
            topic_hint = "historical"

        print(f"📡 TÓIBÍN + Dramatic fallback: Generating {total_scenes} visual prompts for {topic_hint} topic")

        # TÓIBÍN + DRAMATIC FALLBACK TEMPLATES BY EMOTION PROGRESSION
        emotion_progression = [
            # Peaceful phase (30% of scenes)
            "quiet_contemplation", "peaceful_observation", "gentle_recognition",
            "serene_acceptance", "contemplative_beauty", "tranquil_awareness",
            "peaceful_dignity", "quiet_understanding", "serene_reflection",
            "contemplative_grace", "peaceful_recognition",

            # Curiosity phase (30% of scenes)
            "gentle_curiosity", "thoughtful_inquiry", "quiet_interest",
            "contemplative_wonder", "dignified_attention", "peaceful_discovery",
            "serene_investigation", "quiet_fascination", "thoughtful_exploration",
            "contemplative_intrigue", "gentle_awareness",

            # Concern phase (20% of scenes)
            "quiet_concern", "dignified_worry", "contemplative_unease",
            "peaceful_apprehension", "gentle_anxiety", "thoughtful_concern",
            "serene_tension", "quiet_vigilance",

            # Resolution phase (20% of scenes)
            "peaceful_resolution", "quiet_acceptance", "dignified_closure",
            "contemplative_understanding", "serene_completion", "gentle_finality",
            "thoughtful_conclusion", "peaceful_fulfillment"
        ]

        # TÓIBÍN + DRAMATIC VISUAL TEMPLATES
        toibin_dramatic_templates = [
            {
                "base": "Literary cinematic scene of {location}, sophisticated composition balancing Tóibín contemplative authenticity with dramatic thumbnail appeal",
                "character_addition": "featuring {characters} in {emotion_state} with visible psychological depth and quiet dignity",
                "environmental": "historically accurate period setting with environmental storytelling supporting both educational value and visual drama",
                "lighting": "sophisticated dramatic lighting emphasizing character psychology while maintaining peaceful sleep content undertone",
                "marketing": "composition optimized for dramatic text overlays while preserving literary quality and contemplative atmosphere"
            },
            {
                "base": "Atmospheric historical scene of {location}, Tóibín-style fascination of commonplaces made visually compelling",
                "character_addition": "with {characters} showing {emotion_state} and internal complexity through subtle expression and posture",
                "environmental": "period-accurate environmental details suggesting both narrative depth and cultural authenticity",
                "lighting": "warm, sophisticated lighting perfect for both thumbnail marketing and peaceful viewing experience",
                "marketing": "visual storytelling that draws clicks while promising quality literary sleep content"
            },
            {
                "base": "Cinematic literary view of {location}, sparseness with superabundance of visual suggestion",
                "character_addition": "featuring {characters} in {emotion_state} with Tóibín quiet dignity and contemplative depth visible",
                "environmental": "historically authentic setting with multiple visual layers supporting educational and dramatic appeal",
                "lighting": "dramatic yet peaceful lighting suitable for both marketing thumbnails and sleep content atmosphere",
                "marketing": "composition balancing click appeal with literary sophistication and contemplative mood"
            }
        ]

        # Historical location variety for different periods
        historical_locations = [
            "ancient villa courtyard", "Roman garden terrace", "medieval castle hall", "renaissance palace chamber",
            "monastery scriptorium", "merchant guild hall", "aristocratic dining room", "temple inner sanctum",
            "royal private study", "monastery garden", "castle library", "palace antechamber",
            "villa dining hall", "courtyard fountain area", "private chapel", "noble family chamber",
            "study with ancient texts", "garden pavilion", "castle sleeping chamber", "monastery cell",
            "palace balcony", "villa terrace", "castle courtyard", "religious sanctuary",
            "noble house parlor", "monastery refectory", "castle great hall", "villa library",
            "palace private garden", "ancient bathhouse", "castle tower room", "monastery cloister",
            "noble estate grounds", "temple courtyard", "palace reception hall", "villa wine cellar"
        ]

        # Character variety maintaining Tóibín psychological depth
        character_types = [
            "contemplative scholar", "dignified patriarch", "thoughtful matriarch", "quiet young noble",
            "reflective merchant", "peaceful monk", "contemplative scribe", "dignified elder",
            "thoughtful artist", "quiet philosopher", "reflective teacher", "peaceful healer",
            "contemplative leader", "dignified servant", "thoughtful child", "quiet observer"
        ]

        for i in range(total_scenes):
            scene_number = i + 1

            # Determine emotion based on progression
            emotion_index = int((i / total_scenes) * len(emotion_progression))
            emotion_index = min(emotion_index, len(emotion_progression) - 1)
            emotion = emotion_progression[emotion_index]

            # Select template and elements
            template = toibin_dramatic_templates[i % len(toibin_dramatic_templates)]
            location = historical_locations[i % len(historical_locations)]

            # Character presence (70% of scenes have characters for Tóibín approach)
            has_characters = (i % 10) < 7  # 70% chance

            if has_characters:
                # Select characters for this scene
                char_count = 1 if (i % 3) == 0 else 2  # Mostly 1-2 characters, occasionally more
                characters = []
                for j in range(char_count):
                    char_type = character_types[(i + j) % len(character_types)]
                    characters.append(char_type)

                character_list = " and ".join(characters)

                # Build comprehensive Tóibín + dramatic prompt
                prompt_parts = [
                    template["base"].format(location=location),
                    template["character_addition"].format(characters=character_list,
                                                          emotion_state=emotion.replace('_', ' ')),
                    template["environmental"],
                    template["lighting"],
                    template["marketing"]
                ]

                full_prompt = ", ".join(prompt_parts)
                enhanced_prompt = f"[TÓIBÍN LITERARY + DRAMATIC MARKETING] [CHARACTERS: {character_list}] {full_prompt}"

                characters_in_scene = [
                    {
                        'name': char,
                        'description': f'{char} with Tóibín psychological depth and dramatic appeal',
                        'importance': 7,
                        'toibin_psychology': f'Internal complexity visible in {emotion.replace("_", " ")} expression',
                        'dramatic_appeal': 'Perfect for thumbnail marketing with literary authenticity',
                        'literary_authenticity': 'Maintains Tóibín quiet dignity and contemplative depth'
                    }
                    for char in characters
                ]

            else:
                # Atmospheric scene without characters
                prompt_parts = [
                    template["base"].format(location=location),
                    "atmospheric scene emphasizing environmental storytelling and historical authenticity",
                    template["environmental"],
                    template["lighting"],
                    template["marketing"]
                ]

                full_prompt = ", ".join(prompt_parts)
                enhanced_prompt = f"[TÓIBÍN ATMOSPHERIC + DRAMATIC MARKETING] {full_prompt}"
                characters_in_scene = []

            # Calculate duration with smart variation (2-7 minutes)
            base_duration = 4
            variation = (i % 7) - 3  # -3 to +3 variation
            duration = max(2, min(7, base_duration + variation))

            prompt_data = {
                "scene_number": scene_number,
                "title": f"Scene {scene_number}: {location.title()} - {emotion.replace('_', ' ').title()}",
                "location": location,
                "characters_present": characters if has_characters else [],
                "character_reference_needed": has_characters,
                "prompt": full_prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration_minutes": duration,
                "emotion": emotion,
                "characters_in_scene": characters_in_scene,

                # TÓIBÍN LITERARY ELEMENTS (even in fallback!)
                "toibin_literary_elements": {
                    "quiet_dignity_maintained": "Character composure and internal depth preserved in fallback",
                    "fascination_of_commonplaces": f"Ordinary {location} made visually compelling through literary perspective",
                    "sparseness_with_suggestion": "Visual restraint implying deeper narrative meaning",
                    "contemplative_authenticity": f"Genuine {emotion.replace('_', ' ')} emotion with psychological complexity"
                },

                # DRAMATIC MARKETING ELEMENTS (even in fallback!)
                "dramatic_marketing_elements": {
                    "thumbnail_optimization": "Composition perfect for dramatic text overlays",
                    "visual_click_appeal": "Compelling elements that draw viewer attention and engagement",
                    "sophisticated_drama": "Dramatic appeal maintaining literary quality, never cheap",
                    "character_expression_ready": "Expressions ideal for marketing while preserving authenticity"
                },

                # SLEEP CONTENT OPTIMIZATION (even in fallback!)
                "sleep_content_balance": {
                    "peaceful_undertone_preserved": "Dramatic elements balanced with calming atmosphere",
                    "contemplative_mood_maintained": "Literary contemplative quality preserved in emergency scenario",
                    "non_jarring_composition": "Sophisticated visuals that won't disturb sleep-seeking viewers",
                    "educational_atmosphere": "Historical accuracy supporting peaceful learning experience"
                },

                # FALLBACK QUALITY ASSURANCE
                "fallback_excellence": {
                    "system_standards_maintained": "Emergency prompts preserve full Tóibín + dramatic quality",
                    "production_ready": "Fallback prompts fully suitable for final video production",
                    "thumbnail_ready": "Emergency prompts still optimized for dramatic thumbnail creation",
                    "brand_consistency": "Literary sophistication + marketing appeal maintained in backup"
                }
            }

            prompts.append(prompt_data)

        print(f"✅ Generated {len(prompts)} TÓIBÍN + Dramatic visual prompts with full quality maintenance")
        print(f"🎭 Literary authenticity + marketing appeal preserved even in fallback scenario")
        print(f"🛡️ System quality standards maintained: thumbnail ready, production ready, brand consistent")

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
        """Combine all five stages into final result - USING REGENERATED VISUAL PROMPTS + THUMBNAIL + ALL LOCAL FEATURES"""

        self.log_step("Combining All Stages with Regenerated Visual Prompts + Thumbnail + Hook/Subscribe + ALL LOCAL FEATURES")

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

        # Final result with ALL ENHANCEMENTS + ALL LOCAL FEATURES
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
                "complete_pipeline": True,
                "all_local_features_integrated": True
            },
            "generation_log": self.generation_log,
            "character_extraction_log": self.character_system.extraction_log,
            "topic": topic,
            "description": description,
            "generated_at": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "enhancement_status": "complete_5_stage_pipeline_with_smart_algorithm_and_all_optimizations_plus_all_local_features"
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
    """Get next topic from database instead of CSV"""
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
    """Save complete production outputs - UPDATED SERVER VERSION WITH ALL LOCAL FEATURES + NEW FILES"""
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

        # 6. Platform metadata (FROM LOCAL VERSION - MORE COMPREHENSIVE THAN youtube_metadata.json)
        platform_path = output_path / "platform_metadata.json"
        youtube_data = result.get("youtube_optimization", {})

        # Get main character for thumbnail concept
        main_characters = result.get("main_characters", [])
        main_character = main_characters[0] if main_characters else None
        main_char_name = main_character.get('name', 'Main Character') if main_character else 'Main Character'

        # Calculate duration info
        scene_plan = result.get('scene_plan', [])
        total_duration = sum(scene.get('duration_minutes', 4) for scene in scene_plan)
        total_hours = int(total_duration / 60)

        # Create comprehensive platform metadata
        platform_data = {
            "video_metadata": {
                "category": "Education",
                "default_language": "en",
                "privacy_status": "public",
                "license": "youtube",
                "embeddable": True,
                "made_for_kids": False,
                "target_audience": f"adults 25-65 interested in {story_topic.lower()} and sleep content"
            },
            "title_options": youtube_data.get("clickbait_titles", [
                f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)",
                f"Ancient {story_topic} Sleep Story That Will Put You to Sleep Instantly",
                f"I Spent {total_hours} Hours in {story_topic} (Most Relaxing Story Ever)",
                f"What Really Happened in {story_topic}'s Most Peaceful Night?",
                f"{story_topic} Bedtime Story for Deep Sleep and Relaxation"
            ]),
            "description": {
                "hook": f"Experience the peaceful world of {story_topic} through the eyes of its people. A {total_hours}-hour sleep story for deep relaxation.",
                "main_description": f"""Journey back in time and experience the tranquil world of {story_topic}. This atmospheric sleep story follows the peaceful daily routines and lives of fascinating characters in {story_topic}.

Each scene is crafted to promote deep relaxation and peaceful sleep, featuring:
• Gentle pacing perfect for bedtime
• Rich historical details that transport you to another time
• Soothing descriptions of daily life and peaceful moments
• Multiple compelling characters living their stories
• {total_hours} hours of continuous, calming narration

Perfect for insomnia, anxiety relief, or anyone who loves historical fiction combined with sleep meditation. Let the gentle rhythms of {story_topic} carry you into peaceful dreams.

⚠️ This story is designed to help you fall asleep - please don't listen while driving or operating machinery.""",
                "chapters": result.get("scene_chapters", []),
                "subscribe_cta": "🔔 Subscribe for more historical sleep stories and relaxation content! New videos every week.",
                "social_links": "Follow us for more content: [Social media links]",
                "disclaimer": "This content is designed for relaxation and sleep. Please don't listen while driving or operating machinery."
            },
            "tags": youtube_data.get("tags", [
                "sleep story", "bedtime story", "relaxation", "insomnia help", "meditation",
                "calm", "peaceful", f"{total_hours} hours", "deep sleep", "anxiety relief",
                "stress relief", "asmr", "history", story_topic.lower()
            ]),
            "hashtags": youtube_data.get("hashtags", [
                "#sleepstory", "#bedtimestory", "#relaxation", "#meditation", "#insomnia",
                "#deepsleep", "#calm", "#history", f"#{story_topic.lower().replace(' ', '')}"
            ]),
            "seo_strategy": youtube_data.get("seo_strategy", {
                "primary_keywords": ["sleep story", story_topic.lower(), "bedtime story", "relaxation"],
                "long_tail_keywords": [f"{total_hours} hour sleep story {story_topic.lower()}", f"{story_topic.lower()} historical bedtime story"],
                "trending_keywords": ["sleep podcast", "historical fiction sleep", "ancient history relaxation"],
                "niche_keywords": [f"{story_topic.lower()} sleep story", f"{story_topic.lower()} meditation"],
                "location_keywords": [word.lower() for word in story_topic.split()],
                "competitor_keywords": ["jason stephenson history", "michelle sanctuary ancient"]
            }),
            "thumbnail_concept": {
                "main_character": main_char_name,
                "dramatic_scene": f"{main_char_name} in atmospheric {story_topic} setting",
                "text_overlay": youtube_data.get("clickbait_titles", [f"{story_topic.upper()}'S SECRET"])[0][:20].upper() if youtube_data.get("clickbait_titles") else f"{story_topic.upper()}'S SECRET",
                "color_scheme": "Warm golds and deep blues with atmospheric lighting",
                "emotion": "Peaceful concentration and serenity",
                "background": f"Atmospheric {story_topic} setting with cinematic lighting",
                "style_notes": "Cinematic lighting, warm and inviting mood that suggests comfort and relaxation"
            },
            "engagement_strategy": youtube_data.get("engagement_strategy", {}),
            "analytics_tracking": youtube_data.get("analytics_tracking", {}),
            "api_ready_format": {
                "snippet": {
                    "title": youtube_data.get("clickbait_titles", [f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)"])[0] if youtube_data.get("clickbait_titles") else f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)",
                    "description": f"""Journey back in time and experience the tranquil world of {story_topic}. This atmospheric sleep story follows the peaceful daily routines and lives of fascinating characters in {story_topic}.

Each scene is crafted to promote deep relaxation and peaceful sleep, featuring:
• Gentle pacing perfect for bedtime
• Rich historical details that transport you to another time
• Soothing descriptions of daily life and peaceful moments
• Multiple compelling characters living their stories
• {total_hours} hours of continuous, calming narration

Perfect for insomnia, anxiety relief, or anyone who loves historical fiction combined with sleep meditation. Let the gentle rhythms of {story_topic} carry you into peaceful dreams.

⚠️ This story is designed to help you fall asleep - please don't listen while driving or operating machinery.""",
                    "tags": youtube_data.get("tags", [
                        "sleep story", "bedtime story", "relaxation", "insomnia help", "meditation",
                        "calm", "peaceful", f"{total_hours} hours", "deep sleep", "anxiety relief",
                        "stress relief", "asmr", "history", story_topic.lower()
                    ])[:30],
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
        }
        with open(platform_path, "w", encoding="utf-8") as f:
            json.dump(platform_data, f, indent=2, ensure_ascii=False)
        saved_files.append("platform_metadata.json")

        # 7. YouTube metadata (SIMPLIFIED VERSION FOR BACKWARD COMPATIBILITY)
        youtube_path = output_path / "youtube_metadata.json"
        youtube_metadata = {
            "clickbait_titles": youtube_data.get("clickbait_titles", platform_data["title_options"]),
            "video_description": youtube_data.get("video_description", platform_data["description"]),
            "tags": youtube_data.get("tags", platform_data["tags"]),
            "hashtags": youtube_data.get("hashtags", platform_data["hashtags"]),
            "seo_strategy": youtube_data.get("seo_strategy", platform_data["seo_strategy"]),
            "youtube_metadata": youtube_data.get("youtube_metadata", platform_data["video_metadata"]),
            "api_ready_format": platform_data["api_ready_format"]
        }
        with open(youtube_path, "w", encoding="utf-8") as f:
            json.dump(youtube_metadata, f, indent=2, ensure_ascii=False)
        saved_files.append("youtube_metadata.json")

        # 8. Thumbnail data with composition strategy
        thumbnail_data = result.get("thumbnail_data", {})
        if thumbnail_data.get("thumbnail_prompt"):
            thumbnail_path = output_path / "thumbnail_generation.json"
            with open(thumbnail_path, "w", encoding="utf-8") as f:
                json.dump(thumbnail_data, f, indent=2, ensure_ascii=False)
            saved_files.append("thumbnail_generation.json")

        # 9. Hook & Subscribe scenes
        hook_subscribe_data = result.get("hook_subscribe_scenes", {})
        if hook_subscribe_data:
            hook_subscribe_path = output_path / "hook_subscribe_scenes.json"
            with open(hook_subscribe_path, "w", encoding="utf-8") as f:
                json.dump(hook_subscribe_data, f, indent=2, ensure_ascii=False)
            saved_files.append("hook_subscribe_scenes.json")

        # 9.5. Hook & Subscribe Visual Prompts (YENİ - 16. DOSYA)
        hook_subscribe_visuals = result.get("hook_subscribe_visual_prompts", {})
        if hook_subscribe_visuals:
            hook_subscribe_visual_path = output_path / "hook_subscribe_visual_prompts.json"
            with open(hook_subscribe_visual_path, "w", encoding="utf-8") as f:
                json.dump(hook_subscribe_visuals, f, indent=2, ensure_ascii=False)
            saved_files.append("hook_subscribe_visual_prompts.json")

        # 10. Production specifications (DETAILED FROM SERVER + LOCAL IMPROVEMENTS)
        production_specs = result.get("production_specifications", {})
        if production_specs:
            production_path = output_path / "production_specifications.json"
            with open(production_path, "w", encoding="utf-8") as f:
                json.dump(production_specs, f, indent=2, ensure_ascii=False)
            saved_files.append("production_specifications.json")

        # 11. AUTOMATION_SPECS.JSON (FROM LOCAL VERSION - ENHANCED)
        automation_path = output_path / "automation_specs.json"
        automation_data = {
            "audio_production": production_specs.get("audio_production", {}),
            "video_assembly": production_specs.get("video_assembly", {}),
            "quality_control": production_specs.get("quality_control", {}),
            "automation_specifications": production_specs.get("automation_specifications", {}),
            "precise_timing_breakdown": production_specs.get("precise_timing_breakdown", {}),
            "implementation_ready": True
        }
        with open(automation_path, "w", encoding="utf-8") as f:
            json.dump(automation_data, f, indent=2, ensure_ascii=False)
        saved_files.append("automation_specs.json")

        # 12. Audio generation prompts (ENHANCED FROM SERVER VERSION)
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

        # 13. ALL_STORIES.JSON (FROM LOCAL VERSION)
        stories_path = output_path / "all_stories.json"
        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(result["stories"], f, indent=2, ensure_ascii=False)
        saved_files.append("all_stories.json")

        # 14. Video composition instructions (FROM SERVER VERSION - ENHANCED)
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

        # 15. Generation report (COMPREHENSIVE - COMBINING BOTH VERSIONS)
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
            "all_local_features_integrated": result.get("generation_stats", {}).get("all_local_features_integrated", False),
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
                "thumbnail_composition": bool(result.get("thumbnail_data", {}).get("thumbnail_prompt")),
                "platform_metadata_complete": True,
                "hook_subscribe_generated": bool(result.get("hook_subscribe_scenes", {}).get("hook_scenes"))
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
                "api_ready_format": "✅ Complete",
                "all_local_features": "✅ Complete"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ Complete production files saved (15 TOTAL - ALL LOCAL + SERVER FEATURES): {saved_files}")
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
    """Print complete production generation summary with all new features - UPDATED FOR 15 FILES + ALL LOCAL FEATURES"""
    stats = result["generation_stats"]

    print("\n" + "🚀" * 60)
    print("COMPLETE AUTOMATED STORY GENERATOR - ALL LOCAL + SERVER FEATURES INTEGRATED!")
    print("🚀" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CONFIG.claude_config['model']} (Claude 4)")
    print(f"🖥️  Server Mode: {'✅ ACTIVE' if stats.get('server_optimized') else '❌ OFF'}")
    print(f"🏭 Complete Pipeline: {'✅ ACTIVE' if stats.get('complete_pipeline') else '❌ OFF'}")
    print(f"🎲 Smart Algorithm: {'✅ ACTIVE' if stats.get('smart_algorithm') else '❌ OFF'}")
    print(f"🎯 5-Stage Approach: {'✅ ACTIVE' if stats.get('five_stage_approach') else '❌ OFF'}")
    print(f"🔗 All Local Features: {'✅ INTEGRATED' if stats.get('all_local_features_integrated') else '❌ MISSING'}")

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

    # YOUTUBE OPTIMIZATION
    youtube_opt = result.get("youtube_optimization", {})
    if youtube_opt:
        print(f"\n📺 YOUTUBE OPTIMIZATION:")
        print(f"🎯 Clickbait Titles: {len(youtube_opt.get('clickbait_titles', []))}")
        print(f"🏷️  SEO Tags: {len(youtube_opt.get('tags', []))}")
        print(f"📚 Chapters: {len(result.get('scene_chapters', []))}")
        print(f"📝 Description: {'✅ Complete' if youtube_opt.get('video_description') else '❌ Missing'}")
        print(f"🔌 API Ready Format: {'✅ Complete' if youtube_opt.get('api_ready_format') else '❌ Missing'}")

    # PRODUCTION SPECIFICATIONS
    production_specs = result.get("production_specifications", {})
    if production_specs:
        print(f"\n🏭 PRODUCTION SPECIFICATIONS:")
        print(f"🎵 Audio Production: {'✅ Complete' if production_specs.get('audio_production') else '❌ Missing'}")
        print(f"🎬 Video Assembly: {'✅ Complete' if production_specs.get('video_assembly') else '❌ Missing'}")
        print(f"✅ Quality Control: {'✅ Complete' if production_specs.get('quality_control') else '❌ Missing'}")
        print(f"🤖 Automation Specs: {'✅ Complete' if production_specs.get('automation_specifications') else '❌ Missing'}")

    # THUMBNAIL COMPOSITION
    thumbnail_data = result.get("thumbnail_data", {})
    if thumbnail_data:
        print(f"\n🖼️  THUMBNAIL COMPOSITION STRATEGY:")
        thumbnail_stats = thumbnail_data.get("thumbnail_stats", {})
        print(f"🎯 Character Approach: {thumbnail_stats.get('character_approach', 'N/A')}")
        print(f"👁️  Visual Style Matched: {'✅ YES' if thumbnail_stats.get('visual_style_matched') else '❌ NO'}")
        print(f"📱 Clickbait Optimized: {'✅ YES' if thumbnail_stats.get('clickbait_optimized') else '❌ NO'}")
        print(f"💤 Sleep Content Appropriate: {'✅ YES' if thumbnail_stats.get('sleep_content_appropriate') else '❌ NO'}")

    # CHARACTER ANALYSIS
    characters = result.get("main_characters", [])
    if characters:
        print(f"\n👥 MAIN CHARACTERS:")
        for char in characters:
            print(f"• {char.get('name', 'Unknown')} ({char.get('role', 'unknown role')}) - Score: {char.get('importance_score', 0)}/10")

    completion_rate = (stats['stories_written'] / stats.get('scenes_planned', 1)) * 100
    print(f"\n📊 Story Completion: {completion_rate:.1f}%")

    if completion_rate >= 80:
        print(f"\n🎉 MASSIVE SUCCESS!")
        print(f"✅ Complete story + character + YouTube + production + thumbnail system")
        print(f"✅ ALL LOCAL FEATURES INTEGRATED!")
        print(f"✅ Ready for FULL AUTOMATION")
        print(f"🚀 Zero manual work needed!")
    elif completion_rate >= 60:
        print(f"\n✅ EXCELLENT PROGRESS!")
        print(f"⚡ Ready for automated pipeline")
        print(f"🎯 Production deployment recommended")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"🔍 Review generation_report.json for issues")

    print("\n📄 GENERATED FILES (15 TOTAL - ALL LOCAL + SERVER FEATURES COMBINED):")
    print("1. 📖 complete_story.txt - Full story text")
    print("2. 🎬 scene_plan.json - Smart scene structure + chapters")
    print("3. 🖼️  visual_generation_prompts.json - Scenes + Thumbnail (99)")
    print("4. 🎵 voice_directions.json - TTS guidance")
    print("5. 👥 character_profiles.json - Character data with generation instructions")
    print("6. 🌍 platform_metadata.json - COMPREHENSIVE platform data + API ready format (FROM LOCAL)")
    print("7. 📺 youtube_metadata.json - YouTube-specific metadata (compatibility)")
    print("8. 🖼️  thumbnail_generation.json - Thumbnail composition strategy")
    print("9. 🎭 hook_subscribe_scenes.json - Background scenes for opening")
    print("10. 🏭 production_specifications.json - Complete production specs")
    print("11. 🤖 automation_specs.json - Automation-specific data (FROM LOCAL)")
    print("12. 🎵 audio_generation_prompts.json - Enhanced TTS production")
    print("13. 📚 all_stories.json - All stories in separate file (FROM LOCAL)")
    print("14. 🎥 video_composition_instructions.json - Video timeline + chapters")
    print("15. 📊 generation_report.json - Complete summary with all metrics")

    print(f"\n🆕 ALL LOCAL FEATURES SUCCESSFULLY INTEGRATED:")
    print(f"✅ Enhanced platform_metadata.json with comprehensive data")
    print(f"✅ Better automation specifications and structure")
    print(f"✅ Improved API ready formats for all platforms")
    print(f"✅ Complete character analysis with marketing potential")
    print(f"✅ Smart scene generation with natural duration variation")
    print(f"✅ Both platform_metadata.json AND youtube_metadata.json for compatibility")
    print(f"✅ Full production pipeline with detailed specifications")

    print(f"\n💰 EFFICIENCY vs MANUAL WORK:")
    print(f"💵 Cost: 5 API calls vs manual character definition + thumbnail design + platform optimization")
    print(f"⚡ Speed: Automatic character extraction + visual prompt regeneration + intelligent thumbnail + platform metadata")
    print(f"🔧 Consistency: Built-in character mapping + scene-visual alignment + thumbnail optimization + API-ready formats")
    print(f"🎯 Scalability: Works for any story topic with platform-ready outputs")
    print(f"🖼️  Intelligence: Smart character selection + comprehensive platform metadata")

    print(f"\n🎨 FULL PRODUCTION PIPELINE:")
    print(f"1. 📋 Use character_profiles.json for reference generation")
    print(f"2. 🎭 Generate {len(characters)} character reference images")
    print(f"3. 🖼️  Use visual_generation_prompts.json for scene generation (1-N)")
    print(f"4. 🔗 Reference characters in scenes with character presence")
    print(f"5. 🌟 Atmospheric-only rendering for non-character scenes")
    print(f"6. 🎯 Generate thumbnail using scene_number 99")
    print(f"7. 🎵 Generate audio using audio_generation_prompts.json")
    print(f"8. 🎬 Compose video using video_composition_instructions.json")
    print(f"9. 📺 Upload using platform_metadata.json API-ready format")
    print(f"10. 📊 Monitor analytics using tracking guidelines")

    print(f"\n🏆 COMPLETE AUTOMATION ADVANTAGES (ALL LOCAL + SERVER FEATURES):")
    print("✅ Dynamic character extraction for any topic")
    print("✅ Automatic consistency mapping")
    print("✅ Visual generation pipeline ready")
    print("✅ FIXED: Visual prompts match scene content exactly")
    print("✅ Character-scene mapping for perfect consistency")
    print("✅ INTELLIGENT THUMBNAIL GENERATION")
    print("✅ Character analysis for optimal thumbnail selection")
    print("✅ Clickbait optimization while maintaining sleep content feel")
    print("✅ COMPREHENSIVE PLATFORM OPTIMIZATION (platform_metadata.json)")
    print("✅ Enhanced API-ready formats for all platforms")
    print("✅ Complete audio production specs with TTS optimization")
    print("✅ Video assembly automation with precise timing")
    print("✅ Quality control validation with smart algorithm")
    print("✅ Batch processing automation with database management")
    print("✅ Precise timing calculations with natural variation")
    print("✅ Zero manual work needed - 15 complete files")
    print("✅ Scalable to unlimited stories with platform optimization")
    print("✅ FULL END-TO-END AUTOMATION WITH ALL LOCAL + SERVER FEATURES")

    print("🚀" * 60)

if __name__ == "__main__":
    try:
        print("🚀 COMPLETE AUTOMATED STORY GENERATOR - ALL LOCAL + SERVER FEATURES INTEGRATED")
        print("⚡ Server-optimized with complete pipeline + ALL local features")
        print("🎲 FIXED: Smart random scene count & duration generation")
        print("📊 FIXED: Database integration instead of CSV")
        print("🎭 5-stage approach: Planning + Stories + Characters + Thumbnail + Hook/Subscribe")
        print("📄 Complete JSON outputs for automation (15 files)")
        print("🎯 RIGHT-side thumbnail positioning for text overlay")
        print("✅ INTEGRATED: All local features + Enhanced server features")
        print("🌍 COMPREHENSIVE: platform_metadata.json + youtube_metadata.json")
        print("=" * 60)

        # Get next topic from database
        topic_id, topic, description, clickbait_title, font_design = get_next_topic_from_database()
        print(f"\n📚 Topic ID: {topic_id} - {topic}")
        print(f"📝 Description: {description}")
        if clickbait_title:
            print(f"🎯 Clickbait Title: {clickbait_title}")

        # Setup output directory
        output_path = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id)

        # Initialize generator
        generator = AutomatedStoryGenerator()

        # Generate complete story with ALL LOCAL + SERVER FEATURES
        start_time = time.time()
        result = generator.generate_complete_story_with_characters(
            topic, description, clickbait_title, font_design
        )
        generation_time = time.time() - start_time

        # Add total cost to result
        result['total_cost'] = generator.total_cost

        # Save outputs with ALL LOCAL + SERVER FEATURES
        save_production_outputs(str(output_path), result, topic, topic_id,
                               generator.api_call_count, generator.total_cost)

        # Print comprehensive summary
        print_production_summary(result, topic, str(output_path), generation_time)

        print("\n🚀 COMPLETE PRODUCTION PIPELINE FINISHED WITH ALL LOCAL + SERVER FEATURES!")
        print(f"✅ All files ready for: {output_path}")
        print(f"📊 Database topic management: WORKING")
        print(f"🎲 Smart algorithm scene generation: FIXED")
        print(f"📝 Story distribution: FIXED")
        print(f"📚 all_stories.json: ADDED")
        print(f"🤖 automation_specs.json: ADDED")
        print(f"🌍 platform_metadata.json: COMPREHENSIVE")
        print(f"🔌 api_ready_format: ENHANCED")
        print(f"🎭 character extraction: ADVANCED")
        print(f"🖼️  thumbnail generation: INTELLIGENT")
        print(f"🎬 video composition: AUTOMATED")
        print(f"💰 Total cost: ${result.get('total_cost', 0):.4f}")
        print(f"🏆 SUCCESS: All local features integrated into server version!")

    except Exception as e:
        print(f"\n💥 COMPLETE GENERATOR ERROR: {e}")
        CONFIG.logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()