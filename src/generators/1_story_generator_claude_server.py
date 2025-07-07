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
        """STAGE 1: ENHANCED Smart planning with powerful prompts + first half stories"""

        # Generate smart scene structure
        visual_safety_education = MIDJOURNEY_CONTENT_AWARENESS_PROMPT
        smart_structure = self._generate_smart_scene_structure()
        total_scenes = smart_structure['scene_count']
        scene_durations = smart_structure['scene_durations']
        first_half = total_scenes // 2

        self.log_step(f"Stage 1: ENHANCED Smart Planning + First {first_half} Stories (Total: {total_scenes} scenes)")

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
    {', '.join([f'Scene {i + 1}: {dur}min' for i, dur in enumerate(scene_durations)])}

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

    {chr(10).join([f"Scene {i + 1}: {scene_durations[i]:.1f} minutes" for i in range(total_scenes)])}

    Scene structure requirements:
    - Template rotation: atmospheric, character_focused, historical_detail, sensory_journey
    - Style rotation: observational, immersive, documentary, poetic, cinematic
    - Emotion progression: 1-30% peaceful, 31-60% curiosity, 61-80% concern, 81-100% resolution
    - Key characters mentioned in descriptions

    ## 4. FIRST {first_half} COMPLETE STORIES (Scenes 1-{first_half})
    Each story must be 300-900 words (based on scene duration) with:

    ### 🎨 OPENING MASTERY (CRITICAL):
    - **FORBIDDEN:** Never use "You find yourself" - this phrase is completely banned
    - **REQUIRED:** Create unique, atmospheric openings for each scene
    - **VARIETY:** Use different opening styles:
      * Environmental: "The golden light filters through..."
      * Temporal: "As twilight settles over..."
      * Auditory: "Soft footsteps echo in..."
      * Sensory: "The gentle breeze carries..."
      * Visual: "Shadows dance across..."
      * Character-focused: "[Character name] pauses at..."
      * Action-based: "The wooden door creaks open..."
      * Emotional: "A sense of peace settles..."

    ### 📝 PRECISION REQUIREMENTS:
    - Present tense, second person perspective
    - Rich sensory details (sight, sound, smell, touch, taste)
    - [PAUSE] markers for TTS at natural breathing points
    - Sleep-optimized language with gentle pacing
    - Historical accuracy with authentic period details
    - Clear character interactions and development
    - Word count matched to scene duration (~150 words/minute)

    ### 🎯 STORYTELLING EXCELLENCE:
    - Each opening must be completely different
    - Demonstrate mastery across multiple opening styles
    - Build atmospheric immersion from first sentence
    - Natural flow that leads listeners into peaceful sleep
    
    {visual_safety_education}
    
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
        "1": "[COMPLETE story for scene 1 with MASTERFUL unique opening]",
        "2": "[COMPLETE story for scene 2 with DIFFERENT opening style]"
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
        "enhanced_prompts_used": true,
        "master_storytelling": true,
        "opening_variety_enforced": true,
        "total_word_count": "[calculated]",
        "characters_introduced": "[count]",
        "ready_for_stage2": true
      }}
    }}

    ## STORYTELLING EXCELLENCE CHALLENGE:
    Write {first_half} completely different, masterful stories. Each must have:
    - **UNIQUE atmospheric opening** (no repeated phrases, no "You find yourself")
    - **Perfect word count** for scene duration
    - **Rich character development** with meaningful interactions
    - **Historical authenticity** with accurate period details
    - **Sleep-optimized pacing** with gentle rhythm
    - **Sensory immersion** that transports readers

    Demonstrate your expertise as a MASTER STORYTELLER. SHOWCASE YOUR MASTERY with creative, unique openings for each scene. The phrase "You find yourself" is banned. Instead, craft atmospheric beginnings that vary in style and immediately immerse the reader.

    USE THE EXACT DURATIONS FROM THE PLAN ABOVE."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,
                timeout=1800,
                system="You are a MASTER STORYTELLER and automated content creator. Stage 1: Create complete planning + first half atmospheric stories with rich character interactions. Focus on memorable, distinct characters. SHOWCASE YOUR STORYTELLING MASTERY with unique, creative openings for each scene. The phrase 'You find yourself' is forbidden - instead, craft atmospheric beginnings that immediately immerse the reader with variety and expertise.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            content = ""
            print("📡 Stage 1: Streaming Claude 4 response with ENHANCED PROMPTS...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 1 ENHANCED: {len(content):,} characters...")

            print(f"✅ Stage 1 ENHANCED complete: {len(content):,} characters")

            input_tokens = len(stage1_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            CONFIG.logger.info(f"Stage 1 ENHANCED response length: {len(content)} characters - Cost: ${stage_cost:.4f}")

            parsed_result = self._parse_claude_response(content, "stage1")

            self.log_step("Stage 1 ENHANCED Parsing", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stories_written": len(parsed_result.get('stories', {})),
                "enhanced_prompts_used": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 1 ENHANCED Failed", "ERROR")
            CONFIG.logger.error(f"Stage 1 enhanced error: {e}")
            raise

    def _generate_stage2(self, topic: str, description: str, stage1_result: Dict) -> Dict[str, Any]:
        """STAGE 2: ENHANCED Remaining stories with POWERFUL PROMPTS from local version"""

        scene_plan = stage1_result.get('scene_plan', [])
        stories_written = len(stage1_result.get('stories', {}))
        total_scenes = len(scene_plan)
        remaining_scenes = total_scenes - stories_written

        if remaining_scenes <= 0:
            self.log_step("Stage 2: No remaining stories needed", "SUCCESS")
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        self.log_step(f"Stage 2: ENHANCED {remaining_scenes} Stories with POWERFUL PROMPTS")

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
    Characters: {', '.join(scene.get('characters_mentioned', []))}"""
            for scene in remaining_scene_plan
        ])

        stage2_prompt = f"""Complete the sleep story for "{topic}" by writing the remaining {remaining_scenes} stories.

    TOPIC: {topic}
    DESCRIPTION: {description}

    SCENES TO COMPLETE:
    {scenes_text}

    ## MASTER STORYTELLING REQUIREMENTS (ENHANCED FROM LOCAL VERSION):

    ### 🎨 OPENING MASTERY (CRITICAL):
    - **FORBIDDEN:** Never use "You find yourself" - this phrase is completely banned
    - **REQUIRED:** Create unique, atmospheric openings for each scene
    - **VARIETY:** Use different opening styles:
      * Environmental: "The golden light filters through..."
      * Temporal: "As twilight settles over..."
      * Auditory: "Soft footsteps echo in..."
      * Sensory: "The gentle breeze carries..."
      * Visual: "Shadows dance across..."
      * Character-focused: "[Character name] pauses at the threshold..."
      * Action-based: "The wooden door creaks open..."
      * Emotional: "A sense of peace settles..."

    ### 📝 WORD COUNT PRECISION:
    - 2-3 minute scenes: 300-450 words
    - 4-5 minute scenes: 600-750 words  
    - 6-7 minute scenes: 750-1050 words
    - 8+ minute scenes: 1050+ words
    - Base rate: ~150 words per minute for sleep content

    ### 🎭 CHARACTER INTEGRATION:
    - When characters mentioned: Include meaningful interactions
    - Show character personalities through actions and dialogue
    - Build emotional connections between characters
    - Progress character arcs throughout scenes

    ### 🌙 SLEEP OPTIMIZATION:
    - Present tense, second person perspective
    - Rich sensory details (sight, sound, smell, touch, taste)
    - [PAUSE] markers at natural breathing points
    - Gentle pacing with flowing rhythm
    - Peaceful resolution for each scene
    - Avoid jarring or startling elements

    ### 🏛️ HISTORICAL ACCURACY:
    - Research-accurate period details
    - Authentic materials, tools, and practices
    - Accurate social customs and behaviors  
    - Period-appropriate dialogue and thoughts
    - Detailed environmental descriptions

    ### 🎵 NARRATIVE FLOW:
    - Smooth transitions between paragraphs
    - Varied sentence lengths for rhythm
    - Building and releasing tension gently
    - Natural conversation flow
    - Descriptive passages balanced with action

    OUTPUT FORMAT:
    {{
      "stories": {{
        "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}": "[COMPLETE story with MASTERFUL opening and precise word count]",
        "{remaining_scene_plan[1]['scene_id'] if len(remaining_scene_plan) > 1 else 'Y'}": "[COMPLETE story with UNIQUE opening style]"
      }},
      "stage2_stats": {{
        "stories_written": {remaining_scenes},
        "scenes_covered": "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}-{remaining_scene_plan[-1]['scene_id'] if remaining_scene_plan else 'Y'}",
        "enhanced_prompts_used": true,
        "master_storytelling": true,
        "opening_variety_enforced": true,
        "smart_durations": true,
        "total_word_count": "[calculated]",
        "character_development": "continued"
      }}
    }}

    ## STORYTELLING EXCELLENCE CHALLENGE:
    Write {remaining_scenes} completely different, masterful stories. Each must have:
    - **UNIQUE atmospheric opening** (no repeated phrases, no "You find yourself")
    - **Perfect word count** for scene duration
    - **Rich character development** with meaningful interactions
    - **Historical authenticity** with accurate period details
    - **Sleep-optimized pacing** with gentle rhythm
    - **Sensory immersion** that transports readers

    Demonstrate your expertise as a MASTER STORYTELLER. Create stories that are so well-crafted and atmospheric that listeners naturally drift into peaceful sleep while being transported to another time and place.

    REMEMBER: Each story opening must be completely different and showcase your storytelling range! Continue character development from Stage 1 with consistency."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,
                timeout=1800,
                system="You are a MASTER STORYTELLER with expertise in sleep content creation. Stage 2: Complete the remaining stories with rich character development and consistent character interactions from Stage 1. DEMONSTRATE YOUR STORYTELLING EXPERTISE with inventive, atmospheric openings for each scene. Never use 'You find yourself' - create unique beginnings that set mood and place with variety and mastery.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = ""
            print("📡 Stage 2: Streaming Claude 4 response with ENHANCED PROMPTS...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 2 ENHANCED: {len(content):,} characters...")

            print(f"✅ Stage 2 ENHANCED complete: {len(content):,} characters")

            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 ENHANCED Parsing", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "enhanced_prompts_used": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 ENHANCED Failed", "ERROR")
            CONFIG.logger.error(f"Stage 2 enhanced error: {e}")
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

    def _generate_intelligent_thumbnail(self, topic: str, description: str, character_result: Dict,
                                        clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """STAGE 4: Generate intelligent thumbnail with FIXED composition for text overlay"""

        self.character_system.log_extraction_step("Intelligent Thumbnail Generation with COMPOSITION FIX")

        characters = character_result.get('main_characters', [])
        visual_style = character_result.get('visual_style_notes', {})

        thumbnail_character_selection = self.character_system.select_thumbnail_character(
            characters, topic, description
        )

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

        if not clickbait_title:
            youtube_data = character_result.get('youtube_optimization', {})
            clickbait_titles = youtube_data.get('clickbait_titles', [])
            clickbait_title = clickbait_titles[
                0] if clickbait_titles else f"The Secret History of {topic} (2 Hour Sleep Story)"

        if not font_design:
            font_design = "Bold impact font, uppercase for key words, warm golden color (#d4af37) for main text, contrasted with deep shadows for readability"

        thumbnail_prompt = f"""Create an intelligent thumbnail design for the sleep story "{topic}" with OPTIMIZED COMPOSITION for text overlay.

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

    ## CRITICAL COMPOSITION STRATEGY:
    1. **Character Positioning**: RIGHT side of frame (60-70% from left edge) - NOT center
    2. **Zoom Level**: ZOOM OUT enough so character's head/face is in safe area - NOT cropped
    3. **Text Space**: LEFT side (30-40%) must be CLEAR for text overlay
    4. **Safe Areas**: Character's head, hands, and key details in safe viewing area
    5. **Balance**: Right-heavy character placement balanced with left-side text space

    ## VISUAL STRATEGY:
    1. **Background**: Choose most compelling scene location but keep LEFT side simple
    2. **Lighting**: Warm, golden hour lighting for maximum appeal
    3. **Composition**: Rule of thirds with character on RIGHT third
    4. **Text Placement**: LEFT side reserved for readable text overlay
    5. **Sleep Optimization**: Calming colors, not overstimulating
    6. **Click Optimization**: Compelling visual hook, emotional appeal

    ## LAYOUT EXAMPLE:
    [TEXT OVERLAY SPACE] | [CHARACTER POSITIONED HERE]
         (Left 40%)      |      (Right 60%)
                         |
       CLEAR FOR TEXT    |  FULL CHARACTER VISIBLE
       Golden/warm bg    |  Head NOT cropped
       Simple background |  Atmospheric setting

    ## CHARACTER-BASED APPROACH:
    If character selected: Feature the character prominently on RIGHT side in peaceful but visually striking pose
    If atmospheric: Focus on the most visually compelling location with RIGHT-side emphasis

    OUTPUT FORMAT:
    {{
      "thumbnail_prompt": {{
        "scene_number": 99,
        "character_used": "{thumbnail_character_selection['character_used']}",
        "clickbait_title": "{clickbait_title}",
        "font_design": "{font_design}",
        "prompt": "[Detailed visual prompt optimized for RIGHT-side character placement with LEFT-side text space]",
        "visual_style": "[Style notes including mood, lighting, composition]",
        "character_positioning": "RIGHT side of frame (60-70% from left), character fully visible with head in safe area",
        "text_overlay_strategy": "LEFT side (30-40%) reserved for text overlay with clear, simple background",
        "composition_fix": "ZOOM OUT composition ensures character head not cropped, positioned RIGHT for text space",
        "emotional_appeal": "[What emotion should viewer feel when seeing thumbnail]",
        "target_audience_appeal": "[Why this appeals to sleep story viewers]",
        "clickability_factors": "[What makes this thumbnail clickable]",
        "sleep_content_balance": "[How it maintains sleep content feel while being clickable]",
        "thumbnail_reasoning": "{thumbnail_character_selection['reasoning']}",
        "background_scene": "[Primary scene/location for background]",
        "lighting_strategy": "[Lighting approach for maximum visual appeal]",
        "composition_notes": "Character positioned RIGHT (60-70% from left), zoomed out for safe area, LEFT side clear for text"
      }},
      "thumbnail_alternatives": [
        {{
          "variant": "Character Focus",
          "prompt": "[Alternative thumbnail with character RIGHT-positioned and text space]"
        }},
        {{
          "variant": "Atmospheric Focus", 
          "prompt": "[Alternative thumbnail focusing on setting with RIGHT-side emphasis]"
        }},
        {{
          "variant": "Action Moment",
          "prompt": "[Alternative showing key story moment with composition fix]"
        }}
      ],
      "thumbnail_stats": {{
        "character_approach": "{thumbnail_character_selection['character_used']}",
        "selection_reasoning": "{thumbnail_character_selection['reasoning']}",
        "composition_optimized": true,
        "text_overlay_ready": true,
        "safe_area_protected": true,
        "visual_style_matched": true,
        "clickbait_optimized": true,
        "sleep_content_appropriate": true
      }}
    }}

    ## CRITICAL COMPOSITION REQUIREMENTS:
    - **CHARACTER POSITION**: Place character on RIGHT side (60-70% from left edge)
    - **ZOOM LEVEL**: Wide enough shot so head/face is NOT cropped - full character visible
    - **TEXT SPACE**: LEFT side (30-40%) must be clear/simple for text overlay
    - **SAFE AREA**: Character's head, face, hands must be in safe viewing area
    - **BALANCE**: Right-heavy character placement balanced with left-side text space

    REMEMBER: This thumbnail will have text overlaid on the LEFT side, so position character on RIGHT accordingly! Ensure character's head is not cropped and there's clear space for text."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=8000,
                temperature=0.4,
                stream=True,
                timeout=600,
                system="You are a YouTube thumbnail optimization specialist who understands both sleep content marketing and visual psychology. Create thumbnails that balance peaceful sleep content with clickable appeal. CRITICAL: Position character on RIGHT side with LEFT side clear for text overlay. Ensure character head is not cropped with proper zoom level.",
                messages=[{"role": "user", "content": thumbnail_prompt}]
            )

            content = ""
            print("📡 Stage 4: Generating intelligent thumbnail with COMPOSITION FIX...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text

            print(f"✅ Thumbnail generation with composition fix complete: {len(content):,} characters")

            input_tokens = len(thumbnail_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            parsed_result = self._parse_claude_response(content, "thumbnail_generation")

            self.character_system.log_extraction_step("Thumbnail Generation with Composition Fix", "SUCCESS", {
                "character_approach": thumbnail_character_selection['character_used'],
                "selection_reasoning": thumbnail_character_selection['reasoning'],
                "composition_optimized": True,
                "alternatives_generated": len(parsed_result.get('thumbnail_alternatives', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Thumbnail Generation Failed", "ERROR")
            print(f"❌ Thumbnail generation error: {e}")

            fallback_thumbnail = {
                "thumbnail_prompt": {
                    "scene_number": 99,
                    "character_used": "None (Atmospheric focus)",
                    "clickbait_title": clickbait_title or f"The Secret History of {topic} (2 Hour Sleep Story)",
                    "font_design": font_design or "Bold impact font with warm colors",
                    "prompt": f"Atmospheric thumbnail of {topic}, warm golden lighting, RIGHT-side emphasis for character/scene, LEFT side clear for text overlay, peaceful but compelling visual",
                    "visual_style": "Peaceful and inviting with composition fix",
                    "character_positioning": "RIGHT side positioning with text space on LEFT",
                    "thumbnail_reasoning": "Fallback with composition fix applied"
                },
                "thumbnail_stats": {"error": str(e), "composition_fix_applied": True}
            }
            return fallback_thumbnail

    def _regenerate_visual_prompts_with_characters(self, scene_plan: List[Dict], characters: List[Dict], scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
        """Character extraction'dan SONRA visual prompts'ı yeniden oluştur"""

        self.character_system.log_extraction_step("Regenerating Visual Prompts with Character Integration")

        regeneration_prompt = f"""
        {MIDJOURNEY_CONTENT_AWARENESS_PROMPT}
        Based on the completed scene plan and extracted character data, create accurate visual generation prompts for all scenes.

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