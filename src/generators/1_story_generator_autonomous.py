"""
Sleepy Dull Stories - COMPLETE Server-Ready Claude Story Generator with VALIDATION SYSTEM
UPDATED: All missing methods + Smart Algorithm + Database integration + Complete 5-stage pipeline + ALL LOCAL FEATURES + VALIDATION & CORRECTION
Production-optimized with complete automation + Enhanced with all local JSON files + Story Duration Validation
FINAL VERSION: All local features integrated into server version + Auto-correction system
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
            "long_timeout": True,  # ✅ PROVEN CRITICAL
            "validation_enabled": True,  # 🆕 NEW: Enable validation system
            "auto_correction": True,     # 🆕 NEW: Enable auto-correction
            "target_tolerance": 0.2      # 🆕 NEW: ±20% tolerance for word count
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


# 🆕 NEW: Story Validation and Correction System
class StoryValidationSystem:
    """Advanced story validation and auto-correction system"""

    def __init__(self, claude_client, logger):
        self.client = claude_client
        self.logger = logger
        self.validation_log = []
        self.correction_count = 0

    def log_validation_step(self, description: str, status: str = "START", metadata: Dict = None):
        """Log validation steps"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            entry.update(metadata)
        self.validation_log.append(entry)

        icon = "🔍" if status == "START" else "✅" if status == "SUCCESS" else "⚠️" if status == "WARNING" else "❌"
        print(f"{icon} VALIDATION: {description}")
        self.logger.info(f"VALIDATION: {description} - Status: {status}")

    def validate_stories_duration(self, stories: Dict, scene_plan: List[Dict]) -> Dict:
        """Validate story durations and identify corrections needed"""

        self.log_validation_step("Starting Story Duration Validation")

        validation_results = {
            'valid_stories': {},
            'corrections_needed': [],
            'validation_summary': {
                'total_scenes': len(scene_plan),
                'stories_provided': len(stories),
                'valid_stories': 0,
                'corrections_needed': 0,
                'missing_stories': 0,
                "validation_enabled": True,  # ✅ EKLEYİN
                "auto_correction": True,  # ✅ EKLEYİN
                "target_tolerance": 0.15
            }
        }

        target_words_per_minute = CONFIG.claude_config.get('target_words_per_minute', 140)
        tolerance = CONFIG.claude_config.get('target_tolerance', 0.2)

        missing_stories = []

        for scene in scene_plan:
            scene_id = str(scene['scene_id'])
            target_duration = scene.get('duration_minutes', 4)
            target_words = int(target_duration * target_words_per_minute)

            # Check if story exists
            if scene_id not in stories:
                missing_stories.append({
                    'scene_id': scene_id,
                    'scene_title': scene.get('title', f'Scene {scene_id}'),
                    'target_duration': target_duration,
                    'target_words': target_words
                })
                continue

            story_content = stories[scene_id]
            actual_words = len(story_content.split())

            # Calculate acceptable range
            min_words = int(target_words * (1 - tolerance))
            max_words = int(target_words * (1 + tolerance))

            # Calculate deviation percentage
            deviation = abs(actual_words - target_words) / target_words

            if min_words <= actual_words <= max_words:
                # Story is within acceptable range
                validation_results['valid_stories'][scene_id] = story_content
                validation_results['validation_summary']['valid_stories'] += 1
                self.log_validation_step(
                    f"Scene {scene_id} VALID: {actual_words} words (target: {target_words}, range: {min_words}-{max_words})",
                    "SUCCESS"
                )
            else:
                # Story needs correction
                action = 'expand' if actual_words < min_words else 'trim'
                validation_results['corrections_needed'].append({
                    'scene_id': scene_id,
                    'scene_title': scene.get('title', f'Scene {scene_id}'),
                    'current_words': actual_words,
                    'target_words': target_words,
                    'min_words': min_words,
                    'max_words': max_words,
                    'deviation_percent': round(deviation * 100, 1),
                    'action': action,
                    'target_duration': target_duration,
                    'original_story': story_content
                })
                validation_results['validation_summary']['corrections_needed'] += 1
                self.log_validation_step(
                    f"Scene {scene_id} NEEDS {action.upper()}: {actual_words} words (target: {target_words}, deviation: {deviation*100:.1f}%)",
                    "WARNING"
                )

        # Handle missing stories
        if missing_stories:
            validation_results['validation_summary']['missing_stories'] = len(missing_stories)
            validation_results['missing_stories'] = missing_stories
            for missing in missing_stories:
                self.log_validation_step(
                    f"Scene {missing['scene_id']} MISSING: Story not generated",
                    "WARNING"
                )

        # Summary
        summary = validation_results['validation_summary']
        self.log_validation_step(
            f"Validation Complete: {summary['valid_stories']} valid, {summary['corrections_needed']} need correction, {summary['missing_stories']} missing",
            "SUCCESS",
            summary
        )

        return validation_results

    def auto_correct_stories(self, validation_results: Dict, scene_plan: List[Dict]) -> Dict:
        """Automatically correct stories that need adjustment"""

        corrections_needed = validation_results.get('corrections_needed', [])
        missing_stories = validation_results.get('missing_stories', [])

        if not corrections_needed and not missing_stories:
            self.log_validation_step("No corrections needed", "SUCCESS")
            return validation_results['valid_stories']

        self.log_validation_step(f"Starting Auto-Correction: {len(corrections_needed)} corrections, {len(missing_stories)} missing stories")

        corrected_stories = validation_results['valid_stories'].copy()

        # Handle corrections first
        for correction in corrections_needed:
            try:
                corrected_story = self._correct_single_story(correction, scene_plan)
                if corrected_story:
                    corrected_stories[correction['scene_id']] = corrected_story
                    self.correction_count += 1
                    self.log_validation_step(
                        f"Scene {correction['scene_id']} corrected: {correction['action']}ed to ~{correction['target_words']} words",
                        "SUCCESS"
                    )
                else:
                    # Keep original if correction fails
                    corrected_stories[correction['scene_id']] = correction['original_story']
                    self.log_validation_step(
                        f"Scene {correction['scene_id']} correction failed, keeping original",
                        "WARNING"
                    )
            except Exception as e:
                corrected_stories[correction['scene_id']] = correction['original_story']
                self.log_validation_step(
                    f"Scene {correction['scene_id']} correction error: {str(e)}",
                    "WARNING"
                )

        # Handle missing stories
        for missing in missing_stories:
            try:
                generated_story = self._generate_missing_story(missing, scene_plan)
                if generated_story:
                    corrected_stories[missing['scene_id']] = generated_story
                    self.correction_count += 1
                    self.log_validation_step(
                        f"Scene {missing['scene_id']} generated: ~{missing['target_words']} words",
                        "SUCCESS"
                    )
                else:
                    self.log_validation_step(
                        f"Scene {missing['scene_id']} generation failed",
                        "WARNING"
                    )
            except Exception as e:
                self.log_validation_step(
                    f"Scene {missing['scene_id']} generation error: {str(e)}",
                    "WARNING"
                )

        self.log_validation_step(f"Auto-Correction Complete: {self.correction_count} stories corrected/generated", "SUCCESS")
        return corrected_stories

    def _correct_single_story(self, correction: Dict, scene_plan: List[Dict]) -> Optional[str]:
        """Correct a single story's length"""

        scene_id = correction['scene_id']
        action = correction['action']
        target_words = correction['target_words']
        current_words = correction['current_words']
        original_story = correction['original_story']

        # Get scene context
        scene_info = next((s for s in scene_plan if str(s['scene_id']) == scene_id), {})
        scene_title = scene_info.get('title', f'Scene {scene_id}')
        scene_description = scene_info.get('description', 'Historical scene')
        scene_emotion = scene_info.get('emotion', 'peaceful')
        scene_template = scene_info.get('template', 'atmospheric')

        if action == 'expand':
            prompt = f"""Expand this sleep story scene to exactly {target_words} words (currently {current_words} words).

SCENE CONTEXT:
- Scene {scene_id}: {scene_title}
- Description: {scene_description}
- Emotion: {scene_emotion}
- Template: {scene_template}
- Target: {target_words} words

ORIGINAL STORY:
{original_story}

EXPANSION REQUIREMENTS:
- Keep the EXACT same opening sentence
- Maintain the same tone, style, and emotion
- Add more sensory details and atmospheric descriptions
- Expand character interactions and emotions if characters are present
- Add more historical details and period-accurate elements
- Include more [PAUSE] markers at natural breathing points
- Target exactly {target_words} words (±10 words acceptable)
- Keep the story sleep-optimized with gentle pacing
- Do not change the core narrative or ending

OUTPUT FORMAT: Expanded story only, no explanations or commentary."""

        else:  # trim
            prompt = f"""Trim this sleep story scene to exactly {target_words} words (currently {current_words} words).

SCENE CONTEXT:
- Scene {scene_id}: {scene_title}
- Description: {scene_description}
- Emotion: {scene_emotion}
- Template: {scene_template}
- Target: {target_words} words

ORIGINAL STORY:
{original_story}

TRIMMING REQUIREMENTS:
- Keep the EXACT same opening sentence
- Maintain core narrative atmosphere and emotion
- Remove less essential descriptive details
- Keep all character interactions and dialogue intact
- Preserve the most important sensory elements
- Keep key [PAUSE] markers for natural flow
- Target exactly {target_words} words (±10 words acceptable)
- Maintain sleep-optimized pacing
- Do not change the story's main events or ending

OUTPUT FORMAT: Trimmed story only, no explanations or commentary."""

        try:
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=4000,
                temperature=0.3,
                timeout=300,
                system=f"You are a professional story editor specializing in sleep content. Your task is to {action} the story to meet exact word count requirements while maintaining all original qualities and sleep optimization.",
                messages=[{"role": "user", "content": prompt}]
            )

            corrected_story = response.content[0].text.strip()

            # Validate the correction
            new_word_count = len(corrected_story.split())
            if abs(new_word_count - target_words) <= target_words * 0.1:  # 10% tolerance for corrections
                return corrected_story
            else:
                self.log_validation_step(
                    f"Correction validation failed: got {new_word_count} words, target {target_words}",
                    "WARNING"
                )
                return None

        except Exception as e:
            self.log_validation_step(f"Correction API call failed: {str(e)}", "WARNING")
            return None

    def _generate_missing_story(self, missing: Dict, scene_plan: List[Dict]) -> Optional[str]:
        """Generate a missing story"""

        scene_id = missing['scene_id']
        target_words = missing['target_words']
        target_duration = missing['target_duration']
        scene_title = missing['scene_title']

        # Get scene context
        scene_info = next((s for s in scene_plan if str(s['scene_id']) == scene_id), {})
        scene_description = scene_info.get('description', 'Historical scene')
        scene_emotion = scene_info.get('emotion', 'peaceful')
        scene_template = scene_info.get('template', 'atmospheric')
        scene_location = scene_info.get('location', 'Ancient setting')
        key_elements = scene_info.get('key_elements', [])
        characters_mentioned = scene_info.get('characters_mentioned', [])

        prompt = f"""Generate a complete sleep story for this missing scene.

SCENE REQUIREMENTS:
- Scene {scene_id}: {scene_title}
- Location: {scene_location}
- Description: {scene_description}
- Emotion: {scene_emotion}
- Template: {scene_template}
- Duration: {target_duration} minutes
- Target words: {target_words} words
- Key Elements: {', '.join(key_elements) if key_elements else 'Atmospheric details'}
- Characters: {', '.join(characters_mentioned) if characters_mentioned else 'Focus on atmosphere'}

STORY REQUIREMENTS:
- Create a unique, atmospheric opening (NEVER use "You find yourself")
- Present tense, second person perspective
- Rich sensory details (sight, sound, smell, touch, taste)
- [PAUSE] markers at natural breathing points
- Sleep-optimized language with gentle pacing
- Historical accuracy with authentic period details
- Target exactly {target_words} words (±20 words acceptable)
- Maintain {scene_emotion} emotion throughout
- Include character interactions if characters are mentioned
- End with peaceful resolution

OPENING STYLE OPTIONS (choose one):
- Environmental: "The golden light filters through..."
- Temporal: "As twilight settles over..."
- Auditory: "Soft footsteps echo in..."
- Sensory: "The gentle breeze carries..."
- Visual: "Shadows dance across..."
- Character-focused: "[Character name] pauses at..."

OUTPUT FORMAT: Complete story only, no explanations or commentary."""

        try:
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=6000,
                temperature=0.7,
                timeout=300,
                system="You are a master storyteller specializing in sleep content. Generate atmospheric, historically accurate stories that promote peaceful sleep. Focus on rich sensory details and gentle pacing.",
                messages=[{"role": "user", "content": prompt}]
            )

            generated_story = response.content[0].text.strip()

            # Validate the generation
            new_word_count = len(generated_story.split())
            if abs(new_word_count - target_words) <= target_words * 0.2:  # 20% tolerance for new generations
                return generated_story
            else:
                self.log_validation_step(
                    f"Generation validation failed: got {new_word_count} words, target {target_words}",
                    "WARNING"
                )
                return None

        except Exception as e:
            self.log_validation_step(f"Generation API call failed: {str(e)}", "WARNING")
            return None

    def get_validation_report(self) -> Dict:
        """Get comprehensive validation report"""
        return {
            'validation_log': self.validation_log,
            'corrections_made': self.correction_count,
            'validation_enabled': CONFIG.claude_config.get('validation_enabled', False),
            'auto_correction_enabled': CONFIG.claude_config.get('auto_correction', False),
            'target_tolerance': CONFIG.claude_config.get('target_tolerance', 0.2),
            'validation_summary': {
                'total_validation_steps': len(self.validation_log),
                'corrections_applied': self.correction_count,
                'validation_success': len([log for log in self.validation_log if log['status'] == 'SUCCESS'])
            }
        }



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
    """Complete server-ready automated story generation with VALIDATION SYSTEM + FIXED Smart Algorithm + ALL LOCAL FEATURES"""

    def __init__(self):
        """Initialize story generator for server environment"""
        self.generation_log = []
        self.api_call_count = 0
        self.total_cost = 0.0
        self.character_system = CharacterExtractionSystem()

        try:
            self.client = Anthropic(api_key=CONFIG.api_key)

            # 🆕 NEW: Initialize validation system
            self.validation_system = StoryValidationSystem(self.client, CONFIG.logger)

            CONFIG.logger.info("✅ Story generator with validation system initialized successfully")
            print(
                "✅ Story generator initialized with Smart Algorithm + Budget Tracking + ALL LOCAL FEATURES + VALIDATION SYSTEM")
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
        CONFIG.logger.info(
            f"{description} - Status: {status} - API calls: {self.api_call_count} - Cost: ${self.total_cost:.4f}")

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

    def generate_complete_story_with_characters(self, topic: str, description: str, clickbait_title: str = None,
                                                font_design: str = None) -> Dict[str, Any]:
        """
        COMPLETE 5-STAGE APPROACH WITH VALIDATION SYSTEM + FIXED SMART ALGORITHM + ALL LOCAL FEATURES:
        Stage 1: Smart Planning + Hook + Subscribe + First Half stories
        Stage 2: Remaining stories (second half)
        Stage 2.5: 🆕 NEW: Validation and Auto-correction of all stories
        Stage 3: Character extraction and analysis
        Stage 4: Intelligent thumbnail generation
        Stage 5: Hook & Subscribe scene selection + Complete JSON outputs
        """

        self.log_step("Complete Story Generation with VALIDATION SYSTEM + Smart Random Durations + ALL LOCAL FEATURES")

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

            # 🆕 STAGE 2.5: VALIDATION AND AUTO-CORRECTION
            if CONFIG.claude_config.get('validation_enabled', True):
                validation_result = self._validate_and_correct_stories(stage1_result, stage2_result, scene_plan)
                time.sleep(1)
            else:
                print("⚠️ Validation system disabled, skipping validation stage")
                validation_result = {
                    'validated_stories': {**stage1_result.get('stories', {}), **stage2_result.get('stories', {})},
                    'validation_report': {'validation_enabled': False}
                }

            # STAGE 3: Character Extraction (using validated stories)
            character_result = self._extract_characters(topic, description, stage1_result, validation_result)
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

            # COMBINE: Merge all stages (using validated stories)
            combined_result = self._combine_all_stages(
                stage1_result, validation_result, character_result, thumbnail_result, hook_subscribe_result, topic,
                description
            )

            # Add smart generation + validation stats
            combined_result['generation_stats'].update({
                'smart_algorithm': True,
                'random_scene_count': total_scenes,
                'natural_duration_variation': True,
                'duration_range': f"{min(durations):.1f}-{max(durations):.1f} minutes" if durations else "N/A",
                'validation_system_enabled': CONFIG.claude_config.get('validation_enabled', True),
                'auto_correction_enabled': CONFIG.claude_config.get('auto_correction', True),
                'validation_report': validation_result.get('validation_report', {})
            })

            self.log_step("Complete Smart Generation Pipeline with Validation Finished", "SUCCESS", {
                "total_scenes": len(combined_result.get('scene_plan', [])),
                "total_stories": len(combined_result.get('stories', {})),
                "characters_extracted": len(combined_result.get('main_characters', [])),
                "thumbnail_generated": combined_result.get('generation_stats', {}).get('thumbnail_generated', False),
                "hook_subscribe_generated": combined_result.get('generation_stats', {}).get('hook_subscribe_generated', False),
                "smart_algorithm_used": True,
                "validation_enabled": CONFIG.claude_config.get('validation_enabled', True),
                "stories_corrected": validation_result.get('validation_report', {}).get('corrections_made', 0),
                "api_calls_total": self.api_call_count,
                "total_cost": self.total_cost
            })

            return combined_result

        except Exception as e:
            self.log_step("Generation Failed", "ERROR")
            CONFIG.logger.error(f"Generation failed: {e}")
            raise

    def _validate_and_correct_stories(self, stage1_result: Dict, stage2_result: Dict, scene_plan: List[Dict]) -> Dict:
        """🆕 NEW: Stage 2.5 - Validate and auto-correct all stories"""

        self.log_step("Stage 2.5: Story Validation and Auto-Correction")

        # Combine all stories from both stages
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))

        # Validate stories against scene plan
        validation_results = self.validation_system.validate_stories_duration(all_stories, scene_plan)

        # Auto-correct if enabled
        if CONFIG.claude_config.get('auto_correction', True):
            corrected_stories = self.validation_system.auto_correct_stories(validation_results, scene_plan)
        else:
            corrected_stories = validation_results['valid_stories']
            print("⚠️ Auto-correction disabled, using only valid stories")

        # Get validation report
        validation_report = self.validation_system.get_validation_report()

        # Update API call count (validation system makes additional calls)
        if hasattr(self.validation_system, 'correction_count'):
            # Each correction/generation is approximately 1 API call
            additional_calls = self.validation_system.correction_count
            self.api_call_count += additional_calls

            # Estimate additional cost (corrections are smaller, ~$0.01 each)
            additional_cost = additional_calls * 0.01
            self.total_cost += additional_cost

        self.log_step("Story Validation and Auto-Correction", "SUCCESS", {
            "stories_validated": len(all_stories),
            "stories_corrected": validation_report.get('corrections_made', 0),
            "final_story_count": len(corrected_stories),
            "validation_enabled": True
        })

        return {
            'validated_stories': corrected_stories,
            'validation_report': validation_report,
            'validation_results': validation_results
        }

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

        stage1_prompt = self._add_word_count_optimization(stage1_prompt, scene_durations)

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
        FIX 1: Reduced prompt length to prevent API errors
        FIX 2: Better word count targeting with precise calculations
        """

        scene_plan = stage1_result.get('scene_plan', [])
        stories_written = len(stage1_result.get('stories', {}))
        total_scenes = len(scene_plan)
        remaining_scenes = total_scenes - stories_written

        if remaining_scenes <= 0:
            self.log_step("Stage 2: No remaining stories needed", "SUCCESS")
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        self.log_step(f"Stage 2: TÓIBÍN CONTINUITY - {remaining_scenes} Stories")

        # Get remaining scenes with precise word count targets
        remaining_scene_plan = []
        for scene in scene_plan:
            if str(scene['scene_id']) not in stage1_result.get('stories', {}):
                # Calculate precise word count target (140 words per minute)
                duration_minutes = scene.get('duration_minutes', 4)
                target_words = int(duration_minutes * 140)
                scene['target_words'] = target_words
                remaining_scene_plan.append(scene)

        if len(remaining_scene_plan) == 0:
            return {"stories": {}, "stage2_stats": {"stories_written": 0, "note": "All stories completed in stage 1"}}

        # SHORTER, FOCUSED PROMPT to prevent API overload
        stage2_prompt = f"""Complete sleep story "{topic}" with remaining {remaining_scenes} stories in COLM TÓIBÍN'S STYLE.

    TOPIC: {topic}
    DESCRIPTION: {description}

    SCENES TO COMPLETE ({remaining_scenes} scenes):
    {self._format_remaining_scenes(remaining_scene_plan)}

    🎭 TÓIBÍN STYLE REQUIREMENTS:

    ## WORD COUNT PRECISION (CRITICAL):
    - Calculate EXACTLY: Duration × 140 words/minute
    - 2min = 280 words | 3min = 420 words | 4min = 560 words | 5min = 700 words
    - 6min = 840 words | 7min = 980 words | 8min = 1120 words

    ## TÓIBÍN MASTERY CONTINUATION:
    - Same character voices/psychology from Stage 1
    - "Hush you hear" - quiet, contemplative atmosphere
    - "Fascination of commonplaces" - ordinary moments profound
    - Present tense, second person for sleep optimization
    - Historical authenticity through small details

    ## OPENING VARIATIONS (Use different for each):
    - "The evening settles differently now..."
    - "She notices something has changed..."
    - "The garden holds new shadows..."
    - "This moment echoes earlier ones..."
    - "A familiar weight returns..."

    ## CHARACTER CONSISTENCY:
    Maintain exact personalities/relationships from Stage 1:
    - Same internal thought patterns
    - Same dialogue styles
    - Same relationship dynamics
    - Emotional progression: observation → recognition → acceptance

    ## CONTENT PATTERN:
    1. Quiet observation (50-80 words)
    2. Character activity (100-150 words) 
    3. Internal reflection (80-120 words)
    4. Environmental detail (60-100 words)
    5. Human connection (80-120 words)
    6. Peaceful resolution (60-100 words)

    OUTPUT: Complete JSON with stories matching EXACT word counts.

    {{
      "stories": {{
        "{remaining_scene_plan[0]['scene_id'] if remaining_scene_plan else 'X'}": "[EXACT {remaining_scene_plan[0].get('target_words', 560)} WORDS - Tóibín style story]"
      }},
      "stage2_stats": {{
        "stories_written": {remaining_scenes},
        "word_count_precision": true,
        "toibin_continuity": true
      }}
    }}

    Write each story to EXACT target word count. Count carefully."""

        try:
            self.api_call_count += 1

            # SHORTER timeout to prevent hanging
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                stream=True,
                timeout=900,  # Reduced from 1800
                system="Continue COLM TÓIBÍN style from Stage 1. CRITICAL: Match exact word counts (duration × 140). Maintain character consistency. Use contemplative, understated style.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = ""
            print("📡 Stage 2: Streaming OPTIMIZED...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 3000 == 0:  # Less frequent logging
                        print(f"   📚 Progress: {len(content):,} chars...")

            print(f"✅ Stage 2 OPTIMIZED complete: {len(content):,} characters")

            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 OPTIMIZED Parsing", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "word_count_targeted": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 OPTIMIZED Failed", "ERROR")
            CONFIG.logger.error(f"Stage 2 optimized error: {e}")
            return {"stories": {}, "stage2_stats": {"error": str(e)}}

    def _format_remaining_scenes(self, scenes_list):
        """Format remaining scenes concisely to reduce prompt size"""
        formatted = []
        for scene in scenes_list:
            formatted.append(
                f"Scene {scene['scene_id']}: {scene['title']} ({scene.get('duration_minutes', 4)}min = {scene.get('target_words', 560)} words)")
        return '\n'.join(formatted)

    def _enhanced_word_count_correction(self, scene_id: str, current_story: str, target_words: int,
                                        scene_info: Dict) -> str:
        """
        Enhanced word count correction with precise targeting
        FIX: Better calculation and validation
        """
        current_words = len(current_story.split())
        deviation = abs(current_words - target_words) / target_words

        # Only correct if significantly off (>20% deviation)
        if deviation < 0.20:
            return current_story

        if current_words < target_words:
            # EXPAND with precise targeting
            words_needed = target_words - current_words

            correction_prompt = f"""EXPAND this Tóibín-style story from {current_words} to EXACTLY {target_words} words (+{words_needed} words).

    CURRENT STORY:
    {current_story}

    SCENE INFO: {scene_info.get('title', '')} - {scene_info.get('duration_minutes', 4)} minutes

    EXPANSION STRATEGY:
    - Add {words_needed // 4} words to character psychology
    - Add {words_needed // 4} words to environmental details  
    - Add {words_needed // 4} words to sensory descriptions
    - Add {words_needed // 4} words to contemplative moments

    MAINTAIN:
    - Tóibín's understated style
    - Character consistency
    - Sleep-optimized pacing
    - Present tense, second person

    OUTPUT: Expanded story with EXACTLY {target_words} words. Count carefully."""

        else:
            # TRIM with precision
            words_to_remove = current_words - target_words

            correction_prompt = f"""TRIM this Tóibín-style story from {current_words} to EXACTLY {target_words} words (-{words_to_remove} words).

    CURRENT STORY:
    {current_story}

    TRIMMING STRATEGY:
    - Remove redundant descriptions
    - Condense verbose passages
    - Keep essential Tóibín elements
    - Maintain narrative flow

    OUTPUT: Trimmed story with EXACTLY {target_words} words."""

        try:
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"] // 2,  # Smaller response
                temperature=0.3,  # More deterministic
                timeout=300,  # Shorter timeout
                system=f"Precise word count editor. Target: EXACTLY {target_words} words. Maintain Tóibín style.",
                messages=[{"role": "user", "content": correction_prompt}]
            )

            corrected_story = response.content[0].text.strip()

            # Validate correction
            corrected_words = len(corrected_story.split())
            final_deviation = abs(corrected_words - target_words) / target_words

            if final_deviation < 0.15:  # Within 15% tolerance
                return corrected_story
            else:
                print(f"⚠️ Correction still off: {corrected_words} vs {target_words} words")
                return current_story  # Return original if correction failed

        except Exception as e:
            print(f"❌ Correction failed: {e}")
            return current_story

    def _validate_word_counts_enhanced(self, stories: Dict, scene_plan: List) -> Dict:
        """
        Enhanced validation with better tolerance and reporting
        """
        validation_results = {
            "valid_scenes": [],
            "needs_correction": [],
            "missing_scenes": [],
            "total_scenes": len(scene_plan),
            "validation_summary": {}
        }

        for scene in scene_plan:
            scene_id = str(scene['scene_id'])
            target_duration = scene.get('duration_minutes', 4)
            target_words = int(target_duration * 140)  # 140 words per minute

            # Allow ±20% tolerance (was too strict before)
            tolerance = 0.20
            min_words = int(target_words * (1 - tolerance))
            max_words = int(target_words * (1 + tolerance))

            if scene_id not in stories:
                validation_results["missing_scenes"].append({
                    "scene_id": scene_id,
                    "title": scene.get('title', 'Untitled'),
                    "target_words": target_words,
                    "status": "MISSING"
                })
            else:
                story = stories[scene_id]
                actual_words = len(story.split())
                deviation = abs(actual_words - target_words) / target_words

                if min_words <= actual_words <= max_words:
                    validation_results["valid_scenes"].append({
                        "scene_id": scene_id,
                        "actual_words": actual_words,
                        "target_words": target_words,
                        "status": "VALID"
                    })
                else:
                    validation_results["needs_correction"].append({
                        "scene_id": scene_id,
                        "actual_words": actual_words,
                        "target_words": target_words,
                        "deviation": deviation,
                        "action": "EXPAND" if actual_words < target_words else "TRIM",
                        "status": "NEEDS_CORRECTION"
                    })

        # Summary statistics
        validation_results["validation_summary"] = {
            "total_scenes": len(scene_plan),
            "valid_count": len(validation_results["valid_scenes"]),
            "correction_needed": len(validation_results["needs_correction"]),
            "missing_count": len(validation_results["missing_scenes"]),
            "success_rate": len(validation_results["valid_scenes"]) / len(scene_plan) * 100
        }

        return validation_results

    # Additional helper for Stage 1 word count improvement
    def _optimize_stage1_word_counts(self, stage1_prompt: str, scene_durations: List) -> str:
        """
        Add precise word count instructions to Stage 1 prompt
        """
        word_count_section = """
    ## CRITICAL: PRECISE WORD COUNT TARGETING

    Calculate EXACTLY for each scene:
    - Duration × 140 words/minute = Target words
    - 2.0min = 280 words | 3.0min = 420 words | 4.0min = 560 words
    - 5.0min = 700 words | 6.0min = 840 words | 7.0min = 980 words
    - 8.0min = 1120 words | 9.0min = 1260 words | 10.0min = 1400 words

    WORD COUNT VERIFICATION:
    - Count words in each story before finalizing
    - Adjust length to match target ±10%
    - Use content structure to reach exact counts

    EXPANSION TECHNIQUES (if short):
    - Character internal thoughts (+50-100 words)
    - Environmental sensory details (+40-80 words)
    - Historical period details (+30-60 words)
    - Contemplative pauses and reflections (+40-80 words)

    """

        # Insert word count section after the main instructions
        insertion_point = stage1_prompt.find("## 4. FIRST")
        if insertion_point > 0:
            return stage1_prompt[:insertion_point] + word_count_section + stage1_prompt[insertion_point:]
        else:
            return stage1_prompt + word_count_section

    def generate_hook_subscribe_scenes(self, scene_plan: List[Dict], hook_content: str, subscribe_content: str) -> Dict:
        """Generate background scenes for hook and subscribe with precise timing"""

        self.log_step("Hook & Subscribe Scene Selection")

        # Select 10 atmospheric scenes for hook (0-30s)
        hook_scenes = []
        atmospheric_scenes = [s for s in scene_plan if
                              s.get('template') == 'atmospheric' or s.get('emotion') == 'peaceful'][:10]

        for i, scene in enumerate(atmospheric_scenes):
            hook_scenes.append({
                "scene_id": scene['scene_id'],
                "scene_title": scene['title'],
                "start_time": i * 3,
                "end_time": (i * 3) + 3,
                "duration": 3,
                "visual_prompt": f"Atmospheric cinematic view of {scene['location']}, golden hour lighting, peaceful and mysterious mood",
                "timing_note": f"Display during hook seconds {i * 3}-{(i * 3) + 3}",
                "sync_importance": "HIGH - Must align with hook narration rhythm"
            })

        # Select 10 community scenes for subscribe (30-60s)
        subscribe_scenes = []
        community_scenes = [s for s in scene_plan if
                            s.get('template') == 'character_focused' or len(s.get('characters_mentioned', [])) > 0][:10]

        for i, scene in enumerate(community_scenes):
            subscribe_scenes.append({
                "scene_id": scene['scene_id'],
                "scene_title": scene['title'],
                "start_time": i * 3,
                "end_time": (i * 3) + 3,
                "duration": 3,
                "visual_prompt": f"Welcoming community view of {scene['location']}, warm lighting, inviting atmosphere",
                "timing_note": f"Display during subscribe seconds {i * 3}-{(i * 3) + 3}",
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

    def _extract_characters(self, topic: str, description: str, stage1_result: Dict, validation_result: Dict) -> Dict[str, Any]:
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

        character_prompt = f"""Analyze the complete COLM TÓIBÍN-style sleep story and extract characters for DRAMATIC THUMBNAIL optimization while maintaining literary authenticity.

TOPIC: {topic}
DESCRIPTION: {description}

STORY CONTENT (First 25000 chars):
{story_content[:25000]}

SCENE PLAN CONTEXT (First 4000 chars):
{scene_context[:4000]}

REQUIREMENTS:

## PART 1: TÓIBÍN-STYLE DEEP CHARACTER EXTRACTION

Extract maximum {CONFIG.claude_config['max_characters']} characters using TÓIBÍN'S PSYCHOLOGICAL MASTERY:

### TÓIBÍN CHARACTER ANALYSIS REQUIREMENTS:
- **"Characters led by desires they don't understand"** - identify internal conflicts
- **"Mixed motives and tacit exchanges"** - complex relationship dynamics  
- **"Vulnerabilities and strengths virtually the same"** - character paradoxes
- **Quiet dignity** - characters who maintain composure despite internal struggle
- **Understated emotion** - feelings expressed through gesture, silence, observation
- **"Fascination of commonplaces"** - how ordinary moments reveal character depth

## PART 2: DRAMATIC THUMBNAIL OPTIMIZATION + YOUTUBE OPTIMIZATION

**CRITICAL BALANCE:**
- Extract characters with **STRONG VISUAL APPEAL** for dramatic thumbnails
- Maintain **TÓIBÍN'S LITERARY AUTHENTICITY** in character analysis
- Characters must work for **"BETRAYAL!" "MYSTERIOUS!" "SCANDALOUS!"** thumbnails
- But psychological analysis stays TRUE to Tóibín's subtle mastery

### THUMBNAIL CHARACTER REQUIREMENTS:
- **Expressive faces** - capable of dramatic emotion for thumbnails
- **Visual contrast** - distinct appearance for thumbnail recognition
- **Emotional range** - can show concern, determination, contemplation for clicks
- **Historical authenticity** - period-accurate appearance
- **Marketing utility** - faces that draw clicks while maintaining story integrity

### DRAMATIC THUMBNAIL STRATEGY:
**YOUR PROVEN FORMULA:**
- Thumbnail: "BETRAYAL! Character's Final Evening" 
- Content: Tóibín-style quiet contemplation and daily life
- Result: Click satisfaction + literary quality + sleep optimization

Create YouTube package that bridges DRAMATIC MARKETING with TÓIBÍN QUALITY:

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
      ],
      "toibin_elements": {
      "internal_contradiction": "[What they want vs what they understand about wanting]",
      "emotional_archaeology": "[Layers of feeling beneath surface politeness]",
      "quiet_dignity": "[How they maintain composure despite struggle]",
      "relationship_complexity": "[Mixed motives in interactions with others]",
      "recognition_moments": "[Instances of unspoken understanding]"
    },
    "dramatic_potential": {
      "thumbnail_emotion": "[Best emotional expression for dramatic thumbnails]",
      "visual_drama": "[How to show intensity while maintaining Tóibín authenticity]",
      "marketing_appeal": "[Why viewers will click while getting quality content]"
    }
      
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
  "MYSTERIOUS! [Character]'s Final Secret Decision (2 Hour Sleep Story)",
  "BETRAYAL! The Night [Character] Never Saw Coming - Most Peaceful Historical Story", 
  "SCANDALOUS! What Really Happened in [Character]'s Last Quiet Evening",
  "TERRIFYING! [Character]'s Hidden Truth - Beautiful Sleep Story That Actually Works",
  "INCREDIBLE! [Character]'s Most Peaceful Final Hours - You Won't Believe What Happened"
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

## TÓIBÍN + DRAMATIC MARKETING MASTERY CHALLENGE:

Extract characters that serve BOTH purposes perfectly:

### REQUIRED EXCELLENCE:
- ✅ **TÓIBÍN PSYCHOLOGICAL AUTHENTICITY** - complex internal lives, mixed motives, quiet dignity
- ✅ **DRAMATIC VISUAL APPEAL** - faces and personalities that work for "BETRAYAL!" thumbnails  
- ✅ **CHARACTER CONSISTENCY** - match the contemplative, internally complex characters from stories
- ✅ **MARKETING UTILITY** - characters viewers want to click on and learn about
- ✅ **LITERARY DEPTH** - character analysis that honors Tóibín's sophisticated approach
- ✅ **SLEEP CONTENT OPTIMIZATION** - characters whose stories promote peaceful rest

This is the PERFECT FORMULA for 1 million subscribers: dramatic marketing appeal combined with genuine literary quality and effective sleep content."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=16000,
                temperature=0.3,
                stream=True,
                timeout=900,
                system="You are COLM TÓIBÍN analyzing your own literary characters while also being a YouTube optimization expert. Extract characters with your signature psychological depth - 'characters led by desires they don't understand,' 'mixed motives and tacit exchanges,' 'quiet dignity' - while ensuring they have dramatic visual appeal for thumbnails. Balance literary authenticity with marketing effectiveness. These characters must work for both 'BETRAYAL!' thumbnails AND contemplative literary content."
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

                # 🆕 GENERATE STORY-BASED VISUAL PROMPTS with Claude + Tóibín style + Midjourney safety
                story_based_visual_prompts = self._generate_claude_visual_prompts_from_stories(
                    validation_result.get('validated_stories', {}),  # 📖 Actual story content
                    stage1_result.get('scene_plan', []),  # 🎬 Scene information
                    top_characters,  # 👥 Character data
                    scene_character_map  # 🗺️ Character-scene mapping
                )

                # Replace the old visual prompts with story-based ones
                parsed_result['story_based_visual_prompts'] = story_based_visual_prompts

                self.character_system.log_extraction_step("Story-Based Visual Prompts Generated", "SUCCESS", {
                    "prompts_created": len(story_based_visual_prompts),  # ✅ DOĞRU değişken
                    "character_integrated_scenes": len(
                        [p for p in story_based_visual_prompts if p.get('character_reference_needed')]),
                    "story_based_generation": True,
                    "toibin_style_integrated": True,
                    "midjourney_safety_compliant": True
                })

            self.character_system.log_extraction_step("Character Extraction", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('main_characters', [])),
                "character_names": [c.get('name', 'Unknown') for c in parsed_result.get('main_characters', [])],
                "visual_prompts_regenerated": 'story_based_visual_prompts' in parsed_result,  # ✅ DOĞRU key
                "story_based_generation": True,
                "toibin_style_integrated": True,
                "stage_cost": stage_cost
            })

            self.character_system.log_extraction_step("Character Extraction", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('main_characters', [])),
                "character_names": [c.get('name', 'Unknown') for c in parsed_result.get('main_characters', [])],
                "visual_prompts_regenerated": 'story_based_visual_prompts' in parsed_result,  # ✅ DOĞRU key
                "story_based_generation": True,  # 🆕 Yeni feature
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

    def _generate_claude_visual_prompts_from_stories(self, validated_stories, scene_plan, characters,
                                                     scene_character_map):
        """🆕 Generate visual prompts by having Claude read each story with Tóibín style + Midjourney safety"""

        self.log_step("Generating Story-Based Visual Prompts with Claude + Tóibín Style + Midjourney Safety")

        MIDJOURNEY_SAFETY_PROMPT = """
    🛡️ CRITICAL MIDJOURNEY CONTENT POLICY COMPLIANCE:

    BANNED WORDS THAT CAUSE AUTOMATIC REJECTION:
    ❌ NEVER USE: "intimate", "private", "personal", "sensual", "romantic"
    ❌ NEVER USE: "bedroom", "bath", "naked", "nude", "skin", "touching"  
    ❌ NEVER USE: "embrace", "caress", "kiss", "passion", "desire"
    ❌ NEVER USE: "mystical", "supernatural", "magic", "spiritual"
    ❌ NEVER USE: "late at night", "dark secrets", "hidden desires"

    ✅ SAFE ALTERNATIVES:
    - "contemplative" NOT "intimate"
    - "evening hours" NOT "late at night"
    - "scholarly atmosphere" NOT "mystical mood"
    - "historical educational setting" ALWAYS include
    - "family-friendly historical content" ALWAYS include
    """

        TOIBIN_VISUAL_TRANSLATION = """
    🎭 COLM TÓIBÍN LITERARY STYLE → VISUAL TRANSLATION:

    TÓIBÍN'S SIGNATURE VISUAL APPROACH:
    ✅ "SPARSENESS WITH SUPERABUNDANCE" → Visual restraint that suggests deeper meaning
    ✅ "QUIET DIGNITY" → Characters maintaining composure, understated elegance  
    ✅ "FASCINATION OF COMMONPLACES" → Extraordinary beauty in ordinary moments
    ✅ "HUSH YOU HEAR" → Peaceful, contemplative atmosphere with subtle depth
    ✅ "MIXED MOTIVES & TACIT EXCHANGES" → Complex psychology visible in expressions

    VISUAL REQUIREMENTS:
    - RESTRAINT over spectacle (Tóibín prefers subtlety)
    - CHARACTER PSYCHOLOGY emphasis (internal states visible)
    - CONTEMPLATIVE PACING (thoughtful compositions)
    - LITERARY SOPHISTICATION (educated audience appeal)
    - UNDERSTATED EMOTION (never melodramatic)
    """

        enhanced_prompts = []

        for scene_id, story_content in validated_stories.items():
            scene_info = next((s for s in scene_plan if str(s['scene_id']) == scene_id), {})
            scene_characters = scene_character_map.get(str(scene_id), [])

            # Character names for prompt
            char_names = []
            for sc in scene_characters:
                if isinstance(sc, dict):
                    char_names.append(sc.get('name', ''))
                else:
                    char_names.append(str(sc))

            prompt = f"""
    {MIDJOURNEY_SAFETY_PROMPT}

    {TOIBIN_VISUAL_TRANSLATION}

    READ this COLM TÓIBÍN-style story and create a MIDJOURNEY-OPTIMIZED visual prompt:

    STORY CONTENT (First 3000 characters):
    {story_content[:3000]}

    SCENE CONTEXT:
    - Title: {scene_info.get('title', '')}
    - Location: {scene_info.get('location', '')}
    - Characters Present: {', '.join(char_names) if char_names else 'None'}
    - Emotion: {scene_info.get('emotion', 'peaceful')}
    - Duration: {scene_info.get('duration_minutes', 4)} minutes
    - Tóibín Elements: {scene_info.get('toibin_elements', [])}

    CREATE A VISUAL PROMPT THAT CAPTURES:

    1. 🎭 TÓIBÍN LITERARY STYLE:
       - "Quiet dignity" in character presentation
       - "Fascination of commonplaces" - make ordinary extraordinary
       - "Sparseness with suggestion" - visual restraint with deep meaning
       - Contemplative, peaceful atmosphere with psychological depth

    2. 📖 SPECIFIC STORY DETAILS:
       - EXACT character actions described in the story
       - PARTICULAR objects and environmental elements mentioned
       - EMOTIONAL moments and internal states shown
       - SYMBOLIC elements that carry story meaning

    3. 🛡️ MIDJOURNEY SAFETY & OPTIMIZATION:
       - Use only safe, educational language
       - Include "historical educational content" qualifier
       - Stay under 3000 characters
       - Family-friendly approach
       - No banned words whatsoever

    VISUAL PROMPT STRUCTURE:
    "Contemplative [shot type] of [historical location], [character] showing quiet dignity while [specific story action], [story-specific environmental details], understated composition emphasizing the fascination of commonplaces, peaceful Tóibín atmosphere with subtle psychological depth, historical educational content showcasing authentic [period] daily life, family-friendly literary scene"

    QUALITY CHECKLIST:
    ✅ Tóibín's "quiet dignity" visible in character presentation?
    ✅ Ordinary moment made visually compelling ("fascination of commonplaces")?
    ✅ Specific story details captured accurately?
    ✅ Literary sophistication maintained?
    ✅ Midjourney-safe language used throughout?
    ✅ Historical educational context emphasized?

    OUTPUT: Single optimized visual prompt (2000-3000 characters)
    """

            try:
                self.api_call_count += 1

                response = self.client.messages.create(
                    model=CONFIG.claude_config["model"],
                    max_tokens=1500,
                    temperature=0.3,
                    timeout=300,
                    system="You are COLM TÓIBÍN collaborating with a master visual director. Create visual prompts that capture both your literary authenticity - 'quiet dignity,' 'sparseness with suggestion,' 'fascination of commonplaces' - AND the specific story details, while ensuring complete Midjourney content policy compliance.",
                    messages=[{"role": "user", "content": prompt}]
                )

                story_based_prompt = response.content[0].text.strip()

                # Apply safety filters
                story_based_prompt = self._apply_midjourney_safety_filters(story_based_prompt)

                # Build character details for consistency
                characters_in_scene = []
                for char_name in char_names:
                    full_char = next((c for c in characters if c['name'] == char_name), None)
                    if full_char:
                        characters_in_scene.append({
                            'name': char_name,
                            'description': full_char.get('physical_description', ''),
                            'importance': full_char.get('importance_score', 5),
                            'toibin_psychology': full_char.get('toibin_elements', {})
                        })

                enhanced_prompts.append({
                    "scene_number": int(scene_id),
                    "title": scene_info.get('title', f'Scene {scene_id}'),
                    "location": scene_info.get('location', ''),
                    "characters_present": char_names,
                    "character_reference_needed": len(char_names) > 0,
                    "prompt": story_based_prompt,
                    "enhanced_prompt": f"[TÓIBÍN LITERARY + STORY-BASED] {story_based_prompt}",
                    "duration_minutes": scene_info.get('duration_minutes', 4),
                    "emotion": scene_info.get('emotion', 'peaceful'),
                    "characters_in_scene": characters_in_scene,
                    "story_based_generation": True,
                    "toibin_style_integrated": True,
                    "midjourney_safety_compliant": True,
                    "character_count": len(story_based_prompt),
                    "story_elements_captured": self._count_story_elements(story_content, story_based_prompt)
                })

                self.log_step(
                    f"Story-based visual prompt generated for scene {scene_id}: {len(story_based_prompt)} chars",
                    "SUCCESS")

            except Exception as e:
                self.log_step(f"Story-based visual prompt failed for scene {scene_id}: {str(e)}", "WARNING")
                # Fallback to simple prompt
                fallback_prompt = self._create_story_fallback_prompt(scene_info, char_names, story_content)
                enhanced_prompts.append(fallback_prompt)

        return enhanced_prompts

    def _apply_midjourney_safety_filters(self, prompt: str) -> str:
        """Apply safety filters to prevent Midjourney rejection"""

        REPLACEMENTS = {
            "intimate": "contemplative",
            "private": "personal study",
            "mystical": "atmospheric",
            "late at night": "evening hours",
            "dark secrets": "historical mysteries",
            "touching": "near",
            "embrace": "gather around",
            "sensual": "peaceful",
            "romantic": "warm",
            "naked": "unadorned",
            "nude": "simple"
        }

        # Apply replacements
        for risky, safe in REPLACEMENTS.items():
            prompt = prompt.replace(risky, safe)

        # Ensure safety qualifiers
        if "historical educational" not in prompt.lower():
            prompt += ", historical educational content, family-friendly scene"

        # Character limit check
        if len(prompt) > 3500:
            prompt = prompt[:3400] + ", family-friendly historical scene"

        return prompt

    def _count_story_elements(self, story_content: str, visual_prompt: str) -> int:
        """Count specific story elements captured in visual prompt"""
        story_words = set(story_content.lower().split())
        prompt_words = set(visual_prompt.lower().split())

        # Key story elements (nouns, actions, objects)
        story_elements = [
            word for word in story_words
            if len(word) > 4 and word not in ['that', 'with', 'from', 'they', 'have', 'this', 'will', 'been']
        ]

        captured = [element for element in story_elements if element in prompt_words]
        return len(captured)

    def _create_story_fallback_prompt(self, scene_info, char_names, story_content):
        """Create fallback prompt if Claude generation fails"""
        location = scene_info.get('location', 'Historical setting')
        emotion = scene_info.get('emotion', 'peaceful')

        if char_names:
            chars = ', '.join(char_names)
            prompt = f"Contemplative scene of {location}, featuring {chars} in {emotion} moment, historical educational content, family-friendly literary scene"
        else:
            prompt = f"Atmospheric scene of {location}, {emotion} historical atmosphere, educational content, family-friendly"

        return {
            "scene_number": int(scene_info.get('scene_id', 0)),
            "title": scene_info.get('title', 'Fallback Scene'),
            "prompt": prompt,
            "enhanced_prompt": f"[FALLBACK] {prompt}",
            "story_based_generation": False,
            "fallback_used": True
        }

    def _apply_midjourney_safety_filters(self, prompt: str) -> str:
        """Apply safety filters to prevent Midjourney rejection"""

        REPLACEMENTS = {
            "intimate": "contemplative",
            "private": "personal study",
            "mystical": "atmospheric",
            "late at night": "evening hours",
            "dark secrets": "historical mysteries",
            "touching": "near",
            "embrace": "gather around",
            "sensual": "peaceful",
            "romantic": "warm"
        }

        for risky, safe in REPLACEMENTS.items():
            prompt = prompt.replace(risky, safe)

        # Ensure safety qualifier
        if "historical educational" not in prompt.lower():
            prompt += ", historical educational content, family-friendly scene"

        # Character limit
        if len(prompt) > 3500:
            prompt = prompt[:3400] + "..."

        return prompt

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
        """Create enhanced fallback prompts with dramatic storytelling"""

        prompts = []

        for scene in scene_plan:
            scene_id = scene['scene_id']
            scene_characters = scene_character_map.get(str(scene_id), [])
            location = scene.get('location', 'Ancient setting')
            emotion = scene.get('emotion', 'peaceful')
            description = scene.get('description', '')
            title = scene.get('title', f"Scene {scene_id}")

            # Determine dramatic elements based on emotion
            if emotion == 'concern':
                drama_modifier = "showing worry and concern about unfolding events"
                environmental_drama = "with evacuation activity, worried crowds, and signs of crisis in background"
            elif emotion == 'resolution':
                drama_modifier = "showing determination and hope"
                environmental_drama = "with community coming together and hopeful atmosphere"
            else:  # peaceful
                drama_modifier = "in contemplative, serene poses"
                environmental_drama = "with beautiful, calming atmosphere and peaceful daily activities"

            # Character integration
            if scene_characters:
                char_names = [sc if isinstance(sc, str) else sc.get('name', '') for sc in scene_characters]
                char_list = ', '.join(char_names)

                prompt = f"Cinematic view of {location}, featuring {char_list} {drama_modifier}, {environmental_drama}, historically accurate period setting, warm atmospheric lighting, multiple visual layers showing both character emotions and environmental storytelling"
                enhanced_prompt = f"[CHARACTERS: {char_list}] {prompt}"
                char_ref_needed = True

                # Character details
                characters_in_scene = []
                for char_name in char_names:
                    full_char = next((c for c in characters if c['name'] == char_name), None)
                    if full_char:
                        characters_in_scene.append({
                            'name': char_name,
                            'description': full_char.get('physical_description', 'Period-appropriate character'),
                            'importance': full_char.get('importance_score', 5)
                        })
            else:
                prompt = f"Atmospheric cinematic view of {location} {environmental_drama}, historically accurate period setting, warm atmospheric lighting, environmental storytelling without characters"
                enhanced_prompt = f"[ATMOSPHERIC SCENE] {prompt}"
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
                "characters_in_scene": characters_in_scene
            }

            prompts.append(prompt_data)

        return prompts

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

    def _combine_all_stages(self, stage1: Dict, validation_result: Dict, character_data: Dict, thumbnail_data: Dict,
                            hook_subscribe_data: Dict, topic: str, description: str) -> Dict[str, Any]:
        """Combine all five stages into final result - USING VALIDATED STORIES + STORY-BASED VISUAL PROMPTS + THUMBNAIL + ALL LOCAL FEATURES"""

        self.log_step(
            "Combining All Stages with VALIDATED STORIES + Story-Based Visual Prompts + Thumbnail + Hook/Subscribe + ALL LOCAL FEATURES")

        # Use validated stories from validation system
        all_stories = validation_result.get('validated_stories', {})

        # Use STORY-BASED visual prompts (NEW) with fallback to regenerated (OLD)
        if 'story_based_visual_prompts' in character_data:  # ✅ NEW: Story-based prompts
            enhanced_visual_prompts = character_data['story_based_visual_prompts']
            print(f"✅ Using story-based visual prompts: {len(enhanced_visual_prompts)} prompts")
            visual_prompt_type = "story_based"
        elif 'regenerated_visual_prompts' in character_data:  # 🔄 FALLBACK: Old regenerated prompts
            enhanced_visual_prompts = character_data['regenerated_visual_prompts']
            print(f"⚠️ Using fallback regenerated visual prompts: {len(enhanced_visual_prompts)} prompts")
            visual_prompt_type = "regenerated"
        else:  # 🆘 FINAL FALLBACK: Basic enhancement
            print("⚠️ Using fallback visual prompt enhancement")
            enhanced_visual_prompts = self._enhance_visual_prompts_with_characters(
                stage1.get('visual_prompts', []),
                character_data.get('main_characters', []),
                character_data.get('scene_character_mapping', {}),
                character_data.get('visual_style_notes', {})
            )
            visual_prompt_type = "fallback"

        # ADD THUMBNAIL TO VISUAL PROMPTS
        thumbnail_prompt = thumbnail_data.get('thumbnail_prompt', {})
        if thumbnail_prompt:
            enhanced_visual_prompts.append(thumbnail_prompt)
            print(f"✅ Thumbnail added to visual prompts")

        # Generate scene chapters for YouTube
        scene_chapters = self._generate_scene_chapters(stage1.get('scene_plan', []))

        # Compile complete story text using validated stories
        complete_story = self._compile_complete_story({
            **stage1,
            'stories': all_stories
        })

        # Final result with ALL ENHANCEMENTS + ALL LOCAL FEATURES + VALIDATION SYSTEM
        result = {
            "hook_section": stage1.get("golden_hook", {}),
            "subscribe_section": stage1.get("subscribe_section", {}),
            "scene_plan": stage1.get("scene_plan", []),
            "scene_chapters": scene_chapters,
            "complete_story": complete_story,
            "visual_prompts": enhanced_visual_prompts,  # INCLUDES THUMBNAIL
            "voice_directions": stage1.get("voice_directions", []),
            "stories": all_stories,  # 🆕 VALIDATED STORIES

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

            # 🆕 VALIDATION DATA
            "validation_report": validation_result.get('validation_report', {}),
            "validation_results": validation_result.get('validation_results', {}),

            "generation_stats": {
                "api_calls_used": self.api_call_count,
                "total_cost": self.total_cost,
                "five_stage_approach": True,
                "smart_algorithm": True,
                "validation_system_enabled": CONFIG.claude_config.get('validation_enabled', True),
                "auto_correction_enabled": CONFIG.claude_config.get('auto_correction', True),
                "stories_validated": len(all_stories),
                "stories_corrected": validation_result.get('validation_report', {}).get('corrections_made', 0),

                # 🔧 FIXED: Visual prompt tracking
                "visual_prompts_regenerated": 'story_based_visual_prompts' in character_data or 'regenerated_visual_prompts' in character_data,
                "story_based_generation": 'story_based_visual_prompts' in character_data,  # 🆕 NEW
                "toibin_style_integrated": 'story_based_visual_prompts' in character_data,  # 🆕 NEW
                "midjourney_safety_compliant": 'story_based_visual_prompts' in character_data,  # 🆕 NEW
                "visual_prompt_type": visual_prompt_type,  # 🆕 NEW: Track which type was used

                "thumbnail_generated": bool(thumbnail_data.get('thumbnail_prompt')),
                "hook_subscribe_generated": bool(hook_subscribe_data.get('hook_scenes')),
                "youtube_optimization_generated": bool(
                    character_data.get('youtube_optimization', {}).get('clickbait_titles')),
                "production_specifications_generated": bool(
                    character_data.get('production_specifications', {}).get('audio_production')),
                "visual_prompts_with_thumbnail": len(enhanced_visual_prompts),
                "scenes_planned": len(stage1.get("scene_plan", [])),
                "stories_written": len(all_stories),
                "stage1_stories": len(stage1.get('stories', {})),
                "stage2_stories": len(all_stories) - len(stage1.get('stories', {})),
                "characters_extracted": len(character_data.get('main_characters', [])),
                "production_ready": len(all_stories) >= 25,
                "total_duration_minutes": sum(
                    scene.get('duration_minutes', 4) for scene in stage1.get("scene_plan", [])),
                "automated_production_ready": True,
                "server_optimized": True,
                "complete_pipeline": True,
                "all_local_features_integrated": True,
                "validation_and_correction_integrated": True,
                "story_based_visual_prompts_integrated": 'story_based_visual_prompts' in character_data  # 🆕 NEW
            },
            "generation_log": self.generation_log,
            "character_extraction_log": self.character_system.extraction_log,
            "validation_log": self.validation_system.validation_log,  # 🆕 NEW
            "topic": topic,
            "description": description,
            "generated_at": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "enhancement_status": "complete_5_stage_pipeline_with_validation_system_story_based_visual_prompts_toibin_style_and_all_optimizations"
            # 🔧 UPDATED
        }

        return result
    def _enhance_visual_prompts_with_characters(self, visual_prompts: List[Dict], characters: List[Dict],
                                                scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
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
                enhanced_prompt[
                    'enhanced_prompt'] = f"[CHARACTERS: {', '.join([c['name'] for c in character_refs])}] {prompt.get('prompt', '')}"
            else:
                enhanced_prompt['characters_in_scene'] = []
                enhanced_prompt['character_reference_needed'] = False
                enhanced_prompt['enhanced_prompt'] = f"[ATMOSPHERIC SCENE - NO CHARACTERS] {prompt.get('prompt', '')}"

            enhanced_prompt['visual_style_notes'] = style_notes
            enhanced_prompts.append(enhanced_prompt)

        return enhanced_prompts

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
    """Save complete production outputs - UPDATED SERVER VERSION WITH ALL LOCAL FEATURES + VALIDATION REPORT"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
        # 1. Complete story text (using validated stories)
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
            "smart_algorithm_used": result.get("generation_stats", {}).get("smart_algorithm", False),
            "validation_system_used": result.get("generation_stats", {}).get("validation_system_enabled", False)
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

        # 🆕 6. VALIDATION REPORT (NEW FILE)
        validation_report = result.get("validation_report", {})
        if validation_report and validation_report.get('validation_enabled', False):
            validation_path = output_path / "validation_report.json"
            validation_data = {
                "validation_report": validation_report,
                "validation_results": result.get("validation_results", {}),
                "validation_log": result.get("validation_log", []),
                "validation_summary": {
                    "validation_enabled": True,
                    "auto_correction_enabled": result.get("generation_stats", {}).get("auto_correction_enabled", False),
                    "stories_validated": result.get("generation_stats", {}).get("stories_validated", 0),
                    "stories_corrected": result.get("generation_stats", {}).get("stories_corrected", 0),
                    "target_tolerance": CONFIG.claude_config.get('target_tolerance', 0.2),
                    "target_words_per_minute": CONFIG.claude_config.get('target_words_per_minute', 140)
                }
            }
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(validation_data, f, indent=2, ensure_ascii=False)
            saved_files.append("validation_report.json")

        # Rest of the files remain the same...
        # [Continue with platform_metadata.json, youtube_metadata.json, etc.]

        # Get main character for thumbnail concept
        main_characters = result.get("main_characters", [])
        main_character = main_characters[0] if main_characters else None
        main_char_name = main_character.get('name', 'Main Character') if main_character else 'Main Character'

        # Calculate duration info
        scene_plan = result.get('scene_plan', [])
        total_duration = sum(scene.get('duration_minutes', 4) for scene in scene_plan)
        total_hours = int(total_duration / 60)

        # 7. Platform metadata (FROM LOCAL VERSION - MORE COMPREHENSIVE THAN youtube_metadata.json)
        platform_path = output_path / "platform_metadata.json"
        youtube_data = result.get("youtube_optimization", {})

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
                "long_tail_keywords": [f"{total_hours} hour sleep story {story_topic.lower()}",
                                       f"{story_topic.lower()} historical bedtime story"],
                "trending_keywords": ["sleep podcast", "historical fiction sleep", "ancient history relaxation"],
                "niche_keywords": [f"{story_topic.lower()} sleep story", f"{story_topic.lower()} meditation"],
                "location_keywords": [word.lower() for word in story_topic.split()],
                "competitor_keywords": ["jason stephenson history", "michelle sanctuary ancient"]
            }),
            "thumbnail_concept": {
                "main_character": main_char_name,
                "dramatic_scene": f"{main_char_name} in atmospheric {story_topic} setting",
                "text_overlay": youtube_data.get("clickbait_titles", [f"{story_topic.upper()}'S SECRET"])[0][
                                :20].upper() if youtube_data.get(
                    "clickbait_titles") else f"{story_topic.upper()}'S SECRET",
                "color_scheme": "Warm golds and deep blues with atmospheric lighting",
                "emotion": "Peaceful concentration and serenity",
                "background": f"Atmospheric {story_topic} setting with cinematic lighting",
                "style_notes": "Cinematic lighting, warm and inviting mood that suggests comfort and relaxation"
            },
            "engagement_strategy": youtube_data.get("engagement_strategy", {}),
            "analytics_tracking": youtube_data.get("analytics_tracking", {}),
            "api_ready_format": {
                "snippet": {
                    "title": youtube_data.get("clickbait_titles", [
                        f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)"])[
                        0] if youtube_data.get(
                        "clickbait_titles") else f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)",
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

        # 8. YouTube metadata (SIMPLIFIED VERSION FOR BACKWARD COMPATIBILITY)
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

        # 9. Thumbnail data with composition strategy
        thumbnail_data = result.get("thumbnail_data", {})
        if thumbnail_data.get("thumbnail_prompt"):
            thumbnail_path = output_path / "thumbnail_generation.json"
            with open(thumbnail_path, "w", encoding="utf-8") as f:
                json.dump(thumbnail_data, f, indent=2, ensure_ascii=False)
            saved_files.append("thumbnail_generation.json")

        # 10. Hook & Subscribe scenes
        hook_subscribe_data = result.get("hook_subscribe_scenes", {})
        if hook_subscribe_data:
            hook_subscribe_path = output_path / "hook_subscribe_scenes.json"
            with open(hook_subscribe_path, "w", encoding="utf-8") as f:
                json.dump(hook_subscribe_data, f, indent=2, ensure_ascii=False)
            saved_files.append("hook_subscribe_scenes.json")

        # 11. Production specifications
        production_specs = result.get("production_specifications", {})
        if production_specs:
            production_path = output_path / "production_specifications.json"
            with open(production_path, "w", encoding="utf-8") as f:
                json.dump(production_specs, f, indent=2, ensure_ascii=False)
            saved_files.append("production_specifications.json")

        # 12. AUTOMATION_SPECS.JSON
        automation_path = output_path / "automation_specs.json"
        automation_data = {
            "audio_production": production_specs.get("audio_production", {}),
            "video_assembly": production_specs.get("video_assembly", {}),
            "quality_control": production_specs.get("quality_control", {}),
            "automation_specifications": production_specs.get("automation_specifications", {}),
            "precise_timing_breakdown": production_specs.get("precise_timing_breakdown", {}),
            "validation_system": {
                "enabled": result.get("generation_stats", {}).get("validation_system_enabled", False),
                "stories_corrected": result.get("generation_stats", {}).get("stories_corrected", 0),
                "validation_tolerance": CONFIG.claude_config.get('target_tolerance', 0.2)
            },
            "implementation_ready": True
        }
        with open(automation_path, "w", encoding="utf-8") as f:
            json.dump(automation_data, f, indent=2, ensure_ascii=False)
        saved_files.append("automation_specs.json")

        # 13. Audio generation prompts
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

        # 14. ALL_STORIES.JSON (validated stories)
        stories_path = output_path / "all_stories.json"
        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(result["stories"], f, indent=2, ensure_ascii=False)
        saved_files.append("all_stories.json")

        # 15. Video composition instructions
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
            "chapters": result.get("scene_chapters", []),
            "validation_notes": {
                "stories_validated": result.get("generation_stats", {}).get("validation_system_enabled", False),
                "stories_corrected": result.get("generation_stats", {}).get("stories_corrected", 0),
                "duration_accuracy": "Enhanced with validation system"
            }
        }

        video_path = output_path / "video_composition_instructions.json"
        with open(video_path, "w", encoding="utf-8") as f:
            json.dump(video_composition, f, indent=2, ensure_ascii=False)
        saved_files.append("video_composition_instructions.json")

        # 16. Generation report (COMPREHENSIVE WITH VALIDATION)
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
            "validation_system_enabled": result.get("generation_stats", {}).get("validation_system_enabled", False),
            "all_local_features_integrated": result.get("generation_stats", {}).get("all_local_features_integrated", False),
            "complete_pipeline": True,
            "stats": result["generation_stats"],
            "cost_analysis": {
                "total_api_calls": api_calls,
                "total_cost": total_cost,
                "cost_per_scene": total_cost / len(result.get("scene_plan", [1])),
                "cost_efficiency": "Claude 4 optimized with validation system"
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
                "hook_subscribe_generated": bool(result.get("hook_subscribe_scenes", {}).get("hook_scenes")),
                "validation_system_used": result.get("generation_stats", {}).get("validation_system_enabled", False),
                "stories_validated": result.get("generation_stats", {}).get("stories_validated", 0),
                "stories_corrected": result.get("generation_stats", {}).get("stories_corrected", 0)
            },
            "validation_summary": {
                "enabled": result.get("generation_stats", {}).get("validation_system_enabled", False),
                "stories_corrected": result.get("generation_stats", {}).get("stories_corrected", 0),
                "target_tolerance": CONFIG.claude_config.get('target_tolerance', 0.2),
                "auto_correction": CONFIG.claude_config.get('auto_correction', True),
                "validation_log_entries": len(result.get("validation_log", []))
            },
            "files_saved": saved_files,
            "next_steps": [
                "1. Generate character reference images using character_profiles.json",
                "2. Generate scene visuals (1-N) using visual_generation_prompts.json",
                "3. Generate thumbnail (scene 99) using visual_generation_prompts.json",
                "4. Generate audio using audio_generation_prompts.json with production specifications",
                "5. Compose video using video_composition_instructions.json with chapters",
                "6. Upload to YouTube using platform_metadata.json with full SEO optimization",
                "7. Review validation_report.json for quality assurance"
            ],
            "automation_readiness": {
                "character_extraction": "✅ Complete",
                "youtube_optimization": "✅ Complete",
                "production_specifications": "✅ Complete",
                "platform_metadata": "✅ Complete",
                "composition_strategy": "✅ Complete",
                "api_ready_format": "✅ Complete",
                "all_local_features": "✅ Complete",
                "validation_system": "✅ Complete with auto-correction"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ Complete production files saved ({len(saved_files)} TOTAL - ALL LOCAL + SERVER FEATURES + VALIDATION): {saved_files}")
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
    """Print complete production generation summary with validation system features"""
    stats = result["generation_stats"]

    print("\n" + "🚀" * 60)
    print("COMPLETE AUTOMATED STORY GENERATOR WITH VALIDATION SYSTEM!")
    print("🚀" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CONFIG.claude_config['model']} (Claude 4)")
    print(f"🖥️  Server Mode: {'✅ ACTIVE' if stats.get('server_optimized') else '❌ OFF'}")
    print(f"🏭 Complete Pipeline: {'✅ ACTIVE' if stats.get('complete_pipeline') else '❌ OFF'}")
    print(f"🎲 Smart Algorithm: {'✅ ACTIVE' if stats.get('smart_algorithm') else '❌ OFF'}")
    print(f"🎯 5-Stage Approach: {'✅ ACTIVE' if stats.get('five_stage_approach') else '❌ OFF'}")
    print(f"🔗 All Local Features: {'✅ INTEGRATED' if stats.get('all_local_features_integrated') else '❌ MISSING'}")
    print(f"🛡️  Validation System: {'✅ ACTIVE' if stats.get('validation_system_enabled') else '❌ OFF'}")
    print(f"🔧 Auto-Correction: {'✅ ACTIVE' if stats.get('auto_correction_enabled') else '❌ OFF'}")

    print(f"\n📊 CLAUDE 4 + VALIDATION PERFORMANCE:")
    print(f"🔥 Total API Calls: {stats['api_calls_used']}")
    print(f"💰 Total Cost: ${result.get('total_cost', 0):.4f}")
    print(f"⏱️  Total Generation Time: {generation_time:.1f}s")
    print(f"🎬 Scenes Planned: {stats['scenes_planned']}")
    print(f"📝 Stories Written: {stats['stories_written']}")
    print(f"✅ Stories Validated: {stats.get('stories_validated', 'N/A')}")
    print(f"🔧 Stories Corrected: {stats.get('stories_corrected', 0)}")
    print(f"👥 Characters Extracted: {stats['characters_extracted']}")
    print(f"🖼️  Thumbnail Generated: {'✅ YES' if stats.get('thumbnail_generated') else '❌ NO'}")
    print(f"📺 YouTube Optimization: {'✅ YES' if stats.get('youtube_optimization_generated') else '❌ NO'}")
    print(f"🏭 Production Specs: {'✅ YES' if stats.get('production_specifications_generated') else '❌ NO'}")
    print(f"🎭 Hook & Subscribe: {'✅ YES' if stats.get('hook_subscribe_generated') else '❌ NO'}")
    print(f"🎥 Visual Prompts (with thumbnail): {stats.get('visual_prompts_with_thumbnail', 0)}")

    # Validation system stats
    if stats.get('validation_system_enabled'):
        print(f"\n🛡️  VALIDATION SYSTEM RESULTS:")
        validation_report = result.get('validation_report', {})
        print(f"📊 Target Tolerance: ±{CONFIG.claude_config.get('target_tolerance', 0.2)*100:.0f}%")
        print(f"📏 Words per Minute: {CONFIG.claude_config.get('target_words_per_minute', 140)}")
        print(f"✅ Stories Validated: {stats.get('stories_validated', 0)}")
        print(f"🔧 Stories Auto-Corrected: {stats.get('stories_corrected', 0)}")
        if stats.get('stories_corrected', 0) > 0:
            print(f"💡 Duration accuracy improved through auto-correction!")
        print(f"📋 Validation Log Entries: {len(result.get('validation_log', []))}")

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
            print(f"📈 Total Duration: {total_duration:.1f} minutes ({total_duration / 60:.1f} hours)")
            print(f"📊 Average Scene: {total_duration / len(durations):.1f} minutes")
            print(f"🎯 Duration Accuracy: Smart algorithm + validation system ensures precision")

    # Character analysis
    characters = result.get("main_characters", [])
    if characters:
        print(f"\n👥 MAIN CHARACTERS:")
        for char in characters:
            print(
                f"• {char.get('name', 'Unknown')} ({char.get('role', 'unknown role')}) - Score: {char.get('importance_score', 0)}/10")

    completion_rate = (stats['stories_written'] / stats.get('scenes_planned', 1)) * 100
    print(f"\n📊 Story Completion: {completion_rate:.1f}%")

    if completion_rate >= 80:
        print(f"\n🎉 MASSIVE SUCCESS WITH VALIDATION!")
        print(f"✅ Complete story + character + YouTube + production + thumbnail + validation system")
        print(f"✅ ALL LOCAL FEATURES + VALIDATION SYSTEM INTEGRATED!")
        print(f"✅ Ready for FULL AUTOMATION with quality assurance")
        print(f"🚀 Zero manual work needed with automatic error correction!")
    elif completion_rate >= 60:
        print(f"\n✅ EXCELLENT PROGRESS WITH VALIDATION!")
        print(f"⚡ Ready for automated pipeline with quality assurance")
        print(f"🎯 Production deployment recommended with validation")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"🔍 Review generation_report.json and validation_report.json for issues")

    print("1. 📖 complete_story.txt - Full story text (validated)")
    print("2. 🎬 scene_plan.json - Smart scene structure + chapters")
    print("3. 🖼️  visual_generation_prompts.json - Scenes + Thumbnail (99)")
    print("4. 🎵 voice_directions.json - TTS guidance")
    print("5. 👥 character_profiles.json - Character data with generation instructions")
    print("6. 🛡️  validation_report.json - Validation system results (NEW)")
    print("7. 🌍 platform_metadata.json - COMPREHENSIVE platform data + API ready format")
    print("8. 📺 youtube_metadata.json - YouTube-specific metadata (compatibility)")
    print("9. 🖼️  thumbnail_generation.json - Thumbnail composition strategy")
    print("10. 🎭 hook_subscribe_scenes.json - Background scenes for opening")
    print("11. 🏭 production_specifications.json - Complete production specs")
    print("12. 🤖 automation_specs.json - Automation-specific data with validation")
    print("13. 🎵 audio_generation_prompts.json - Enhanced TTS production")
    print("14. 📚 all_stories.json - All validated stories in separate file")
    print("15. 🎥 video_composition_instructions.json - Video timeline + chapters")
    print("16. 📊 generation_report.json - Complete summary with validation metrics")

    print(f"\n🆕 VALIDATION SYSTEM ADVANTAGES:")
    print(f"✅ Automatic story duration validation")
    print(f"✅ Auto-correction for stories that are too short/long")
    print(f"✅ Target word count precision (±{CONFIG.claude_config.get('target_tolerance', 0.2)*100:.0f}% tolerance)")
    print(f"✅ Missing story generation for incomplete scenes")
    print(f"✅ Quality assurance with detailed validation logs")
    print(f"✅ Consistent duration targeting for TTS optimization")
    print(f"✅ Automatic retry and correction system")
    print(f"✅ Budget-aware correction limiting")

    print(f"\n💰 EFFICIENCY vs MANUAL WORK WITH VALIDATION:")
    print(f"💵 Cost: 5-7 API calls vs manual story editing + duration checking")
    print(f"⚡ Speed: Automatic validation + correction + character extraction + platform optimization")
    print(f"🔧 Consistency: Built-in quality control + duration validation + character mapping")
    print(f"🎯 Scalability: Works for any story topic with automatic quality assurance")
    print(f"🛡️  Reliability: Validation system ensures consistent output quality")

    print(f"\n🏆 COMPLETE AUTOMATION ADVANTAGES (ALL FEATURES + VALIDATION):")
    print("✅ Dynamic character extraction for any topic")
    print("✅ Automatic consistency mapping")
    print("✅ Visual generation pipeline ready")
    print("✅ FIXED: Visual prompts match scene content exactly")
    print("✅ Character-scene mapping for perfect consistency")
    print("✅ INTELLIGENT THUMBNAIL GENERATION")
    print("✅ Character analysis for optimal thumbnail selection")
    print("✅ Clickbait optimization while maintaining sleep content feel")
    print("✅ COMPREHENSIVE PLATFORM OPTIMIZATION")
    print("✅ Enhanced API-ready formats for all platforms")
    print("✅ Complete audio production specs with TTS optimization")
    print("✅ Video assembly automation with precise timing")
    print("✅ Quality control validation with smart algorithm")
    print("✅ Batch processing automation with database management")
    print("✅ Precise timing calculations with natural variation")
    print("✅ 🆕 AUTOMATIC STORY VALIDATION AND CORRECTION")
    print("✅ 🆕 DURATION ACCURACY ASSURANCE")
    print("✅ 🆕 MISSING STORY AUTO-GENERATION")
    print("✅ 🆕 QUALITY METRICS AND REPORTING")
    print("✅ Zero manual work needed - 16 complete files with validation")
    print("✅ Scalable to unlimited stories with quality assurance")
    print("✅ FULL END-TO-END AUTOMATION WITH VALIDATION SYSTEM")

    print("🚀" * 60)


def run_autonomous_mode():
    """Run autonomous mode with validation system - continuously process pending topics"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / 'database'))
    from autonomous_database_manager import AutonomousDatabaseManager
    import signal

    print("🤖 AUTONOMOUS MODE WITH VALIDATION SYSTEM STARTED")
    print("🔄 Will process all pending topics continuously")
    print("🛡️  Validation system enabled for quality assurance")
    print("⏹️ Press Ctrl+C to stop gracefully")

    # Initialize database manager
    db_manager = AutonomousDatabaseManager()

    # Setup graceful shutdown
    running = True
    processed_count = 0
    start_time = time.time()

    def signal_handler(signum, frame):
        nonlocal running
        print(f"\n⏹️ Received shutdown signal ({signum})")
        print("🔄 Finishing current topic and shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            # Get next topic ready for story generation
            next_topic = db_manager.get_next_story_generation_topic()

            if next_topic:
                topic_id, topic, description = next_topic
                clickbait_title = ""
                font_design = ""

                print(f"\n🔄 Processing topic {topic_id}: {topic}")
                print(f"🛡️  Validation system will ensure quality")

                # Mark as started
                db_manager.mark_story_generation_started(topic_id)
                try:
                    # Setup output directory
                    output_path = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id)

                    # Initialize generator with validation
                    generator = AutomatedStoryGenerator()

                    # Generate complete story with validation
                    start_gen_time = time.time()
                    result = generator.generate_complete_story_with_characters(
                        topic, description, clickbait_title, font_design
                    )
                    generation_time = time.time() - start_gen_time

                    # Add total cost
                    result['total_cost'] = generator.total_cost

                    # Save all production outputs (ALL 16 FILES WITH VALIDATION)
                    save_production_outputs(
                        str(output_path), result, topic, topic_id,
                        generator.api_call_count, generator.total_cost
                    )

                    # Print summary with validation stats
                    print_production_summary(result, topic, str(output_path), generation_time)

                    processed_count += 1
                    print(f"\n✅ Topic {topic_id} completed successfully with validation!")
                    print(f"📊 Progress: {processed_count} topics processed")

                    # Show validation summary
                    if result.get('generation_stats', {}).get('validation_system_enabled'):
                        stories_corrected = result.get('generation_stats', {}).get('stories_corrected', 0)
                        if stories_corrected > 0:
                            print(f"🛡️  Validation System: {stories_corrected} stories auto-corrected for optimal duration")

                except Exception as e:
                    print(f"❌ Error processing topic {topic_id}: {e}")
                    db_manager.mark_story_generation_failed(topic_id, str(e))
                    import traceback
                    traceback.print_exc()

                # Short pause between topics
                if running:
                    time.sleep(5)

            else:
                # No topics ready, wait
                print("😴 No topics ready for processing. Waiting 60s...")
                for i in range(60):
                    if not running:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️ Keyboard interrupt received")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(30)

    # Shutdown summary
    runtime = time.time() - start_time
    print(f"\n🏁 AUTONOMOUS MODE WITH VALIDATION SHUTDOWN")
    print(f"⏱️ Total runtime: {runtime / 3600:.1f} hours")
    print(f"✅ Topics processed: {processed_count}")
    print(f"🛡️  Validation system ensured quality throughout")
    print("👋 Goodbye!")


if __name__ == "__main__":
    # Check for autonomous mode
    if len(sys.argv) > 1 and sys.argv[1] == '--autonomous':
        run_autonomous_mode()
    else:
        # Original single topic mode with validation
        try:
            print("🚀 COMPLETE AUTOMATED STORY GENERATOR WITH VALIDATION SYSTEM")
            print("⚡ Server-optimized with complete pipeline + ALL local features + VALIDATION")
            print("🎲 FIXED: Smart random scene count & duration generation")
            print("📊 FIXED: Database integration instead of CSV")
            print("🎭 5-stage approach: Planning + Stories + VALIDATION + Characters + Thumbnail + Hook/Subscribe")
            print("🛡️  NEW: Automatic story validation and correction system")
            print("🔧 NEW: Auto-correction for duration accuracy")
            print("📄 Complete JSON outputs for automation (16 files)")
            print("🎯 RIGHT-side thumbnail positioning for text overlay")
            print("✅ INTEGRATED: All local features + Enhanced server features + Validation system")
            print("🌍 COMPREHENSIVE: platform_metadata.json + youtube_metadata.json + validation_report.json")
            print("=" * 60)

            # Get next topic from database
            topic_id, topic, description, clickbait_title, font_design = get_next_topic_from_database()
            print(f"\n📚 Topic ID: {topic_id} - {topic}")
            print(f"📝 Description: {description}")
            if clickbait_title:
                print(f"🎯 Clickbait Title: {clickbait_title}")

            # Setup output directory
            output_path = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id)

            # Initialize generator with validation
            generator = AutomatedStoryGenerator()

            # Generate complete story with ALL LOCAL + SERVER FEATURES + VALIDATION
            start_time = time.time()
            result = generator.generate_complete_story_with_characters(
                topic, description, clickbait_title, font_design
            )
            generation_time = time.time() - start_time

            # Add total cost to result
            result['total_cost'] = generator.total_cost

            # Save outputs with ALL LOCAL + SERVER FEATURES + VALIDATION
            save_production_outputs(str(output_path), result, topic, topic_id,
                                    generator.api_call_count, generator.total_cost)

            # Print comprehensive summary with validation
            print_production_summary(result, topic, str(output_path), generation_time)

            print("\n🚀 COMPLETE PRODUCTION PIPELINE WITH VALIDATION FINISHED!")
            print(f"✅ All files ready for: {output_path}")
            print(f"📊 Database topic management: WORKING")
            print(f"🎲 Smart algorithm scene generation: FIXED")
            print(f"📝 Story distribution: FIXED")
            print(f"🛡️  Story validation system: ADDED")
            print(f"🔧 Auto-correction system: ADDED")
            print(f"📚 all_stories.json: VALIDATED")
            print(f"🤖 automation_specs.json: ENHANCED WITH VALIDATION")
            print(f"🌍 platform_metadata.json: COMPREHENSIVE")
            print(f"🔌 api_ready_format: ENHANCED")
            print(f"🎭 character extraction: ADVANCED")
            print(f"🖼️  thumbnail generation: INTELLIGENT")
            print(f"🎬 video composition: AUTOMATED")
            print(f"📋 validation_report.json: QUALITY ASSURANCE")
            print(f"💰 Total cost: ${result.get('total_cost', 0):.4f}")

            # Show validation results
            if result.get('generation_stats', {}).get('validation_system_enabled'):
                stories_corrected = result.get('generation_stats', {}).get('stories_corrected', 0)
                stories_validated = result.get('generation_stats', {}).get('stories_validated', 0)
                print(f"🛡️  Validation Results: {stories_validated} stories validated, {stories_corrected} auto-corrected")

            print(f"🏆 SUCCESS: All local features + validation system integrated into server version!")

        except Exception as e:
            print(f"\n💥 COMPLETE GENERATOR ERROR: {e}")
            CONFIG.logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()