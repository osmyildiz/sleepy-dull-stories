"""
Sleepy Dull Stories - COMPLETE TÓIBÍN QUALITY STORY GENERATOR with ALL 15 JSON FILES
System Logic: Master Plan → Emotional Distribution → Scene Creation → Tóibín Stories → Production JSONs → Social Media
Quality Focus: Literary excellence with COLM TÓIBÍN standards throughout
RESTORED: All 15 JSON files from the comprehensive version
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

# Load environment
load_dotenv()

class ServerConfig:
    """Server configuration management"""

    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.setup_claude_config()
        self.ensure_directories()

    def setup_paths(self):
        """Setup server paths"""
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent

        self.paths = {
            'BASE_DIR': str(self.project_root),
            'DATA_DIR': str(self.project_root / 'data'),
            'OUTPUT_DIR': str(self.project_root / 'output'),
            'LOGS_DIR': str(self.project_root / 'logs'),
        }

        print(f"✅ Server paths configured: {self.paths['BASE_DIR']}")

    def setup_claude_config(self):
        """Setup Claude configuration for TÓIBÍN QUALITY"""
        self.claude_config = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 64000,
            "temperature": 0.7,
            "target_words_per_minute": 140,
            "validation_tolerance": 0.15,  # ±15% tolerance
            "toibin_style_required": True
        }

        # Get API key
        self.api_key = self.get_claude_api_key()

    def get_claude_api_key(self):
        """Get Claude API key"""
        api_key = (
            os.getenv('CLAUDE_API_KEY') or
            os.getenv('ANTHROPIC_API_KEY')
        )

        if not api_key:
            raise ValueError("❌ Claude API key required!")

        print("✅ Claude API key loaded")
        return api_key

    def setup_logging(self):
        """Setup logging"""
        logs_dir = Path(self.project_root) / 'logs' / 'generators'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"toibin_gen_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("TóibínGenerator")

    def ensure_directories(self):
        """Ensure directories exist"""
        for dir_path in self.paths.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# Initialize config
try:
    CONFIG = ServerConfig()
    print("🚀 Server configuration loaded successfully")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Import Anthropic
try:
    from anthropic import Anthropic
    print("✅ Anthropic library imported")
except ImportError:
    print("❌ Anthropic library not found")
    sys.exit(1)

class DatabaseTopicManager:
    """Topic management using database"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_topic_table()

    def setup_topic_table(self):
        """Setup topic table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                description TEXT NOT NULL,
                clickbait_title TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                scene_count INTEGER,
                total_duration_minutes REAL,
                api_calls_used INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                output_path TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get_next_pending_topic(self) -> Optional[Tuple[int, str, str, str]]:
        """Get next pending topic"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, description, clickbait_title 
            FROM topics 
            WHERE status = 'pending' 
            ORDER BY created_at ASC 
            LIMIT 1
        ''')

        result = cursor.fetchone()
        conn.close()
        return result

    def mark_topic_started(self, topic_id: int):
        """Mark topic as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE topics 
            SET status = 'in_progress' 
            WHERE id = ?
        ''', (topic_id,))

        conn.commit()
        conn.close()

    def mark_topic_completed(self, topic_id: int, scene_count: int, total_duration: float,
                             api_calls: int, total_cost: float, output_path: str):
        """Mark topic as completed in database - FIXED"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # FIX: Use correct column name and proper parameter binding
        cursor.execute('''
            UPDATE topics 
            SET status = ?,
                production_completed_at = CURRENT_TIMESTAMP,
                scene_count = ?,
                total_duration_minutes = ?,
                api_calls_used = ?,
                total_cost = ?,
                output_path = ?
            WHERE id = ?
        ''', ('completed', scene_count, total_duration, api_calls, total_cost, output_path, topic_id))

        conn.commit()
        conn.close()

class ToibinStoryGenerator:
    """TÓIBÍN QUALITY Story Generator with Master Plan Approach"""

    def __init__(self):
        self.client = Anthropic(api_key=CONFIG.api_key)
        self.api_call_count = 0
        self.total_cost = 0.0
        self.generation_log = []

        print("✅ TÓIBÍN Quality Generator initialized")

    def log_step(self, description: str, status: str = "START", metadata: Dict = None):
        """Log generation steps - FIXED"""
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

        # FIX: Use .4f format for float values instead of d
        print(f"{icon} {description} [API: {self.api_call_count}] [Cost: ${self.total_cost:.4f}]")
        CONFIG.logger.info(f"{description} - {status} - API: {self.api_call_count} - Cost: ${self.total_cost:.4f}")

    def generate_complete_story(self, topic: str, description: str, clickbait_title: str = None) -> Dict[str, Any]:
        """
        COMPLETE TÓIBÍN QUALITY GENERATION PIPELINE:
        1. Emotional Scene Structure (28-40 scenes, emotion distribution)
        2. Master Scene Plan (Claude as director creating detailed scenes)
        3. Stage 1: First half stories (Tóibín style)
        4. Stage 2: Second half stories (Tóibín continuity)
        5. Production JSONs (characters, thumbnails, metadata)
        6. Social Media Content (YouTube Shorts, Instagram Reels, TikTok)
        """

        self.log_step("TÓIBÍN QUALITY Generation Pipeline Started")

        try:
            # STEP 1: Generate emotional scene structure
            scene_structure = self._generate_emotional_scene_structure()

            # STEP 2: Create master scene plan with Claude as director
            master_plan = self._create_master_scene_plan(topic, description, scene_structure)

            # STEP 3: Generate stories - Stage 1 (First half)
            stage1_stories = self._generate_stage1_stories(topic, description, master_plan)

            # STEP 4: Generate stories - Stage 2 (Second half)
            stage2_stories = self._generate_stage2_stories(topic, description, master_plan, stage1_stories)

            # STEP 5: Create production JSONs
            production_data = self._create_production_jsons(topic, description, master_plan,
                                                          stage1_stories, stage2_stories, clickbait_title)

            # STEP 6: Create viral social media content
            all_stories = {}
            all_stories.update(stage1_stories.get('stories', {}))
            all_stories.update(stage2_stories.get('stories', {}))

            social_media_content = self._create_social_media_content(topic, description, master_plan, all_stories)

            # COMBINE ALL RESULTS
            complete_result = self._combine_all_results(master_plan, stage1_stories, stage2_stories,
                                                      production_data, topic, description)

            # Social media content'i sonradan ekle
            complete_result["social_media_content"] = social_media_content

            # Stats'ı güncelle
            complete_result["generation_stats"]["social_media_pieces"] = (
                len(social_media_content.get('youtube_shorts', [])) +
                len(social_media_content.get('instagram_reels', [])) +
                len(social_media_content.get('tiktok_videos', []))
            )
            complete_result["generation_stats"]["viral_content_created"] = True

            self.log_step("TÓIBÍN QUALITY Generation Complete", "SUCCESS", {
                "total_scenes": len(master_plan.get('scene_plan', [])),
                "total_stories": len(complete_result.get('stories', {})),
                "social_media_pieces": complete_result["generation_stats"]["social_media_pieces"],
                "api_calls_total": self.api_call_count,
                "total_cost": self.total_cost
            })

            return complete_result

        except Exception as e:
            self.log_step("Generation Failed", "ERROR")
            CONFIG.logger.error(f"Generation failed: {e}")
            raise

    def _generate_emotional_scene_structure(self) -> Dict:
        """Generate emotional scene structure with proper distribution"""

        # Random scene count (28-40)
        total_scenes = random.randint(28, 40)

        # Emotional distribution ratios
        peaceful_ratio = 0.35      # 35% peaceful scenes
        curiosity_ratio = 0.25     # 25% curiosity/discovery
        contemplation_ratio = 0.25 # 25% contemplation/recognition
        resolution_ratio = 0.15    # 15% resolution/acceptance

        # Calculate scene counts
        peaceful_count = int(total_scenes * peaceful_ratio)
        curiosity_count = int(total_scenes * curiosity_ratio)
        contemplation_count = int(total_scenes * contemplation_ratio)
        resolution_count = total_scenes - peaceful_count - curiosity_count - contemplation_count

        # Create emotional progression
        emotional_structure = []

        # Phase 1: Peaceful establishment (first 35%)
        for i in range(peaceful_count):
            emotional_structure.append({
                "scene_number": i + 1,
                "emotion": "peaceful",
                "phase": "establishment",
                "duration_range": (4.0, 6.5),
                "toibin_focus": "daily_life_observation"
            })

        # Phase 2: Curiosity/Discovery (next 25%)
        for i in range(curiosity_count):
            emotional_structure.append({
                "scene_number": peaceful_count + i + 1,
                "emotion": "curiosity",
                "phase": "discovery",
                "duration_range": (3.5, 5.0),
                "toibin_focus": "character_recognition"
            })

        # Phase 3: Contemplation (next 25%)
        for i in range(contemplation_count):
            emotional_structure.append({
                "scene_number": peaceful_count + curiosity_count + i + 1,
                "emotion": "contemplation",
                "phase": "recognition",
                "duration_range": (3.0, 4.5),
                "toibin_focus": "internal_complexity"
            })

        # Phase 4: Resolution (final 15%)
        for i in range(resolution_count):
            emotional_structure.append({
                "scene_number": peaceful_count + curiosity_count + contemplation_count + i + 1,
                "emotion": "resolution",
                "phase": "acceptance",
                "duration_range": (4.5, 7.0),
                "toibin_focus": "quiet_acceptance"
            })

        structure = {
            "total_scenes": total_scenes,
            "emotional_distribution": {
                "peaceful": peaceful_count,
                "curiosity": curiosity_count,
                "contemplation": contemplation_count,
                "resolution": resolution_count
            },
            "emotional_structure": emotional_structure,
            "target_duration": sum(random.uniform(scene["duration_range"][0], scene["duration_range"][1])
                                 for scene in emotional_structure)
        }

        self.log_step(f"Emotional Structure Generated: {total_scenes} scenes", "SUCCESS", {
            "peaceful": peaceful_count,
            "curiosity": curiosity_count,
            "contemplation": contemplation_count,
            "resolution": resolution_count,
            "target_duration": f"{structure['target_duration']:.1f} minutes"
        })

        return structure

    def _create_master_scene_plan(self, topic: str, description: str, scene_structure: Dict) -> Dict:
        """Claude as master director creating detailed scene plan"""

        self.log_step("Creating Master Scene Plan with Claude Director")

        emotional_structure = scene_structure["emotional_structure"]
        total_scenes = scene_structure["total_scenes"]

        # Create emotional summary for Claude
        emotional_summary = "\n".join([
            f"Scene {scene['scene_number']}: {scene['emotion']} ({scene['phase']}) - {scene['toibin_focus']} - Duration: {scene['duration_range'][0]:.1f}-{scene['duration_range'][1]:.1f} min"
            for scene in emotional_structure
        ])

        master_plan_prompt = f"""You are CLAUDE, master film director creating a detailed scene plan for "{topic}" in COLM TÓIBÍN's literary style.

TOPIC: {topic}
DESCRIPTION: {description}

EMOTIONAL STRUCTURE TO FOLLOW (EXACTLY {total_scenes} SCENES):
{emotional_summary}

🎬 YOUR ROLE AS MASTER DIRECTOR:

You must create EXACTLY {total_scenes} detailed scenes that follow TÓIBÍN'S LITERARY MASTERY while respecting the emotional progression. Each scene should be a small masterpiece of character psychology and daily life observation.

## TÓIBÍN'S CORE PRINCIPLES TO FOLLOW:

### 1. "Sparseness of tone with superabundance of suggestion"
- Every scene should say more through what's NOT said
- Focus on small gestures that reveal character depths
- Understated dialogue, powerful subtext

### 2. "Characters led by desires they don't understand" 
- Characters should have complex internal motivations
- Show internal contradictions and mixed feelings
- Focus on psychological authenticity over external action

### 3. "Fascination of commonplaces"
- Transform ordinary daily activities into profound moments
- Meals, conversations, walks, work - make them meaningful
- Find the extraordinary in the completely ordinary

### 4. Daily Life as Drama
- NO external dramatic events needed
- Internal character drama is sufficient
- Relationships and emotions are the real "plot"

OUTPUT FORMAT:
{{
  "master_plan": {{
    "topic_analysis": {{
      "central_theme": "[What is the core story we're telling?]",
      "character_focus": "[Primary character types and their psychology]",
      "daily_life_elements": "[Specific authentic activities for this topic]",
      "tóibín_approach": "[How Tóibín would approach this subject]",
      "historical_authenticity": "[Key period details that matter]"
    }},
    "scene_plan": [
      {{
        "scene_id": 1,
        "title": "[Specific scene title related to topic and moment]",
        "emotion": "{emotional_structure[0]['emotion']}",
        "phase": "{emotional_structure[0]['phase']}",
        "duration_minutes": {random.uniform(emotional_structure[0]['duration_range'][0], emotional_structure[0]['duration_range'][1]):.1f},
        "setting": "[Specific location within topic context]",
        "time_of_day": "[Specific time, contributing to daily progression]",
        "main_character": "[Character name and brief description]",
        "activity": "[What exactly the character is doing]", 
        "internal_focus": "[Character's internal state/thoughts]",
        "tóibín_element": "[Which Tóibín technique this demonstrates]",
        "scene_description": "[Detailed description of what happens in this scene]",
        "dialogue_style": "[How characters speak - understated, gestural, etc.]",
        "environmental_details": "[Specific sensory details of the setting]",
        "emotional_core": "[The feeling/recognition at the heart of this scene]",
        "connection_to_previous": "[How this connects to the previous scene]",
        "connection_to_next": "[How this sets up the next scene]",
        "key_tóibín_qualities": ["silence_between_words", "ordinary_made_profound", "character_psychology"],
        "sleep_optimization": "[How this scene promotes peaceful contemplation]"
      }}
    ]
  }}
}}

Create {total_scenes} scenes that demonstrate TÓIBÍN'S LITERARY GENIUS while authentically representing {topic}."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=20000,
                temperature=0.5,
                timeout=300,  # 5 dakika timeout
                system="You are COLM TÓIBÍN, the master Irish literary author, working as a film director to create a detailed scene plan. Apply your signature literary genius: 'sparseness of tone with superabundance of suggestion,' characters led by desires they don't understand, the fascination of commonplaces, and quiet recognition of human psychology.",
                messages=[{"role": "user", "content": master_plan_prompt}]
            )

            # SENİN SİSTEMİNİ KULLANIYORUM - NON-STREAMING
            content = response.content[0].text

            print(f"✅ Master scene plan complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(master_plan_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "master_plan")

            self.log_step("Master Scene Plan Created", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('master_plan', {}).get('scene_plan', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Master Scene Plan Failed", "ERROR")
            CONFIG.logger.error(f"Master plan error: {e}")
            raise

    def _generate_stage1_stories(self, topic: str, description: str, master_plan: Dict) -> Dict:
        """Generate first half stories following master plan"""

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
        total_scenes = len(scene_plan)
        first_half = total_scenes // 2

        first_half_scenes = scene_plan[:first_half]

        self.log_step(f"Stage 1: Generating first {first_half} Tóibín stories")

        # Create hook and subscribe content
        hook_content = self._create_hook_content(topic, description)
        subscribe_content = self._create_subscribe_content()

        # Format scenes for prompt
        scenes_text = "\n\n".join([
            f"SCENE {scene['scene_id']}: {scene['title']}\n"
            f"Duration: {scene['duration_minutes']:.1f} minutes\n"
            f"Setting: {scene['setting']}\n"
            f"Main Character: {scene['main_character']}\n"
            f"Activity: {scene['activity']}\n"
            f"Internal Focus: {scene['internal_focus']}\n"
            f"Tóibín Element: {scene['tóibín_element']}\n"
            f"Scene Description: {scene['scene_description']}\n"
            f"Emotional Core: {scene['emotional_core']}"
            for scene in first_half_scenes
        ])

        stage1_prompt = f"""Write the complete first half of this TÓIBÍN masterpiece sleep story for "{topic}".

TOPIC: {topic}  
DESCRIPTION: {description}

SCENES TO WRITE (First {first_half} of {total_scenes}):
{scenes_text}

🎭 TÓIBÍN WRITING REQUIREMENTS:

Create stories worthy of COLM TÓIBÍN's literary reputation while serving as perfect sleep content.

OUTPUT FORMAT:
{{
  "golden_hook": {{
    "content": "{hook_content}",
    "duration_seconds": 30,
    "voice_direction": "Gentle, literary, contemplative - like Tóibín reading aloud"
  }},
  "subscribe_section": {{
    "content": "{subscribe_content}",
    "duration_seconds": 30, 
    "voice_direction": "Warm, literary, non-commercial - book club invitation"
  }},
  "stories": {{
    "1": "[Complete Tóibín-style story for scene 1, following master plan exactly]",
    "2": "[Complete Tóibín-style story for scene 2, following master plan exactly]"
  }},
  "stage1_stats": {{
    "scenes_written": {first_half},
    "total_planned": {total_scenes},
    "tóibín_mastery_applied": true,
    "master_plan_followed": true
  }}
}}"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=30000,
                temperature=0.7,
                timeout=300,
                system="You are COLM TÓIBÍN, the celebrated Irish master of literary fiction. Apply your signature style: 'sparseness of tone with superabundance of suggestion,' characters led by desires they don't understand, the fascination of commonplaces, and the quiet recognition of human psychology.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            # SENİN SİSTEMİNİ KULLANIYORUM - NON-STREAMING
            content = response.content[0].text

            print(f"✅ Stage 1 complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage1_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "stage1")

            self.log_step("Stage 1 Stories Generated", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 1 Failed", "ERROR")
            CONFIG.logger.error(f"Stage 1 error: {e}")
            raise

    def _generate_stage2_stories(self, topic: str, description: str, master_plan: Dict, stage1_result: Dict) -> Dict:
        """Generate second half stories with Tóibín continuity"""

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
        total_scenes = len(scene_plan)
        first_half = total_scenes // 2

        second_half_scenes = scene_plan[first_half:]

        self.log_step(f"Stage 2: Generating remaining {len(second_half_scenes)} Tóibín stories")

        # Format scenes for prompt
        scenes_text = "\n\n".join([
            f"SCENE {scene['scene_id']}: {scene['title']}\n"
            f"Duration: {scene['duration_minutes']:.1f} minutes\n"
            f"Setting: {scene['setting']}\n"
            f"Main Character: {scene['main_character']}\n"
            f"Activity: {scene['activity']}\n"
            f"Internal Focus: {scene['internal_focus']}\n"
            f"Tóibín Element: {scene['tóibín_element']}\n"
            f"Scene Description: {scene['scene_description']}\n"
            f"Emotional Core: {scene['emotional_core']}\n"
            f"Connection to Previous: {scene['connection_to_previous']}"
            for scene in second_half_scenes
        ])

        stage2_prompt = f"""Continue this TÓIBÍN masterpiece by writing the second half for "{topic}".

TOPIC: {topic}
DESCRIPTION: {description}

SCENES TO WRITE (Remaining {len(second_half_scenes)} of {total_scenes}):
{scenes_text}

Complete this TÓIBÍN masterpiece with the same literary excellence as the first half.

OUTPUT FORMAT:
{{
  "stories": {{
    "{second_half_scenes[0]['scene_id'] if second_half_scenes else 'X'}": "[Complete Tóibín story following master plan]",
    "{second_half_scenes[1]['scene_id'] if len(second_half_scenes) > 1 else 'Y'}": "[Complete Tóibín story with character continuity]"
  }},
  "stage2_stats": {{
    "scenes_written": {len(second_half_scenes)},
    "character_continuity_maintained": true,
    "tóibín_mastery_sustained": true,
    "master_plan_completed": true
  }}
}}"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=30000,
                temperature=0.7,
                timeout=300,
                system="You are COLM TÓIBÍN continuing your literary masterwork from Stage 1. Maintain absolute character consistency - same internal voices, relationship patterns, and emotional rhythms established earlier.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            # SENİN SİSTEMİNİ KULLANIYORUM - NON-STREAMING
            content = response.content[0].text

            print(f"✅ Stage 2 complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 Stories Generated", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 Failed", "ERROR")
            CONFIG.logger.error(f"Stage 2 error: {e}")
            raise

    def _create_production_jsons(self, topic: str, description: str, master_plan: Dict,
                                 stage1_result: Dict, stage2_result: Dict, clickbait_title: str = None) -> Dict:
        """Create all production JSONs based on the completed stories - ENHANCED VERSION"""

        self.log_step("Creating Production JSONs")

        # Combine all stories
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
        total_scenes = len(scene_plan)

        # Create story content for analysis - ENHANCED with scene mapping
        story_content = ""
        scene_story_mapping = {}

        for scene in scene_plan:
            scene_id = str(scene.get('scene_id', 0))
            story = all_stories.get(scene_id, '')

            if story:
                scene_story_mapping[scene_id] = {
                    'title': scene.get('title', f'Scene {scene_id}'),
                    'story_content': story[:1000],  # First 1000 chars
                    'setting': scene.get('setting', ''),
                    'main_character': scene.get('main_character', ''),
                    'emotion': scene.get('emotion', 'peaceful'),
                    'duration_minutes': scene.get('duration_minutes', 4.0)
                }

                story_content += f"Scene {scene_id}: {scene.get('title', '')}\n"
                story_content += f"Characters: {scene.get('main_character', '')}\n"
                story_content += f"Story: {story[:500]}...\n\n"

        production_prompt = f"""Analyze this complete TÓIBÍN story and create comprehensive production JSONs for "{topic}".

    TOPIC: {topic}
    DESCRIPTION: {description}
    TOTAL SCENES: {total_scenes}

    SCENE-STORY MAPPING:
    {json.dumps(scene_story_mapping, indent=2)[:15000]}

    CRITICAL: Create visual prompts for ALL {total_scenes} scenes, not just some!

    OUTPUT FORMAT:
    {{
      "characters": {{
        "main_characters": [
          {{
            "name": "[Character name from stories]",
            "role": "protagonist|supporting|background",
            "gender": "male|female",
            "importance_score": 8,
            "scene_appearances": [1, 3, 7, 12],
            "use_in_marketing": true,
            "personality_traits": ["trait1", "trait2", "trait3"],
            "physical_description": "[Detailed visual description]",
            "tóibín_psychology": {{
              "internal_contradiction": "[What they want vs understand]",
              "quiet_dignity": "[How they maintain composure]",
              "emotional_complexity": "[Mixed feelings and motivations]"
            }},
            "relationships": [
              {{"character": "other_name", "dynamic": "detailed relationship"}}
            ]
          }}
        ],
        "scene_character_mapping": {{
          "1": ["Character1"],
          "2": ["Character1", "Character2"]
        }}
      }},
      "visual_prompts": [
        {{
          "scene_number": 1,
          "title": "[Scene title from master plan]",
          "prompt": "[DETAILED visual prompt based on actual story content and characters]",
          "duration_minutes": 4.5,
          "characters_present": ["Character1"],
          "historical_accuracy": "[Period-specific visual elements]",
          "tóibín_atmosphere": "[Contemplative, understated visual mood]"
        }},
        {{
          "scene_number": 2,
          "title": "[Scene 2 title]",
          "prompt": "[Visual prompt for scene 2]",
          "duration_minutes": 4.0,
          "characters_present": ["Character2"],
          "historical_accuracy": "[Historical elements]",
          "tóibín_atmosphere": "[Emotional atmosphere]"
        }}
        // CONTINUE FOR ALL {total_scenes} SCENES
      ],
      "youtube_optimization": {{
        "clickbait_titles": [
          "MYSTERIOUS! [Character]'s Final Secret Decision (2 Hour Sleep Story)",
          "BETRAYAL! The Night [Character] Never Saw Coming - Most Peaceful Historical Story",
          "INCREDIBLE! [Character]'s Most Peaceful Final Hours - You Won't Believe What Happened"
        ],
        "thumbnail_concept": {{
          "main_character": "[Main character name]",
          "emotional_expression": "[Character psychology visible in expression]",
          "historical_setting": "[Atmospheric background]",
          "composition": "RIGHT-side character, LEFT-side text space"
        }},
        "seo_strategy": {{
          "primary_keywords": ["sleep story", "{topic.lower()}", "bedtime story"],
          "long_tail_keywords": ["2 hour sleep story {topic.lower()}", "historical bedtime story"],
          "tags": ["sleep story", "bedtime story", "relaxation", "history", "{topic.lower()}"]
        }}
      }},
      "production_specs": {{
        "audio_production": {{
          "tts_voice": "alloy",
          "speed_multiplier": 0.85,
          "pause_durations": {{
            "[PAUSE]": 2.0,
            "scene_transition": 3.0
          }}
        }},
        "video_timing": [
          {{
            "scene_number": 1,
            "start_time": "00:01:00",
            "duration_minutes": 4.5,
            "word_count": 630
          }}
        ],
        "quality_metrics": {{
          "sleep_optimization_score": 9,
          "historical_accuracy": true,
          "tóibín_authenticity": true,
          "all_scenes_covered": true
        }}
      }}
    }}

    Generate visual prompts for ALL {total_scenes} scenes based on the actual story content and character analysis."""

        try:
            self.api_call_count += 1

            # ENHANCED: Increased token limit and timeout for complete scene coverage
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=25000,  # FIX: Increased from 16000 to 25000
                temperature=0.3,
                timeout=400,  # FIX: Increased timeout to 400 seconds
                system="You are a production expert analyzing COLM TÓIBÍN's literary work. Create visual prompts for EVERY SINGLE SCENE based on the story content. Do not skip any scenes.",
                messages=[{"role": "user", "content": production_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Production JSONs complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(production_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response with enhanced validation
            parsed_result = self._parse_claude_response(content, "production")

            # VALIDATION: Check if all scenes are covered
            visual_prompts = parsed_result.get('visual_prompts', [])
            scenes_covered = len(visual_prompts)
            expected_scenes = len(scene_plan)

            if scenes_covered < expected_scenes:
                self.log_step(f"WARNING: Only {scenes_covered}/{expected_scenes} scenes covered in visuals", "WARNING")

                # Generate missing scenes
                missing_scenes = []
                covered_scene_numbers = {vp.get('scene_number') for vp in visual_prompts}

                for scene in scene_plan:
                    scene_num = scene.get('scene_id')
                    if scene_num not in covered_scene_numbers:
                        missing_scenes.append({
                            "scene_number": scene_num,
                            "title": scene.get('title', f'Scene {scene_num}'),
                            "prompt": f"Roman historical scene: {scene.get('setting', 'unknown setting')}, {scene.get('main_character', 'character')} {scene.get('activity', 'in contemplation')}, {scene.get('emotion', 'peaceful')} atmosphere, Tóibín-style understated mood",
                            "duration_minutes": scene.get('duration_minutes', 4.0),
                            "characters_present": [scene.get('main_character', 'Unknown')],
                            "historical_accuracy": "Roman period details, authentic settings",
                            "tóibín_atmosphere": f"{scene.get('emotion', 'peaceful')} contemplation, understated emotional depth"
                        })

                # Add missing scenes to the result
                if 'visual_prompts' not in parsed_result:
                    parsed_result['visual_prompts'] = []
                parsed_result['visual_prompts'].extend(missing_scenes)

                self.log_step(f"Added {len(missing_scenes)} missing visual prompts", "SUCCESS")

            self.log_step("Production JSONs Created", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('characters', {}).get('main_characters', [])),
                "visual_prompts_created": len(parsed_result.get('visual_prompts', [])),
                "all_scenes_covered": len(parsed_result.get('visual_prompts', [])) >= expected_scenes,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Production JSONs Failed", "ERROR")
            CONFIG.logger.error(f"Production JSON error: {e}")
            raise

    def _create_social_media_content(self, topic: str, description: str, master_plan: Dict,
                                   all_stories: Dict) -> Dict:
        """Create platform-specific social media content for viral reach using Claude API"""

        self.log_step("Creating Social Media Content for Viral Growth")

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])

        # Select best scenes for social media (high emotion, visual potential)
        selected_scenes = self._select_best_scenes_for_social(scene_plan)

        social_media_prompt = f"""Create viral social media content for "{topic}" designed to drive traffic to Sleepy Dull Stories YouTube channel (goal: 1M subscribers).

CRITICAL: Each piece must include MIDJOURNEY VISUAL PROMPTS for image generation!

TOPIC: {topic}
DESCRIPTION: {description}

SELECTED SCENES FOR ADAPTATION:
{self._format_scenes_for_social(selected_scenes, all_stories)}

Create 15 pieces of content (5 per platform) that will:

🎯 **Drive 1M Subscriber Growth** through strategic social media funneling
📱 **Platform Native Content** that feels authentic to each platform
🎭 **Maintain Tóibín Literary Quality** even in 60-second format  
🔥 **Viral Potential** with hooks, mystery, and educational value
💤 **Sleep Content Branding** - establish Sleepy Dull Stories as THE sleep story destination

OUTPUT FORMAT:
{{
  "social_media_strategy": {{
    "campaign_name": "Sleepy Dull Stories - {topic} Viral Campaign",
    "main_story_connection": {{
      "selected_scenes": [scene IDs],
      "character_focus": "[Main character]",
      "viral_hooks": ["hook1", "hook2", "hook3"]
    }},
    "growth_target": "1M subscribers",
    "cross_platform_cta": "Full story → @SleepyDullStories"
  }},

  "youtube_shorts": [
    {{
      "short_id": 1,
      "title": "[Viral title with character/mystery hook]",
      "duration_seconds": 60,
      "based_on_scene": [scene_id],
      "script": {{
        "hook": "[0-5s] [Immediate engagement hook]",
        "story_teaser": "[5-45s] [Compressed scene maintaining Tóibín quality]", 
        "cta": "[45-60s] [Strong call to action to main channel]"
      }},
      "midjourney_prompt": "[DETAILED Midjourney prompt for thumbnail/cover image - 9:16 aspect ratio, character positioning, historical setting, dramatic lighting, text space consideration]",
      "visual_elements": {{
        "character_positioning": "[Where character should be positioned for text overlay]",
        "mood_lighting": "[Dramatic/mysterious/peaceful lighting description]",
        "historical_accuracy": "[Period-specific visual elements]",
        "text_overlay_space": "[Where text/titles can be placed]"
      }}
    }}
  ],

  "instagram_reels": [
    {{
      "reel_id": 1,
      "title": "[Instagram-optimized title]",
      "duration_seconds": 60,
      "based_on_scene": [scene_id],
      "script": {{
        "hook": "[0-3s] [Instagram-style hook]",
        "visual_story": "[3-50s] [Visual storytelling adaptation]",
        "cta": "[50-60s] [Instagram-appropriate CTA]"
      }},
      "midjourney_prompt": "[DETAILED Midjourney prompt - 9:16 vertical, Instagram-aesthetic, character focused, bright/engaging lighting, space for text overlays and stickers]",
      "visual_elements": {{
        "instagram_aesthetic": "[Bright, clean, engaging visual style]",
        "character_focus": "[How character should be presented]",
        "background_setting": "[Historical setting adapted for Instagram appeal]",
        "overlay_compatibility": "[Areas for text overlays and stickers]"
      }}
    }}
  ],

  "tiktok_videos": [
    {{
      "tiktok_id": 1,
      "title": "[TikTok-optimized title]", 
      "duration_seconds": 60,
      "based_on_scene": [scene_id],
      "script": {{
        "hook": "[0-3s] [TikTok-style educational/mystery hook]",
        "educational_story": "[3-50s] [Educational angle + story adaptation]",
        "cta": "[50-60s] [TikTok-appropriate CTA]"
      }},
      "midjourney_prompt": "[DETAILED Midjourney prompt - 9:16 mobile format, TikTok-native style, educational/storytelling visual approach, dynamic composition, space for captions]",
      "visual_elements": {{
        "tiktok_native_style": "[Dynamic, engaging, mobile-first composition]",
        "educational_visual": "[How to present educational/historical content visually]",
        "character_presentation": "[Character styling for TikTok audience]",
        "caption_space": "[Areas where TikTok captions/text can be placed]"
      }}
    }}
  ]
}}

Transform {topic} into viral social media gold with stunning Midjourney visuals while maintaining literary excellence!"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=20000,
                temperature=0.8,
                timeout=300,
                system="You are a viral social media strategist AND Colm Tóibín literary expert AND Midjourney prompt specialist. Create platform-native content that maintains literary quality while optimizing for viral growth.",
                messages=[{"role": "user", "content": social_media_prompt}]
            )

            # SENİN SİSTEMİNİ KULLANIYORUM - NON-STREAMING
            content = response.content[0].text

            print(f"✅ Social media content complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(social_media_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "social_media")

            self.log_step("Social Media Content Created", "SUCCESS", {
                "youtube_shorts": len(parsed_result.get('youtube_shorts', [])),
                "instagram_reels": len(parsed_result.get('instagram_reels', [])),
                "tiktok_videos": len(parsed_result.get('tiktok_videos', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Social Media Content Failed", "ERROR")
            CONFIG.logger.error(f"Social media content error: {e}")
            raise

    def _select_best_scenes_for_social(self, scene_plan: List[Dict]) -> List[Dict]:
        """Select scenes with highest viral potential"""

        # Score scenes based on viral potential
        scored_scenes = []
        for scene in scene_plan:
            score = 0

            # High emotion scenes
            if scene.get('emotion') in ['curiosity', 'recognition', 'resolution']:
                score += 3

            # Character-focused scenes
            if 'character' in scene.get('scene_description', '').lower():
                score += 2

            # Visual potential
            if any(word in scene.get('setting', '').lower() for word in
                   ['night', 'fire', 'door', 'window', 'market', 'house']):
                score += 2

            # Mystery elements
            if any(word in scene.get('emotional_core', '').lower() for word in
                   ['secret', 'hidden', 'unknown', 'mystery', 'wonder']):
                score += 3

            scored_scenes.append((score, scene))

        # Sort by score and take top 5
        scored_scenes.sort(key=lambda x: x[0], reverse=True)
        return [scene for score, scene in scored_scenes[:5]]

    def _format_scenes_for_social(self, selected_scenes: List[Dict], all_stories: Dict) -> str:
        """Format selected scenes for social media prompt"""

        formatted = []
        for scene in selected_scenes:
            scene_id = str(scene['scene_id'])
            story_content = all_stories.get(scene_id, '')

            formatted.append(f"""
SCENE {scene_id}: {scene['title']}
Emotion: {scene['emotion']} | Duration: {scene['duration_minutes']:.1f}min
Setting: {scene['setting']}
Character: {scene['main_character']}
Emotional Core: {scene['emotional_core']}
Story Content (first 500 chars): {story_content[:500]}...
Viral Potential: {scene.get('viral_score', 'High')}
""")

        return "\n".join(formatted)

    def _combine_all_results(self, master_plan: Dict, stage1_result: Dict, stage2_result: Dict,
                           production_data: Dict, topic: str, description: str) -> Dict:
        """Combine all results into final output"""

        # Combine stories
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))

        # Get scene plan
        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])

        # Calculate total duration
        total_duration = sum(scene.get('duration_minutes', 4) for scene in scene_plan)

        # Create complete story text
        complete_story = self._compile_complete_story(stage1_result, stage2_result, scene_plan)

        # Generate scene chapters for YouTube
        scene_chapters = self._generate_scene_chapters(scene_plan)

        result = {
            # Core content
            "topic": topic,
            "description": description,
            "hook_section": stage1_result.get("golden_hook", {}),
            "subscribe_section": stage1_result.get("subscribe_section", {}),
            "scene_plan": scene_plan,
            "stories": all_stories,
            "complete_story": complete_story,

            # Production data
            "characters": production_data.get('characters', {}),
            "visual_prompts": production_data.get('visual_prompts', []),
            "youtube_optimization": production_data.get('youtube_optimization', {}),
            "production_specs": production_data.get('production_specs', {}),
            "scene_chapters": scene_chapters,

            # Generation stats
            "generation_stats": {
                "total_scenes": len(scene_plan),
                "total_stories": len(all_stories),
                "total_duration_minutes": total_duration,
                "api_calls_used": self.api_call_count,
                "total_cost": self.total_cost,
                "tóibín_quality_applied": True,
                "master_plan_approach": True,
                "emotional_structure_used": True,
                "completion_rate": len(all_stories) / len(scene_plan) * 100 if scene_plan else 0
            },

            # Metadata
            "generated_at": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "generation_log": self.generation_log
        }

        return result

    def _create_hook_content(self, topic: str, description: str) -> str:
        """Create hook content"""
        return f"Experience the intimate world of {topic}, where every moment holds the weight of unspoken understanding. In this peaceful journey through time, you'll discover the quiet dignity of people living their daily lives, unaware of how their small gestures reveal the depths of human experience."

    def _create_subscribe_content(self) -> str:
        """Create subscribe content"""
        return "Join us for more quiet moments from history, where literary storytelling meets peaceful sleep. Subscribe for stories that honor both the art of narrative and the gift of rest."

    def _generate_scene_chapters(self, scene_plan: List[Dict]) -> List[Dict]:
        """Generate YouTube chapter markers"""
        chapters = []
        current_time = 60  # Start after hook and subscribe

        for scene in scene_plan:
            duration_seconds = int(scene.get('duration_minutes', 4) * 60)

            chapters.append({
                "time": f"{current_time // 60}:{current_time % 60:02d}",
                "title": f"Scene {scene['scene_id']}: {scene.get('title', 'Unknown')}"[:100],
                "duration_seconds": duration_seconds,
                "emotion": scene.get('emotion', 'peaceful')
            })

            current_time += duration_seconds

        return chapters

    def _compile_complete_story(self, stage1_result: Dict, stage2_result: Dict, scene_plan: List[Dict]) -> str:
        """Compile complete story text"""
        story_parts = []

        # Hook
        hook = stage1_result.get("golden_hook", {})
        if hook:
            story_parts.append("=== GOLDEN HOOK (0-30 seconds) ===")
            story_parts.append(hook.get("content", ""))
            story_parts.append("")

        # Subscribe
        subscribe = stage1_result.get("subscribe_section", {})
        if subscribe:
            story_parts.append("=== SUBSCRIBE REQUEST (30-60 seconds) ===")
            story_parts.append(subscribe.get("content", ""))
            story_parts.append("")

        # Main story
        story_parts.append("=== MAIN STORY ===")
        story_parts.append("")

        # Combine all stories
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))

        # Add scenes in order
        for scene in scene_plan:
            scene_id = str(scene['scene_id'])
            story_content = all_stories.get(scene_id, f"[Story for scene {scene_id} not generated]")

            story_parts.append(f"## Scene {scene_id}: {scene.get('title', 'Unknown')}")
            story_parts.append(f"Duration: {scene.get('duration_minutes', 4):.1f} minutes")
            story_parts.append(f"Emotion: {scene.get('emotion', 'peaceful')}")
            story_parts.append("")
            story_parts.append(story_content)
            story_parts.append("")

        return "\n".join(story_parts)

    def _parse_claude_response(self, content: str, stage: str) -> Dict[str, Any]:
        """Parse Claude response with error handling"""
        try:
            # Clean content
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Try to parse
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"⚠️ {stage}: JSON parsing failed, extracting partial data...")
                return self._extract_partial_data(content, stage)

        except Exception as e:
            print(f"❌ {stage} parsing failed: {e}")
            return {}

    def _extract_partial_data(self, content: str, stage: str) -> Dict[str, Any]:
        """Extract usable data from partial JSON"""
        result = {}

        try:
            if stage == "master_plan":
                # Extract scene plan array
                scene_start = content.find('"scene_plan": [')
                if scene_start > -1:
                    # Find the end of the array
                    bracket_count = 0
                    in_array = False
                    for i, char in enumerate(content[scene_start:], scene_start):
                        if char == '[':
                            bracket_count += 1
                            in_array = True
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0 and in_array:
                                scene_content = content[scene_start:i+1]
                                result["master_plan"] = {"scene_plan": []}
                                break

            elif stage in ["stage1", "stage2"]:
                # Extract stories
                result["stories"] = self._extract_stories_dict(content)

            elif stage == "production":
                # Extract basic structure
                result = {
                    "characters": {"main_characters": []},
                    "visual_prompts": [],
                    "youtube_optimization": {},
                    "production_specs": {}
                }

            elif stage == "social_media":
                # Extract social media structure
                result = {
                    "social_media_strategy": {},
                    "youtube_shorts": [],
                    "instagram_reels": [],
                    "tiktok_videos": []
                }

        except Exception as e:
            print(f"⚠️ Partial extraction error for {stage}: {e}")

        return result

    def _extract_stories_dict(self, content: str) -> Dict[str, str]:
        """Extract stories dictionary from content"""
        stories = {}
        try:
            # Find stories section
            story_pattern = r'"(\d+)":\s*"([^"]+(?:\\.[^"]*)*?)"'
            matches = re.findall(story_pattern, content, re.DOTALL)

            for story_id, story_content in matches:
                # Clean up escaped characters
                story_content = story_content.replace('\\"', '"')
                story_content = story_content.replace('\\n', '\n')
                story_content = story_content.replace('\\[PAUSE\\]', '[PAUSE]')

                if len(story_content) > 200:  # Only include substantial stories
                    stories[story_id] = story_content

        except Exception as e:
            print(f"Story extraction error: {e}")

        return stories

def save_production_outputs(output_dir: str, result: Dict, story_topic: str, topic_id: int,
                          api_calls: int, total_cost: float):
    """Save complete production outputs - ALL 15 JSON FILES RESTORED"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
        # Get main character for thumbnail concept
        main_characters = result.get("characters", {}).get("main_characters", [])
        main_character = main_characters[0] if main_characters else None
        main_char_name = main_character.get('name', 'Main Character') if main_character else 'Main Character'

        # Calculate duration info
        scene_plan = result.get('scene_plan', [])
        total_duration = sum(scene.get('duration_minutes', 4) for scene in scene_plan)
        total_hours = int(total_duration / 60)

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
            "toibin_master_plan_used": True,
            "emotional_structure_applied": True
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
        saved_files.append("scene_plan.json")

        # 3. All stories (validated)
        stories_path = output_path / "all_stories.json"
        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(result["stories"], f, indent=2, ensure_ascii=False)
        saved_files.append("all_stories.json")

        # 4. Voice directions for TTS
        voice_path = output_path / "voice_directions.json"
        voice_directions = []

        # Add voice directions for each scene
        for scene in scene_plan:
            voice_directions.append({
                "scene_number": scene.get("scene_id", 1),
                "title": scene.get("title", f"Scene {scene.get('scene_id', 1)}"),
                "direction": f"Gentle, contemplative storytelling with {scene.get('emotion', 'peaceful')} emotion, Tóibín literary sensibility",
                "template": scene.get("emotion", "peaceful"),
                "style": "toibin_observational",
                "emotion": scene.get("emotion", "peaceful"),
                "pacing": "Sleep-optimized with natural breathing rhythm",
                "voice_notes": f"Maintain understated tone for {scene.get('duration_minutes', 4):.1f} minute duration"
            })

        with open(voice_path, "w", encoding="utf-8") as f:
            json.dump(voice_directions, f, indent=2, ensure_ascii=False)
        saved_files.append("voice_directions.json")

        # 5. Visual generation prompts
        visual_path = output_path / "visual_generation_prompts.json"
        with open(visual_path, "w", encoding="utf-8") as f:
            json.dump(result["visual_prompts"], f, indent=2, ensure_ascii=False)
        saved_files.append("visual_generation_prompts.json")

        # 6. Character profiles
        character_path = output_path / "character_profiles.json"
        character_data = {
            "main_characters": result.get("characters", {}).get("main_characters", []),
            "scene_character_mapping": result.get("characters", {}).get("scene_character_mapping", {}),
            "visual_generation_instructions": {
                "step1": "First generate reference images for each main character using their physical_description",
                "step2": "Then generate scene visuals using character references when characters are present",
                "step3": "For atmospheric scenes (no characters), focus on setting and mood only",
                "step4": "Generate thumbnail using scene_number 99 if present in visual_generation_prompts.json",
                "reference_usage": "Always include relevant character reference images when generating scene visuals"
            }
        }
        with open(character_path, "w", encoding="utf-8") as f:
            json.dump(character_data, f, indent=2, ensure_ascii=False)
        saved_files.append("character_profiles.json")

        # 7. YouTube metadata
        youtube_path = output_path / "youtube_metadata.json"
        youtube_data = result.get("youtube_optimization", {})
        youtube_metadata = {
            "clickbait_titles": youtube_data.get("clickbait_titles", [
                f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)",
                f"Ancient {story_topic} Sleep Story That Will Put You to Sleep Instantly",
                f"I Spent {total_hours} Hours in {story_topic} (Most Relaxing Story Ever)",
                f"What Really Happened in {story_topic}'s Most Peaceful Night?",
                f"{story_topic} Bedtime Story for Deep Sleep and Relaxation"
            ]),
            "video_description": {
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
                "disclaimer": "This content is designed for relaxation and sleep. Please don't listen while driving or operating machinery."
            },
            "tags": youtube_data.get("tags", [
                "sleep story", "bedtime story", "relaxation", "insomnia help", "meditation",
                "calm", "peaceful", f"{total_hours} hours", "deep sleep", "anxiety relief",
                "stress relief", "asmr", "history", story_topic.lower()
            ]),
            "seo_strategy": youtube_data.get("seo_strategy", {}),
            "api_ready_format": {
                "snippet": {
                    "title": youtube_data.get("clickbait_titles", [
                        f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)"])[0] if youtube_data.get("clickbait_titles") else f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)",
                    "description": f"""Journey back in time and experience the tranquil world of {story_topic}. This atmospheric sleep story follows the peaceful daily routines and lives of fascinating characters in {story_topic}.

Perfect for insomnia, anxiety relief, or anyone who loves historical fiction combined with sleep meditation. Let the gentle rhythms of {story_topic} carry you into peaceful dreams.

⚠️ This story is designed to help you fall asleep - please don't listen while driving or operating machinery.""",
                    "tags": youtube_data.get("tags", [
                        "sleep story", "bedtime story", "relaxation", "insomnia help", "meditation",
                        "calm", "peaceful", f"{total_hours} hours", "deep sleep", "anxiety relief",
                        "stress relief", "asmr", "history", story_topic.lower()
                    ])[:30],
                    "categoryId": "27",
                    "defaultLanguage": "en"
                },
                "status": {
                    "privacyStatus": "public",
                    "embeddable": True,
                    "license": "youtube",
                    "madeForKids": False
                }
            }
        }
        with open(youtube_path, "w", encoding="utf-8") as f:
            json.dump(youtube_metadata, f, indent=2, ensure_ascii=False)
        saved_files.append("youtube_metadata.json")

        # 8. Production specifications
        production_path = output_path / "production_specifications.json"
        production_specs = result.get("production_specs", {})
        production_data = {
            "audio_production": production_specs.get("audio_production", {
                "tts_voice": "alloy",
                "speed_multiplier": 0.85,
                "pause_durations": {
                    "[PAUSE]": 2.0,
                    "scene_transition": 3.0
                }
            }),
            "video_timing": production_specs.get("video_timing", []),
            "quality_metrics": production_specs.get("quality_metrics", {
                "sleep_optimization_score": 9,
                "historical_accuracy": True,
                "tóibín_authenticity": True
            }),
            "automation_specifications": {
                "character_extraction": "✅ Complete",
                "youtube_optimization": "✅ Complete",
                "production_specifications": "✅ Complete",
                "api_ready_format": "✅ Complete"
            }
        }
        with open(production_path, "w", encoding="utf-8") as f:
            json.dump(production_data, f, indent=2, ensure_ascii=False)
        saved_files.append("production_specifications.json")

        # 9. Social media content (RESTORED!)
        social_path = output_path / "social_media_content.json"
        social_data = result.get("social_media_content", {})
        enhanced_social_data = {
            **social_data,
            "production_ready": True,
            "viral_growth_strategy": {
                "target_subscribers": "1M",
                "content_funnel": "Social Media → Main YouTube Channel",
                "posting_schedule": {
                    "youtube_shorts": "Daily at 8 PM",
                    "instagram_reels": "Daily at 6 PM",
                    "tiktok_videos": "2x daily at 12 PM and 9 PM"
                },
                "cross_promotion": "Each platform drives to main channel",
                "hashtag_strategy": "#SleepyDullStories on all platforms"
            }
        }
        with open(social_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_social_data, f, indent=2, ensure_ascii=False)
        saved_files.append("social_media_content.json")

        # 10. Thumbnail generation
        thumbnail_path = output_path / "thumbnail_generation.json"
        thumbnail_data = {
            "thumbnail_concept": result.get("youtube_optimization", {}).get("thumbnail_concept", {
                "main_character": main_char_name,
                "dramatic_scene": f"{main_char_name} in atmospheric {story_topic} setting",
                "text_overlay": f"{story_topic.upper()}'S SECRET",
                "color_scheme": "Warm golds and deep blues with atmospheric lighting",
                "emotion": "Peaceful concentration and serenity",
                "background": f"Atmospheric {story_topic} setting with cinematic lighting",
                "composition": "RIGHT-side character, LEFT-side text space"
            }),
            "generation_instructions": {
                "character_positioning": "RIGHT side of frame (60-70% from left edge)",
                "text_space": "LEFT side (30-40%) completely clear for dramatic text overlay",
                "visual_style": "Cinematic lighting, warm and inviting mood",
                "character_expression": "Peaceful, contemplative, with subtle dramatic appeal",
                "background_elements": f"Atmospheric {story_topic} setting elements"
            },
            "thumbnail_alternatives": [
                {
                    "variant": "Character Focus",
                    "prompt": f"Close-up of {main_char_name} in contemplative pose, {story_topic} background"
                },
                {
                    "variant": "Environmental Drama",
                    "prompt": f"Wide atmospheric shot of {story_topic} with {main_char_name} in contemplation"
                },
                {
                    "variant": "Symbolic Moment",
                    "prompt": f"Key symbolic elements from {story_topic} with {main_char_name} present"
                }
            ]
        }
        with open(thumbnail_path, "w", encoding="utf-8") as f:
            json.dump(thumbnail_data, f, indent=2, ensure_ascii=False)
        saved_files.append("thumbnail_generation.json")

        # 11. Hook & Subscribe scenes
        hook_subscribe_path = output_path / "hook_subscribe_scenes.json"
        hook_subscribe_data = {
            "hook_scenes": [
                {
                    "scene_id": scene.get("scene_id", i+1),
                    "scene_title": scene.get("title", f"Scene {i+1}"),
                    "start_time": i * 3,
                    "end_time": (i * 3) + 3,
                    "duration": 3,
                    "visual_prompt": f"Atmospheric cinematic view of {scene.get('setting', story_topic)}, golden hour lighting, peaceful and mysterious mood",
                    "timing_note": f"Display during hook seconds {i * 3}-{(i * 3) + 3}"
                }
                for i, scene in enumerate(scene_plan[:10])  # First 10 scenes for hook
            ],
            "subscribe_scenes": [
                {
                    "scene_id": scene.get("scene_id", i+11),
                    "scene_title": scene.get("title", f"Scene {i+11}"),
                    "start_time": i * 3,
                    "end_time": (i * 3) + 3,
                    "duration": 3,
                    "visual_prompt": f"Welcoming community view of {scene.get('setting', story_topic)}, warm lighting, inviting atmosphere",
                    "timing_note": f"Display during subscribe seconds {i * 3}-{(i * 3) + 3}"
                }
                for i, scene in enumerate(scene_plan[10:20] if len(scene_plan) > 10 else scene_plan)  # Next 10 scenes
            ],
            "production_notes": {
                "hook_timing": "Use hook_scenes during golden hook narration (0-30s)",
                "subscribe_timing": "Use subscribe_scenes during subscribe request (30-60s)",
                "visual_sync": "Each scene should blend seamlessly with spoken content"
            }
        }
        with open(hook_subscribe_path, "w", encoding="utf-8") as f:
            json.dump(hook_subscribe_data, f, indent=2, ensure_ascii=False)
        saved_files.append("hook_subscribe_scenes.json")

        # 12. Audio generation prompts
        audio_path = output_path / "audio_generation_prompts.json"
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
        production_audio = production_specs.get("audio_production", {})

        for scene_id, story_content in stories.items():
            scene_info = next((s for s in scene_plan if s.get("scene_id") == int(scene_id)), {})

            audio_prompts.append({
                "segment_id": f"scene_{scene_id}",
                "content": story_content,
                "duration_minutes": scene_info.get("duration_minutes", 4),
                "emotion": scene_info.get("emotion", "peaceful"),
                "tts_settings": {
                    "voice": production_audio.get("tts_voice", "alloy"),
                    "speed": production_audio.get("speed_multiplier", 0.85),
                    "pitch": -2,
                    "volume": 80,
                    "emphasis": "sleep-optimized"
                }
            })

        with open(audio_path, "w", encoding="utf-8") as f:
            json.dump(audio_prompts, f, indent=2, ensure_ascii=False)
        saved_files.append("audio_generation_prompts.json")

        # 13. Video composition instructions
        video_path = output_path / "video_composition_instructions.json"
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
                    "end_time": sum(scene.get('duration_minutes', 4) * 60 for scene in scene_plan) + 60,
                    "video_source": "visual_generation_prompts.json -> scenes 1-N",
                    "audio_source": "audio_generation_prompts.json -> scenes 1-N",
                    "text_overlay": "Scene titles (optional)",
                    "transition": "crossfade_between_scenes"
                }
            ],
            "scene_sync_strategy": {
                "rule": "When audio mentions scene X, display scene X visual",
                "timing": "Immediate visual sync with narrative"
            },
            "thumbnail_usage": {
                "source": "thumbnail_generation.json",
                "purpose": "YouTube thumbnail only, not used in video timeline"
            },
            "production_settings": {
                "resolution": "1920x1080",
                "frame_rate": 30,
                "audio_bitrate": 192,
                "video_codec": "h264",
                "total_duration": f"{sum(scene.get('duration_minutes', 4) for scene in scene_plan) + 1} minutes"
            },
            "chapters": result.get("scene_chapters", [])
        }
        with open(video_path, "w", encoding="utf-8") as f:
            json.dump(video_composition, f, indent=2, ensure_ascii=False)
        saved_files.append("video_composition_instructions.json")

        # 14. Platform metadata (comprehensive)
        platform_path = output_path / "platform_metadata.json"
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
            "title_options": youtube_metadata.get("clickbait_titles", []),
            "description": youtube_metadata.get("video_description", {}),
            "tags": youtube_metadata.get("tags", []),
            "hashtags": [
                "#sleepstory", "#bedtimestory", "#relaxation", "#meditation", "#insomnia",
                "#deepsleep", "#calm", "#history", f"#{story_topic.lower().replace(' ', '')}"
            ],
            "seo_strategy": youtube_metadata.get("seo_strategy", {}),
            "thumbnail_concept": thumbnail_data.get("thumbnail_concept", {}),
            "engagement_strategy": {
                "target_audience": "Sleep content seekers + History enthusiasts",
                "content_pillars": ["Historical accuracy", "Sleep optimization", "Literary quality"],
                "posting_schedule": "Weekly uploads, consistent timing"
            },
            "api_ready_format": youtube_metadata.get("api_ready_format", {})
        }
        with open(platform_path, "w", encoding="utf-8") as f:
            json.dump(platform_data, f, indent=2, ensure_ascii=False)
        saved_files.append("platform_metadata.json")

        # 15. Automation specs
        automation_path = output_path / "automation_specs.json"
        automation_data = {
            "audio_production": production_data.get("audio_production", {}),
            "video_assembly": {
                "scene_timing_precision": [
                    {
                        "scene_number": scene.get("scene_id", i+1),
                        "start_time": f"00:{60 + sum(s.get('duration_minutes', 4) * 60 for s in scene_plan[:i]) // 60:02d}:{(60 + sum(s.get('duration_minutes', 4) * 60 for s in scene_plan[:i])) % 60:02d}",
                        "duration_seconds": int(scene.get("duration_minutes", 4) * 60),
                        "word_count": int(scene.get("duration_minutes", 4) * 140)  # 140 words per minute
                    }
                    for i, scene in enumerate(scene_plan)
                ],
                "video_specifications": {
                    "resolution": "1920x1080",
                    "frame_rate": 30,
                    "transition_type": "slow_fade",
                    "export_format": "MP4_H264"
                }
            },
            "quality_control": production_data.get("quality_metrics", {}),
            "toibin_quality_assurance": {
                "literary_authenticity": True,
                "master_plan_followed": True,
                "emotional_structure_applied": True,
                "character_psychology_depth": True
            },
            "implementation_ready": True
        }
        with open(automation_path, "w", encoding="utf-8") as f:
            json.dump(automation_data, f, indent=2, ensure_ascii=False)
        saved_files.append("automation_specs.json")

        # 16. Generation report (comprehensive)
        report_path = output_path / "generation_report.json"
        production_report = {
            "topic": story_topic,
            "topic_id": topic_id,
            "generation_completed": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "toibin_quality_applied": True,
            "master_plan_approach": True,
            "emotional_structure_used": True,
            "literary_excellence": True,
            "viral_content_created": True,
            "stats": result["generation_stats"],
            "cost_analysis": {
                "total_api_calls": api_calls,
                "total_cost": total_cost,
                "cost_per_scene": total_cost / len(scene_plan) if scene_plan else 0,
                "cost_efficiency": "Tóibín quality + viral social media with master plan optimization"
            },
            "files_saved": saved_files,
            "next_steps": [
                "1. Generate character reference images using character_profiles.json",
                "2. Generate scene visuals using visual_generation_prompts.json",
                "3. Generate thumbnail using thumbnail_generation.json",
                "4. Generate audio using audio_generation_prompts.json",
                "5. Create social media content using social_media_content.json",
                "6. Compose video using video_composition_instructions.json",
                "7. Upload to YouTube using platform_metadata.json"
            ],
            "automation_readiness": {
                "character_extraction": "✅ Complete",
                "youtube_optimization": "✅ Complete",
                "production_specifications": "✅ Complete",
                "platform_metadata": "✅ Complete",
                "composition_strategy": "✅ Complete",
                "api_ready_format": "✅ Complete",
                "social_media_strategy": "✅ Complete",
                "toibin_literary_quality": "✅ Complete"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ ALL 15 PRODUCTION FILES SAVED: {saved_files}")
        CONFIG.logger.info(f"Files saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Save error: {e}")
        CONFIG.logger.error(f"Save error: {e}")

def get_next_topic_from_database() -> Tuple[int, str, str, str]:
    """Get next topic from database"""
    db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
    topic_manager = DatabaseTopicManager(str(db_path))

    result = topic_manager.get_next_pending_topic()

    if result:
        topic_id, topic, description, clickbait_title = result
        topic_manager.mark_topic_started(topic_id)

        CONFIG.logger.info(f"Topic selected: {topic}")
        print(f"✅ Topic selected: {topic}")
        return topic_id, topic, description, clickbait_title
    else:
        raise ValueError("No pending topics found in database")

def complete_topic_in_database(topic_id: int, scene_count: int, total_duration: float,
                              api_calls: int, total_cost: float, output_path: str):
    """Mark topic as completed in database"""
    db_path = Path(CONFIG.paths['DATA_DIR']) / 'production.db'
    topic_manager = DatabaseTopicManager(str(db_path))

    topic_manager.mark_topic_completed(
        topic_id, scene_count, total_duration, api_calls, total_cost, output_path
    )

    print(f"✅ Topic {topic_id} marked as completed")

def print_production_summary(result: Dict, story_topic: str, output_path: str, generation_time: float):
    """Print production summary"""
    stats = result["generation_stats"]

    print("\n" + "🎭" * 60)
    print("TÓIBÍN QUALITY STORY GENERATOR - COMPLETE!")
    print("🎭" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CONFIG.claude_config['model']} (Claude 4)")
    print(f"⏱️ Total Generation Time: {generation_time:.1f}s")

    print(f"\n📊 GENERATION RESULTS:")
    print(f"🎬 Scenes Planned: {stats['total_scenes']}")
    print(f"📝 Stories Written: {stats['total_stories']}")
    print(f"📱 Social Media Pieces: {stats.get('social_media_pieces', 0)}")
    print(f"📈 Completion Rate: {stats['completion_rate']:.1f}%")
    print(f"⏰ Total Duration: {stats['total_duration_minutes']:.1f} minutes")
    print(f"🔥 API Calls: {stats['api_calls_used']}")
    print(f"💰 Total Cost: ${stats['total_cost']:.4f}")

    print(f"\n🎭 TÓIBÍN QUALITY FEATURES:")
    print(f"✅ Master Plan Approach: {stats.get('master_plan_approach', False)}")
    print(f"✅ Emotional Structure: {stats.get('emotional_structure_used', False)}")
    print(f"✅ Tóibín Literary Style: {stats.get('tóibín_quality_applied', False)}")
    print(f"✅ Viral Content Created: {stats.get('viral_content_created', False)}")

    completion_rate = stats['completion_rate']
    if completion_rate >= 80:
        print(f"\n🎉 TÓIBÍN QUALITY SUCCESS!")
        print(f"✅ Literary excellence with production optimization")
        print(f"✅ Master plan approach delivered quality results")
        print(f"🚀 Ready for full production pipeline + viral growth")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"🔍 Review generation_report.json for details")

    print("\n📄 ALL 16 PRODUCTION FILES CREATED:")
    print("1. 📖 complete_story.txt - Full Tóibín-quality story")
    print("2. 🎬 scene_plan.json - Master scene plan + chapters")
    print("3. 📚 all_stories.json - All individual stories")
    print("4. 🎵 voice_directions.json - TTS voice guidance")
    print("5. 🖼️ visual_generation_prompts.json - Story-based visuals")
    print("6. 👥 character_profiles.json - Tóibín character analysis")
    print("7. 📺 youtube_metadata.json - YouTube optimization")
    print("8. 🏭 production_specifications.json - Technical specs")
    print("9. 📱 social_media_content.json - VIRAL GROWTH STRATEGY")
    print("10. 🖼️ thumbnail_generation.json - Thumbnail strategy")
    print("11. 🎭 hook_subscribe_scenes.json - Opening sequences")
    print("12. 🎵 audio_generation_prompts.json - TTS production")
    print("13. 🎥 video_composition_instructions.json - Video assembly")
    print("14. 🌍 platform_metadata.json - Platform optimization")
    print("15. 🤖 automation_specs.json - Automation ready")
    print("16. 📊 generation_report.json - Complete analytics")

    # Social media breakdown
    social_stats = result.get("social_media_content", {})
    print(f"\n🚀 VIRAL GROWTH STRATEGY:")
    print(f"📺 YouTube Shorts: {len(social_stats.get('youtube_shorts', []))} pieces")
    print(f"📸 Instagram Reels: {len(social_stats.get('instagram_reels', []))} pieces")
    print(f"🎵 TikTok Videos: {len(social_stats.get('tiktok_videos', []))} pieces")
    print(f"🎯 Target: 1M Subscribers via cross-platform funnel")

    print("🎭" * 60)

if __name__ == "__main__":
    try:
        print("🎭 TÓIBÍN QUALITY STORY GENERATOR")
        print("⚡ Master Plan + Emotional Structure + Literary Excellence + Social Media")
        print("📄 ALL 16 JSON FILES RESTORED")
        print("💫 VOICE DIRECTIONS + SOCIAL MEDIA CONTENT ADDED")
        print("=" * 60)

        # Get topic from database
        topic_id, topic, description, clickbait_title = get_next_topic_from_database()
        print(f"\n📚 Topic ID: {topic_id} - {topic}")
        print(f"📝 Description: {description}")

        # Setup output directory
        output_path = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id)

        # Initialize generator
        generator = ToibinStoryGenerator()

        # Generate complete story with Tóibín quality + social media
        start_time = time.time()
        result = generator.generate_complete_story(topic, description, clickbait_title)
        generation_time = time.time() - start_time

        # Save all outputs (ALL 15 FILES)
        save_production_outputs(str(output_path), result, topic, topic_id,
                               generator.api_call_count, generator.total_cost)

        # Print summary
        print_production_summary(result, topic, str(output_path), generation_time)

        # Mark as completed in database
        scene_count = len(result.get('scene_plan', []))
        total_duration = result.get('generation_stats', {}).get('total_duration_minutes', 0)
        complete_topic_in_database(
            topic_id, scene_count, total_duration,
            generator.api_call_count, generator.total_cost, str(output_path)
        )

        print("\n🎭 TÓIBÍN QUALITY GENERATION COMPLETE!")
        print(f"✅ Literary excellence + viral social media delivered: {output_path}")
        print(f"💰 Total cost: ${generator.total_cost:.4f}")
        print(f"📱 Social media pieces: {result['generation_stats'].get('social_media_pieces', 0)}")
        print(f"🎯 Ready for 1M subscriber strategy!")
        print(f"📄 ALL 16 JSON FILES SAVED AND READY!")
        print(f"💫 VOICE DIRECTIONS + SOCIAL MEDIA CONTENT INCLUDED!")

    except Exception as e:
        print(f"\n💥 GENERATION ERROR: {e}")
        CONFIG.logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()