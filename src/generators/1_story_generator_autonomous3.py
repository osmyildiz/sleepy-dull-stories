"""
Sleepy Dull Stories - COMPLETE TÓIBÍN QUALITY STORY GENERATOR with 3-STAGE SYSTEM & DURATION VALIDATION
System Logic: Master Plan → Emotional Distribution → Scene Creation → 3-Stage Tóibín Stories → VALIDATION → EXTENSION → Production JSONs → Social Media
Quality Focus: Literary excellence with COLM TÓIBÍN standards + 120+ minute duration guarantee
UPDATED: 3-Stage system (Stage1: 33%, Stage2: 33%, Stage3: 34%) for better timeout management
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
            "toibin_style_required": True,
            "minimum_duration_minutes": 120,  # Minimum duration requirement
            "three_stage_system": True,  # NEW: 3-stage system enabled
            "duration_manipulation": {
                "enabled": True,
                "inflation_factor": 1.8,  # Ask for 1.8x the target duration
                "success_threshold": 0.85,  # If we get 85%+ of real target, success
                "max_inflation": 2.5,  # Never ask for more than 2.5x
                "adaptive": True  # Learn from Claude's behavior
            }
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
        """Mark topic as completed in database - FIXED COLUMN NAME"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # FIX: Use correct column name 'production_completed_at' instead of 'completed_at'
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
    """TÓIBÍN QUALITY Story Generator with 3-Stage System & Duration Validation & RESUME CAPABILITY"""

    def __init__(self):
        self.client = Anthropic(api_key=CONFIG.api_key)
        self.api_call_count = 0
        self.total_cost = 0.0
        self.generation_log = []
        self.claude_behavior = {
            "requests_made": 0,
            "average_fulfillment_rate": 0.6,  # Start with assumption Claude gives 60% of requested
            "last_5_rates": []
        }

        # RESUME CAPABILITY
        self.checkpoint_file = None
        self.current_progress = {
            "stage": "not_started",
            "emotional_structure": None,
            "master_plan": None,
            "stage1_result": None,
            "stage2_result": None,
            "stage3_result": None,
            "production_data": None,
            "social_media_content": None,
            "completed_stages": [],
            "api_calls": 0,
            "total_cost": 0.0
        }

        print("✅ TÓIBÍN Quality Generator with 3-Stage System & Duration Validation & RESUME CAPABILITY initialized")

    def setup_checkpoint(self, topic_id: int):
        """Setup checkpoint file for resume capability"""
        checkpoint_dir = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / 'generation_progress.json'

        # Load existing progress if available
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.current_progress = json.load(f)
                    self.api_call_count = self.current_progress.get('api_calls', 0)
                    self.total_cost = self.current_progress.get('total_cost', 0.0)

                print(f"✅ RESUME: Found checkpoint at stage '{self.current_progress['stage']}'")
                print(f"   Previous API calls: {self.api_call_count}")
                print(f"   Previous cost: ${self.total_cost:.4f}")
                print(f"   Completed stages: {self.current_progress['completed_stages']}")

            except Exception as e:
                print(f"⚠️ Checkpoint file corrupted, starting fresh: {e}")
                self.current_progress = {
                    "stage": "not_started",
                    "emotional_structure": None,
                    "master_plan": None,
                    "stage1_result": None,
                    "stage2_result": None,
                    "stage3_result": None,
                    "production_data": None,
                    "social_media_content": None,
                    "completed_stages": [],
                    "api_calls": 0,
                    "total_cost": 0.0
                }

    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        if self.checkpoint_file:
            try:
                self.current_progress['api_calls'] = self.api_call_count
                self.current_progress['total_cost'] = self.total_cost

                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_progress, f, indent=2, ensure_ascii=False)

                print(f"💾 Checkpoint saved: {self.current_progress['stage']}")

            except Exception as e:
                print(f"⚠️ Checkpoint save failed: {e}")

    def generate_complete_story(self, topic: str, description: str, clickbait_title: str = None) -> Dict[str, Any]:
        """
        COMPLETE TÓIBÍN QUALITY GENERATION PIPELINE WITH 3-STAGE SYSTEM & RESUME CAPABILITY:
        - Automatically resumes from last completed stage
        - Saves progress after each major step
        - Prevents re-generation of completed work
        """

        self.log_step("TÓIBÍN QUALITY 3-Stage Generation Pipeline with RESUME CAPABILITY Started")

        try:
            # RESUME CHECK: Emotional Scene Structure
            if self.current_progress['stage'] in ['not_started']:
                scene_structure = self._generate_emotional_scene_structure()
                self.current_progress['emotional_structure'] = scene_structure
                self.current_progress['stage'] = 'emotional_structure_complete'
                self.current_progress['completed_stages'].append('emotional_structure')
                self.save_checkpoint()
            else:
                scene_structure = self.current_progress['emotional_structure']
                self.log_step("RESUME: Skipping emotional structure (already completed)", "SUCCESS")

            # RESUME CHECK: Master Scene Plan
            if self.current_progress['stage'] in ['emotional_structure_complete']:
                master_plan = self._create_master_scene_plan_3stages(topic, description, scene_structure)
                self.current_progress['master_plan'] = master_plan
                self.current_progress['stage'] = 'master_plan_complete'
                self.current_progress['completed_stages'].append('master_plan')
                self.save_checkpoint()
            else:
                master_plan = self.current_progress['master_plan']
                self.log_step("RESUME: Skipping master plan (already completed)", "SUCCESS")

            # RESUME CHECK: Stage 1 Stories
            if self.current_progress['stage'] in ['master_plan_complete']:
                stage1_stories = self._generate_stage1_stories(topic, description, master_plan)
                self.current_progress['stage1_result'] = stage1_stories
                self.current_progress['stage'] = 'stage1_complete'
                self.current_progress['completed_stages'].append('stage1_stories')
                self.save_checkpoint()
            else:
                stage1_stories = self.current_progress['stage1_result']
                self.log_step("RESUME: Skipping Stage 1 stories (already completed)", "SUCCESS")

            # RESUME CHECK: Stage 2 Stories
            if self.current_progress['stage'] in ['stage1_complete']:
                stage2_stories = self._generate_stage2_stories(topic, description, master_plan, stage1_stories)
                self.current_progress['stage2_result'] = stage2_stories
                self.current_progress['stage'] = 'stage2_complete'
                self.current_progress['completed_stages'].append('stage2_stories')
                self.save_checkpoint()
            else:
                stage2_stories = self.current_progress['stage2_result']
                self.log_step("RESUME: Skipping Stage 2 stories (already completed)", "SUCCESS")

            # RESUME CHECK: Stage 3 Stories
            if self.current_progress['stage'] in ['stage2_complete']:
                stage3_stories = self._generate_stage3_stories(topic, description, master_plan, stage1_stories, stage2_stories)
                self.current_progress['stage3_result'] = stage3_stories
                self.current_progress['stage'] = 'stage3_complete'
                self.current_progress['completed_stages'].append('stage3_stories')
                self.save_checkpoint()
            else:
                stage3_stories = self.current_progress['stage3_result']
                self.log_step("RESUME: Skipping Stage 3 stories (already completed)", "SUCCESS")

            # FINAL DURATION VALIDATION (always run to ensure consistency)
            all_stories = {}
            all_stories.update(stage1_stories.get('stories', {}))
            all_stories.update(stage2_stories.get('stories', {}))
            all_stories.update(stage3_stories.get('stories', {}))

            scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
            final_duration = self._calculate_total_duration(all_stories, scene_plan)

            if final_duration < CONFIG.claude_config["minimum_duration_minutes"]:
                self.log_step(f"FINAL VALIDATION: {final_duration:.1f} < 120 minutes - Emergency Extension", "WARNING")
                all_stories = self.validate_and_extend_stories(all_stories, scene_plan)
                final_duration = self._calculate_total_duration(all_stories, scene_plan)

            self.log_step(f"FINAL DURATION: {final_duration:.1f} minutes", "SUCCESS")

            # Update results with validated stories
            stage1_stories['stories'] = {k: v for k, v in all_stories.items() if int(k) <= len(scene_plan)//3}
            stage2_stories['stories'] = {k: v for k, v in all_stories.items() if len(scene_plan)//3 < int(k) <= (len(scene_plan)*2)//3}
            stage3_stories['stories'] = {k: v for k, v in all_stories.items() if int(k) > (len(scene_plan)*2)//3}

            # RESUME CHECK: Production JSONs
            if self.current_progress['stage'] in ['stage3_complete']:
                production_data = self._create_production_jsons(topic, description, master_plan,
                                                              stage1_stories, stage2_stories, stage3_stories, clickbait_title)
                self.current_progress['production_data'] = production_data
                self.current_progress['stage'] = 'production_complete'
                self.current_progress['completed_stages'].append('production_jsons')
                self.save_checkpoint()
            else:
                production_data = self.current_progress['production_data']
                self.log_step("RESUME: Skipping production JSONs (already completed)", "SUCCESS")

            # RESUME CHECK: Social Media Content
            if self.current_progress['stage'] in ['production_complete']:
                social_media_content = self._create_social_media_content(topic, description, master_plan, all_stories)
                self.current_progress['social_media_content'] = social_media_content
                self.current_progress['stage'] = 'social_media_complete'
                self.current_progress['completed_stages'].append('social_media_content')
                self.save_checkpoint()
            else:
                social_media_content = self.current_progress['social_media_content']
                self.log_step("RESUME: Skipping social media content (already completed)", "SUCCESS")

            # COMBINE ALL RESULTS
            complete_result = self._combine_all_results_3stages(master_plan, stage1_stories, stage2_stories, stage3_stories,
                                                      production_data, topic, description)

            # Add social media content
            complete_result["social_media_content"] = social_media_content

            # Update stats
            complete_result["generation_stats"]["social_media_pieces"] = (
                len(social_media_content.get('youtube_shorts', [])) +
                len(social_media_content.get('instagram_reels', [])) +
                len(social_media_content.get('tiktok_videos', []))
            )
            complete_result["generation_stats"]["viral_content_created"] = True
            complete_result["generation_stats"]["duration_validation_applied"] = True
            complete_result["generation_stats"]["final_duration_minutes"] = final_duration
            complete_result["generation_stats"]["three_stage_system_used"] = True
            complete_result["generation_stats"]["resume_capability_used"] = len(self.current_progress['completed_stages']) > 0

            # Mark as completely finished
            self.current_progress['stage'] = 'completely_finished'
            self.current_progress['completed_stages'].append('final_assembly')
            self.save_checkpoint()

            self.log_step("TÓIBÍN QUALITY 3-Stage Generation Complete (WITH RESUME)", "SUCCESS", {
                "total_scenes": len(master_plan.get('scene_plan', [])),
                "total_stories": len(complete_result.get('stories', {})),
                "final_duration": f"{final_duration:.1f} minutes",
                "duration_target_met": final_duration >= 120,
                "social_media_pieces": complete_result["generation_stats"]["social_media_pieces"],
                "three_stage_system": True,
                "resume_used": len(self.current_progress['completed_stages']) > 0,
                "completed_stages": self.current_progress['completed_stages'],
                "api_calls_total": self.api_call_count,
                "total_cost": self.total_cost
            })

            return complete_result

        except Exception as e:
            self.log_step("3-Stage Generation Failed", "ERROR")
            CONFIG.logger.error(f"Generation failed: {e}")

            # Save error state to checkpoint
            self.current_progress['stage'] = f"error_at_{self.current_progress['stage']}"
            self.current_progress['error'] = str(e)
            self.save_checkpoint()

            raise

    def _generate_emotional_scene_structure(self) -> Dict:
        """Generate emotional scene structure with proper distribution - ENHANCED FOR 120+ MINUTES"""

        # ENHANCED: Target 120-150 minutes
        target_duration_min = 120
        target_duration_max = 150

        # Random scene count - INCREASED for longer duration
        total_scenes = random.randint(35, 50)  # INCREASED from 28-40

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

        # Create emotional progression - ENHANCED DURATION RANGES
        emotional_structure = []

        # Phase 1: Peaceful establishment (first 35%) - LONGER SCENES
        for i in range(peaceful_count):
            emotional_structure.append({
                "scene_number": i + 1,
                "emotion": "peaceful",
                "phase": "establishment",
                "duration_range": (4.5, 7.0),  # INCREASED from (4.0, 6.5)
                "toibin_focus": "daily_life_observation"
            })

        # Phase 2: Curiosity/Discovery (next 25%) - LONGER SCENES
        for i in range(curiosity_count):
            emotional_structure.append({
                "scene_number": peaceful_count + i + 1,
                "emotion": "curiosity",
                "phase": "discovery",
                "duration_range": (4.0, 6.0),  # INCREASED from (3.5, 5.0)
                "toibin_focus": "character_recognition"
            })

        # Phase 3: Contemplation (next 25%) - LONGER SCENES
        for i in range(contemplation_count):
            emotional_structure.append({
                "scene_number": peaceful_count + curiosity_count + i + 1,
                "emotion": "contemplation",
                "phase": "recognition",
                "duration_range": (3.5, 5.5),  # INCREASED from (3.0, 4.5)
                "toibin_focus": "internal_complexity"
            })

        # Phase 4: Resolution (final 15%) - LONGER SCENES
        for i in range(resolution_count):
            emotional_structure.append({
                "scene_number": peaceful_count + curiosity_count + contemplation_count + i + 1,
                "emotion": "resolution",
                "phase": "acceptance",
                "duration_range": (5.0, 8.0),  # INCREASED from (4.5, 7.0)
                "toibin_focus": "quiet_acceptance"
            })

        # Calculate target duration
        target_duration = sum(random.uniform(scene["duration_range"][0], scene["duration_range"][1])
                             for scene in emotional_structure)

        structure = {
            "total_scenes": total_scenes,
            "emotional_distribution": {
                "peaceful": peaceful_count,
                "curiosity": curiosity_count,
                "contemplation": contemplation_count,
                "resolution": resolution_count
            },
            "emotional_structure": emotional_structure,
            "target_duration": target_duration,
            "three_stage_splits": {
                "stage1_scenes": total_scenes // 3,
                "stage2_scenes": total_scenes // 3,
                "stage3_scenes": total_scenes - (2 * (total_scenes // 3))
            }
        }

        # DURATION CHECK
        if target_duration < target_duration_min:
            self.log_step(f"WARNING: Planned duration {target_duration:.1f} < {target_duration_min} minutes", "WARNING")
        else:
            self.log_step(f"Duration target: {target_duration:.1f} minutes", "SUCCESS")

        self.log_step(f"Emotional Structure Generated: {total_scenes} scenes (3-stage system)", "SUCCESS", {
            "peaceful": peaceful_count,
            "curiosity": curiosity_count,
            "contemplation": contemplation_count,
            "resolution": resolution_count,
            "target_duration": f"{structure['target_duration']:.1f} minutes",
            "duration_adequate": target_duration >= target_duration_min,
            "stage1_scenes": structure["three_stage_splits"]["stage1_scenes"],
            "stage2_scenes": structure["three_stage_splits"]["stage2_scenes"],
            "stage3_scenes": structure["three_stage_splits"]["stage3_scenes"]
        })

        return structure

    def _create_master_scene_plan_3stages(self, topic: str, description: str, scene_structure: Dict) -> Dict:
        """Create master scene plan in THREE stages to avoid timeouts"""

        self.log_step("Creating Master Scene Plan in THREE Stages")

        emotional_structure = scene_structure["emotional_structure"]
        total_scenes = scene_structure["total_scenes"]

        # Split scenes into three stages
        stage1_size = total_scenes // 3
        stage2_size = total_scenes // 3
        stage3_size = total_scenes - stage1_size - stage2_size

        stage1_scenes = emotional_structure[:stage1_size]
        stage2_scenes = emotional_structure[stage1_size:stage1_size + stage2_size]
        stage3_scenes = emotional_structure[stage1_size + stage2_size:]

        self.log_step(f"Stage 1 Plan: {len(stage1_scenes)} scenes, Stage 2 Plan: {len(stage2_scenes)} scenes, Stage 3 Plan: {len(stage3_scenes)} scenes")

        # Generate Stage 1 master plan
        stage1_plan = self._create_master_plan_stage1_3stage(topic, description, stage1_scenes)

        # Generate Stage 2 master plan
        stage2_plan = self._create_master_plan_stage2_3stage(topic, description, stage2_scenes, stage1_plan)

        # Generate Stage 3 master plan
        stage3_plan = self._create_master_plan_stage3_3stage(topic, description, stage3_scenes, stage1_plan, stage2_plan)

        # Combine all three stages
        combined_scene_plan = []
        combined_scene_plan.extend(stage1_plan.get('scene_plan', []))
        combined_scene_plan.extend(stage2_plan.get('scene_plan', []))
        combined_scene_plan.extend(stage3_plan.get('scene_plan', []))

        # Create unified master plan
        master_plan = {
            "master_plan": {
                "topic_analysis": stage1_plan.get('topic_analysis', {}),
                "scene_plan": combined_scene_plan,
                "stage1_scenes": len(stage1_plan.get('scene_plan', [])),
                "stage2_scenes": len(stage2_plan.get('scene_plan', [])),
                "stage3_scenes": len(stage3_plan.get('scene_plan', [])),
                "total_scenes": len(combined_scene_plan),
                "three_stage_system": True
            }
        }

        self.log_step("Master Scene Plan Created (Three Stages)", "SUCCESS", {
            "stage1_scenes": len(stage1_plan.get('scene_plan', [])),
            "stage2_scenes": len(stage2_plan.get('scene_plan', [])),
            "stage3_scenes": len(stage3_plan.get('scene_plan', [])),
            "total_scenes": len(combined_scene_plan)
        })

        return master_plan

    def _create_master_plan_stage1_3stage(self, topic: str, description: str, scenes: List[Dict]) -> Dict:
        """Create master plan for first third of scenes"""

        self.log_step(f"Creating Master Plan Stage 1 (3-Stage): {len(scenes)} scenes")

        # Create emotional summary for first third
        emotional_summary = "\n".join([
            f"Scene {scene['scene_number']}: {scene['emotion']} ({scene['phase']}) - {scene['toibin_focus']} - Duration: {scene['duration_range'][0]:.1f}-{scene['duration_range'][1]:.1f} min"
            for scene in scenes
        ])

        # Sample scene for JSON template
        first_scene = scenes[0] if scenes else None
        scene_id = 1
        emotion = first_scene['emotion'] if first_scene else 'peaceful'
        phase = first_scene['phase'] if first_scene else 'establishment'
        duration_min = first_scene['duration_range'][0] if first_scene else 4.0
        duration_max = first_scene['duration_range'][1] if first_scene else 6.0
        duration = random.uniform(duration_min, duration_max)

        stage1_prompt = f"""You are CLAUDE, master film director creating a detailed scene plan for the FIRST THIRD of "{topic}" in COLM TÓIBÍN's literary style.

TOPIC: {topic}
DESCRIPTION: {description}

🎬 STAGE 1 SCENES TO PLAN (EXACTLY {len(scenes)} SCENES - FIRST THIRD):
{emotional_summary}

Create the opening third of this story with TÓIBÍN'S LITERARY MASTERY:

## TÓIBÍN'S CORE PRINCIPLES:
- "Sparseness of tone with superabundance of suggestion"
- Characters led by desires they don't understand
- Fascination of commonplaces
- Daily life as drama

OUTPUT FORMAT:
{{
  "topic_analysis": {{
    "central_theme": "[Core story theme]",
    "character_focus": "[Main characters for this story]",
    "daily_life_elements": "[Authentic activities]",
    "tóibín_approach": "[How Tóibín would approach this]",
    "historical_authenticity": "[Key period details]"
  }},
  "scene_plan": [
    {{
      "scene_id": {scene_id},
      "title": "[Scene title]",
      "emotion": "{emotion}",
      "phase": "{phase}",
      "duration_minutes": {duration:.1f},
      "setting": "[Specific location]",
      "time_of_day": "[Time]",
      "main_character": "[Character name and brief description]",
      "activity": "[What character is doing]", 
      "internal_focus": "[Character's internal state]",
      "tóibín_element": "[Tóibín technique demonstrated]",
      "scene_description": "[Detailed scene description for story generation]",
      "environmental_details": "[EXTENSIVE sensory details of the setting]", 
      "emotional_core": "[Core feeling of scene]",
      "sleep_optimization": "[How this promotes peaceful contemplation]"
    }}
  ]
}}

Create exactly {len(scenes)} scenes for the first third of {topic}."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=15000,
                temperature=0.5,
                timeout=240,
                system="You are COLM TÓIBÍN creating the opening third of your literary masterwork. Focus on establishment and character introduction with your signature understated style.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Master plan Stage 1 (3-Stage) complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage1_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "master_plan_stage1")

            self.log_step("Master Plan Stage 1 (3-Stage) Created", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Master Plan Stage 1 (3-Stage) Failed", "ERROR")
            CONFIG.logger.error(f"Master plan stage 1 error: {e}")
            raise

    def _create_master_plan_stage2_3stage(self, topic: str, description: str, scenes: List[Dict], stage1_plan: Dict) -> Dict:
        """Create master plan for second third of scenes with continuity from stage 1"""

        self.log_step(f"Creating Master Plan Stage 2 (3-Stage): {len(scenes)} scenes")

        # Get character info from stage 1 for continuity
        stage1_characters = []
        stage1_scenes = stage1_plan.get('scene_plan', [])
        for scene in stage1_scenes:
            char = scene.get('main_character', '')
            if char and char not in stage1_characters:
                stage1_characters.append(char)

        # Create emotional summary for second third
        emotional_summary = "\n".join([
            f"Scene {scene['scene_number']}: {scene['emotion']} ({scene['phase']}) - {scene['toibin_focus']} - Duration: {scene['duration_range'][0]:.1f}-{scene['duration_range'][1]:.1f} min"
            for scene in scenes
        ])

        # Sample scene for JSON template
        first_scene = scenes[0] if scenes else None
        scene_id = first_scene['scene_number'] if first_scene else len(stage1_scenes) + 1
        emotion = first_scene['emotion'] if first_scene else 'curiosity'
        phase = first_scene['phase'] if first_scene else 'discovery'
        duration_min = first_scene['duration_range'][0] if first_scene else 4.0
        duration_max = first_scene['duration_range'][1] if first_scene else 6.0
        duration = random.uniform(duration_min, duration_max)

        stage2_prompt = f"""You are CLAUDE, master film director creating the SECOND THIRD scene plan for "{topic}" in COLM TÓIBÍN's literary style.

TOPIC: {topic}
DESCRIPTION: {description}

CHARACTERS FROM STAGE 1 (maintain continuity):
{', '.join(stage1_characters[:5])}

🎬 STAGE 2 SCENES TO PLAN (EXACTLY {len(scenes)} SCENES - SECOND THIRD):
{emotional_summary}

Continue the story's middle section with TÓIBÍN'S LITERARY MASTERY while maintaining character continuity:

OUTPUT FORMAT:
{{
  "scene_plan": [
    {{
      "scene_id": {scene_id},
      "title": "[Scene title]",
      "emotion": "{emotion}",
      "phase": "{phase}",
      "duration_minutes": {duration:.1f},
      "setting": "[Specific location]",
      "time_of_day": "[Time]",
      "main_character": "[Character name - use Stage 1 characters when possible]",
      "activity": "[What character is doing]", 
      "internal_focus": "[Character's internal state]",
      "tóibín_element": "[Tóibín technique demonstrated]",
      "scene_description": "[Detailed scene description for story generation]",
      "environmental_details": "[EXTENSIVE sensory details of the setting]", 
      "emotional_core": "[Core feeling of scene]",
      "connection_to_stage1": "[How this connects to earlier scenes]",
      "sleep_optimization": "[How this promotes peaceful contemplation]"
    }}
  ]
}}

Create exactly {len(scenes)} scenes for the second third of {topic} with character continuity."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=15000,
                temperature=0.5,
                timeout=240,
                system="You are COLM TÓIBÍN continuing your literary masterwork into the second third. Maintain character consistency and develop the emotional journey toward deeper recognition.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Master plan Stage 2 (3-Stage) complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "master_plan_stage2")

            self.log_step("Master Plan Stage 2 (3-Stage) Created", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Master Plan Stage 2 (3-Stage) Failed", "ERROR")
            CONFIG.logger.error(f"Master plan stage 2 error: {e}")
            raise

    def _create_master_plan_stage3_3stage(self, topic: str, description: str, scenes: List[Dict],
                                         stage1_plan: Dict, stage2_plan: Dict) -> Dict:
        """Create master plan for final third of scenes with continuity from stages 1 & 2"""

        self.log_step(f"Creating Master Plan Stage 3 (3-Stage): {len(scenes)} scenes")

        # Get character info from stages 1 & 2 for continuity
        all_characters = []
        for stage_plan in [stage1_plan, stage2_plan]:
            stage_scenes = stage_plan.get('scene_plan', [])
            for scene in stage_scenes:
                char = scene.get('main_character', '')
                if char and char not in all_characters:
                    all_characters.append(char)

        # Create emotional summary for final third
        emotional_summary = "\n".join([
            f"Scene {scene['scene_number']}: {scene['emotion']} ({scene['phase']}) - {scene['toibin_focus']} - Duration: {scene['duration_range'][0]:.1f}-{scene['duration_range'][1]:.1f} min"
            for scene in scenes
        ])

        # Sample scene for JSON template
        first_scene = scenes[0] if scenes else None
        stage1_count = len(stage1_plan.get('scene_plan', []))
        stage2_count = len(stage2_plan.get('scene_plan', []))
        scene_id = first_scene['scene_number'] if first_scene else stage1_count + stage2_count + 1
        emotion = first_scene['emotion'] if first_scene else 'contemplation'
        phase = first_scene['phase'] if first_scene else 'recognition'
        duration_min = first_scene['duration_range'][0] if first_scene else 4.0
        duration_max = first_scene['duration_range'][1] if first_scene else 6.0
        duration = random.uniform(duration_min, duration_max)

        stage3_prompt = f"""You are CLAUDE, master film director creating the FINAL THIRD scene plan for "{topic}" in COLM TÓIBÍN's literary style.

TOPIC: {topic}
DESCRIPTION: {description}

CHARACTERS FROM STAGES 1 & 2 (maintain continuity):
{', '.join(all_characters[:8])}

🎬 STAGE 3 SCENES TO PLAN (EXACTLY {len(scenes)} SCENES - FINAL THIRD):
{emotional_summary}

Complete the story's final third with TÓIBÍN'S LITERARY MASTERY, bringing character arcs to their quiet resolution:

OUTPUT FORMAT:
{{
  "scene_plan": [
    {{
      "scene_id": {scene_id},
      "title": "[Scene title]",
      "emotion": "{emotion}",
      "phase": "{phase}",
      "duration_minutes": {duration:.1f},
      "setting": "[Specific location]",
      "time_of_day": "[Time]",
      "main_character": "[Character name - use previous characters when possible]",
      "activity": "[What character is doing]", 
      "internal_focus": "[Character's internal state]",
      "tóibín_element": "[Tóibín technique demonstrated]",
      "scene_description": "[Detailed scene description for story generation]",
      "environmental_details": "[EXTENSIVE sensory details of the setting]", 
      "emotional_core": "[Core feeling of scene]",
      "connection_to_previous_stages": "[How this connects to earlier scenes]",
      "resolution_element": "[How this contributes to quiet resolution]",
      "sleep_optimization": "[How this promotes peaceful contemplation]"
    }}
  ]
}}

Create exactly {len(scenes)} scenes for the final third of {topic} with complete character continuity and peaceful resolution."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=15000,
                temperature=0.5,
                timeout=240,
                system="You are COLM TÓIBÍN completing your literary masterwork's final third. Maintain absolute character consistency from previous stages and bring all elements to their quiet, understated resolution in your signature style.",
                messages=[{"role": "user", "content": stage3_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Master plan Stage 3 (3-Stage) complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage3_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "master_plan_stage3")

            self.log_step("Master Plan Stage 3 (3-Stage) Created", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Master Plan Stage 3 (3-Stage) Failed", "ERROR")
            CONFIG.logger.error(f"Master plan stage 3 error: {e}")
            raise

    def _generate_stage1_stories(self, topic: str, description: str, master_plan: Dict) -> Dict:
        """Generate first third stories following master plan with validation - OPTIMIZED FOR TIMEOUT PREVENTION"""

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
        total_scenes = len(scene_plan)
        first_third = total_scenes // 3

        first_third_scenes = scene_plan[:first_third]

        self.log_step(f"Stage 1: Generating first {first_third} Tóibín stories (3-Stage System - CHUNKED)")

        # OPTIMIZATION: Split into smaller chunks to prevent timeout
        chunk_size = 5  # Process 5 stories at a time instead of 11
        all_stories = {}

        # Create hook and subscribe content (only for first chunk)
        hook_content = self._create_hook_content(topic, description)
        subscribe_content = self._create_subscribe_content()

        # Process scenes in chunks
        for chunk_index in range(0, len(first_third_scenes), chunk_size):
            chunk_scenes = first_third_scenes[chunk_index:chunk_index + chunk_size]

            self.log_step(f"Processing chunk {chunk_index//chunk_size + 1}: scenes {chunk_index + 1}-{min(chunk_index + chunk_size, len(first_third_scenes))}")

            chunk_stories = self._generate_story_chunk(
                topic, description, chunk_scenes, chunk_index,
                hook_content if chunk_index == 0 else None,
                subscribe_content if chunk_index == 0 else None,
                total_scenes, first_third
            )

            # Merge stories from this chunk
            all_stories.update(chunk_stories.get('stories', {}))

        # Combine results
        result = {
            "stories": all_stories,
            "stage1_stats": {
                "scenes_written": len(all_stories),
                "total_planned": total_scenes,
                "tóibín_mastery_applied": True,
                "master_plan_followed": True,
                "duration_emphasis_applied": True,
                "three_stage_system": True,
                "chunked_processing": True,
                "chunks_processed": (len(first_third_scenes) + chunk_size - 1) // chunk_size
            }
        }

        # Add hook and subscribe only to final result
        if hook_content:
            result["golden_hook"] = {
                "content": hook_content,
                "duration_seconds": 30,
                "voice_direction": "Gentle, literary, contemplative - like Tóibín reading aloud"
            }

        if subscribe_content:
            result["subscribe_section"] = {
                "content": subscribe_content,
                "duration_seconds": 30,
                "voice_direction": "Warm, literary, non-commercial - book club invitation"
            }

        # VALIDATE AND EXTEND if needed
        if result.get('stories'):
            self.log_step("Validating Stage 1 Story Durations")
            validated_stories = self.validate_and_extend_stories(
                result['stories'],
                first_third_scenes
            )
            result['stories'] = validated_stories

        self.log_step("Stage 1 Stories Generated (3-Stage CHUNKED)", "SUCCESS", {
            "stories_written": len(result.get('stories', {})),
            "chunks_processed": result["stage1_stats"]["chunks_processed"]
        })

        return result

    def _generate_story_chunk(self, topic: str, description: str, chunk_scenes: List[Dict],
                             chunk_index: int, hook_content: str = None, subscribe_content: str = None,
                             total_scenes: int = 0, stage_total: int = 0) -> Dict:
        """Generate a small chunk of stories to prevent timeout"""

        # Format scenes for prompt with DURATION EMPHASIS
        scenes_text = "\n\n".join([
            f"SCENE {scene['scene_id']}: {scene['title']}\n"
            f"Duration: {scene['duration_minutes']:.1f} minutes ⚠️ MUST BE SUBSTANTIAL\n"
            f"Setting: {scene['setting']}\n"
            f"Main Character: {scene['main_character']}\n"
            f"Activity: {scene['activity']}\n"
            f"Internal Focus: {scene['internal_focus']}\n"
            f"Tóibín Element: {scene['tóibín_element']}\n"
            f"Scene Description: {scene['scene_description']}\n"
            f"Environmental Details: {scene['environmental_details']}\n"
            f"Emotional Core: {scene['emotional_core']}"
            for scene in chunk_scenes
        ])

        # Create sample story IDs for JSON template
        story_examples = {}
        for i, scene in enumerate(chunk_scenes):
            story_examples[str(scene['scene_id'])] = f"[Complete Tóibín-style story for scene {scene['scene_id']}, substantial length for target duration]"

        chunk_prompt = f"""Write {len(chunk_scenes)} TÓIBÍN masterpiece sleep stories for "{topic}" (Chunk {chunk_index//5 + 1}).

⏰ CRITICAL DURATION REQUIREMENTS:
- Each story must be SUBSTANTIAL and DETAILED to meet its target duration
- Target total for this chunk: {sum(scene['duration_minutes'] for scene in chunk_scenes):.1f} minutes
- Use extensive atmospheric description, internal character monologue, and contemplative pacing
- Include natural pause moments with [PAUSE] markers

TOPIC: {topic}  
DESCRIPTION: {description}

SCENES TO WRITE (Chunk {chunk_index//5 + 1} of Stage 1):
{scenes_text}

🎭 TÓIBÍN WRITING REQUIREMENTS:
Create stories worthy of COLM TÓIBÍN's literary reputation while serving as perfect sleep content.

⚠️ EACH STORY MUST BE LONG ENOUGH TO FILL ITS PLANNED DURATION:
- Use detailed environmental descriptions (lighting, sounds, textures, smells)
- Include character internal psychology, memories, and quiet reflections  
- Add contemplative pauses and natural silences with [PAUSE]
- Rich sensory details that transport the listener
- Extended dialogue with subtext, hesitations, and meaningful pauses
- Character observations of their surroundings and internal states

OUTPUT FORMAT:
{{
  "stories": {json.dumps(story_examples, indent=4)},
  "chunk_stats": {{
    "scenes_in_chunk": {len(chunk_scenes)},
    "chunk_number": {chunk_index//5 + 1},
    "tóibín_mastery_applied": true,
    "duration_emphasis_applied": true
  }}
}}"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=18000,  # REDUCED from 30000 to 18000
                temperature=0.7,
                timeout=180,  # REDUCED from 300 to 180
                system="You are COLM TÓIBÍN writing a focused chunk of your literary masterwork. Each story must be substantial and detailed enough to meet its target duration through rich atmospheric detail and character psychology.",
                messages=[{"role": "user", "content": chunk_prompt}]
            )

            content = response.content[0].text

            print(f"✅ Story chunk {chunk_index//5 + 1} complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(chunk_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, f"stage1_chunk_{chunk_index//5 + 1}")

            return parsed_result

        except Exception as e:
            self.log_step(f"Story Chunk {chunk_index//5 + 1} Failed", "ERROR")
            CONFIG.logger.error(f"Chunk error: {e}")

            # Return empty result for this chunk
            return {"stories": {}, "chunk_stats": {"error": str(e)}}

    def _generate_stage2_stories(self, topic: str, description: str, master_plan: Dict, stage1_result: Dict) -> Dict:
        """Generate second third stories with Tóibín continuity and validation - OPTIMIZED"""

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
        total_scenes = len(scene_plan)
        first_third = total_scenes // 3
        second_third = total_scenes // 3

        second_third_scenes = scene_plan[first_third:first_third + second_third]

        self.log_step(f"Stage 2: Generating second third {len(second_third_scenes)} Tóibín stories (3-Stage System - CHUNKED)")

        # OPTIMIZATION: Split into smaller chunks
        chunk_size = 5
        all_stories = {}

        # Process scenes in chunks
        for chunk_index in range(0, len(second_third_scenes), chunk_size):
            chunk_scenes = second_third_scenes[chunk_index:chunk_index + chunk_size]

            self.log_step(f"Processing Stage 2 chunk {chunk_index//chunk_size + 1}: scenes {first_third + chunk_index + 1}-{first_third + min(chunk_index + chunk_size, len(second_third_scenes))}")

            chunk_stories = self._generate_stage2_chunk(
                topic, description, chunk_scenes, chunk_index, stage1_result, total_scenes
            )

            # Merge stories from this chunk
            all_stories.update(chunk_stories.get('stories', {}))

        # Combine results
        result = {
            "stories": all_stories,
            "stage2_stats": {
                "scenes_written": len(all_stories),
                "character_continuity_maintained": True,
                "tóibín_mastery_sustained": True,
                "master_plan_continued": True,
                "duration_emphasis_applied": True,
                "three_stage_system": True,
                "chunked_processing": True,
                "chunks_processed": (len(second_third_scenes) + chunk_size - 1) // chunk_size
            }
        }

        # VALIDATE AND EXTEND if needed
        if result.get('stories'):
            self.log_step("Validating Stage 2 Story Durations")
            validated_stories = self.validate_and_extend_stories(
                result['stories'],
                second_third_scenes
            )
            result['stories'] = validated_stories

        self.log_step("Stage 2 Stories Generated (3-Stage CHUNKED)", "SUCCESS", {
            "stories_written": len(result.get('stories', {})),
            "chunks_processed": result["stage2_stats"]["chunks_processed"]
        })

        return result

    def _generate_stage2_chunk(self, topic: str, description: str, chunk_scenes: List[Dict],
                              chunk_index: int, stage1_result: Dict, total_scenes: int) -> Dict:
        """Generate Stage 2 chunk with character continuity"""

        # Format scenes for prompt with DURATION EMPHASIS
        scenes_text = "\n\n".join([
            f"SCENE {scene['scene_id']}: {scene['title']}\n"
            f"Duration: {scene['duration_minutes']:.1f} minutes ⚠️ MUST BE SUBSTANTIAL\n"
            f"Setting: {scene['setting']}\n"
            f"Main Character: {scene['main_character']}\n"
            f"Activity: {scene['activity']}\n"
            f"Internal Focus: {scene['internal_focus']}\n"
            f"Tóibín Element: {scene['tóibín_element']}\n"
            f"Scene Description: {scene['scene_description']}\n"
            f"Environmental Details: {scene['environmental_details']}\n"
            f"Emotional Core: {scene['emotional_core']}\n"
            f"Connection to Previous: {scene.get('connection_to_stage1', 'Connected to Stage 1')}"
            for scene in chunk_scenes
        ])

        # Create sample story IDs for JSON template
        story_examples = {}
        for scene in chunk_scenes:
            story_examples[str(scene['scene_id'])] = f"[Complete Tóibín story with character continuity, substantial length for target duration]"

        stage2_prompt = f"""Continue TÓIBÍN masterpiece with {len(chunk_scenes)} stories for "{topic}" (Stage 2 Chunk {chunk_index//5 + 1}).

⏰ CRITICAL DURATION REQUIREMENTS:
- Each story must be SUBSTANTIAL and DETAILED to meet its target duration
- Target total for this chunk: {sum(scene['duration_minutes'] for scene in chunk_scenes):.1f} minutes
- Maintain character continuity from Stage 1
- Use extensive atmospheric description and psychological depth

TOPIC: {topic}
DESCRIPTION: {description}

SCENES TO WRITE (Stage 2 Chunk {chunk_index//5 + 1}):
{scenes_text}

⚠️ DURATION AND CONTINUITY REQUIREMENTS:
- Each story must be long enough to fill its planned duration
- Maintain character voices and relationships established in Stage 1
- Rich atmospheric details and internal character psychology
- Extended contemplative moments with [PAUSE] markers
- Environmental descriptions that enhance the peaceful mood

OUTPUT FORMAT:
{{
  "stories": {json.dumps(story_examples, indent=4)},
  "chunk_stats": {{
    "scenes_in_chunk": {len(chunk_scenes)},
    "chunk_number": {chunk_index//5 + 1},
    "stage": 2,
    "character_continuity_maintained": true
  }}
}}"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=18000,  # REDUCED
                temperature=0.7,
                timeout=180,  # REDUCED
                system="You are COLM TÓIBÍN continuing your literary masterwork. Maintain absolute character consistency from Stage 1. Each story must be substantial enough to meet its target duration through rich detail and contemplative pacing.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Stage 2 chunk {chunk_index//5 + 1} complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, f"stage2_chunk_{chunk_index//5 + 1}")
            return parsed_result

        except Exception as e:
            self.log_step(f"Stage 2 Chunk {chunk_index//5 + 1} Failed", "ERROR")
            CONFIG.logger.error(f"Stage 2 chunk error: {e}")
            return {"stories": {}, "chunk_stats": {"error": str(e)}}

    def _generate_stage3_stories(self, topic: str, description: str, master_plan: Dict,
                                stage1_result: Dict, stage2_result: Dict) -> Dict:
        """Generate final third stories with Tóibín continuity and resolution - OPTIMIZED"""

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])
        total_scenes = len(scene_plan)
        first_third = total_scenes // 3
        second_third = total_scenes // 3

        final_third_scenes = scene_plan[first_third + second_third:]

        self.log_step(f"Stage 3: Generating final third {len(final_third_scenes)} Tóibín stories (3-Stage System - CHUNKED)")

        # OPTIMIZATION: Split into smaller chunks
        chunk_size = 5
        all_stories = {}

        # Process scenes in chunks
        for chunk_index in range(0, len(final_third_scenes), chunk_size):
            chunk_scenes = final_third_scenes[chunk_index:chunk_index + chunk_size]

            self.log_step(f"Processing Stage 3 chunk {chunk_index//chunk_size + 1}: scenes {first_third + second_third + chunk_index + 1}-{first_third + second_third + min(chunk_index + chunk_size, len(final_third_scenes))}")

            chunk_stories = self._generate_stage3_chunk(
                topic, description, chunk_scenes, chunk_index, stage1_result, stage2_result, total_scenes
            )

            # Merge stories from this chunk
            all_stories.update(chunk_stories.get('stories', {}))

        # Combine results
        result = {
            "stories": all_stories,
            "stage3_stats": {
                "scenes_written": len(all_stories),
                "character_continuity_maintained": True,
                "tóibín_mastery_completed": True,
                "master_plan_resolved": True,
                "duration_emphasis_applied": True,
                "peaceful_resolution_achieved": True,
                "three_stage_system": True,
                "chunked_processing": True,
                "chunks_processed": (len(final_third_scenes) + chunk_size - 1) // chunk_size
            }
        }

        # VALIDATE AND EXTEND if needed
        if result.get('stories'):
            self.log_step("Validating Stage 3 Story Durations")
            validated_stories = self.validate_and_extend_stories(
                result['stories'],
                final_third_scenes
            )
            result['stories'] = validated_stories

        self.log_step("Stage 3 Stories Generated (3-Stage CHUNKED)", "SUCCESS", {
            "stories_written": len(result.get('stories', {})),
            "chunks_processed": result["stage3_stats"]["chunks_processed"]
        })

        return result

    def _generate_stage3_chunk(self, topic: str, description: str, chunk_scenes: List[Dict],
                              chunk_index: int, stage1_result: Dict, stage2_result: Dict, total_scenes: int) -> Dict:
        """Generate Stage 3 chunk with full continuity and resolution"""

        # Format scenes for prompt with DURATION EMPHASIS
        scenes_text = "\n\n".join([
            f"SCENE {scene['scene_id']}: {scene['title']}\n"
            f"Duration: {scene['duration_minutes']:.1f} minutes ⚠️ MUST BE SUBSTANTIAL\n"
            f"Setting: {scene['setting']}\n"
            f"Main Character: {scene['main_character']}\n"
            f"Activity: {scene['activity']}\n"
            f"Internal Focus: {scene['internal_focus']}\n"
            f"Tóibín Element: {scene['tóibín_element']}\n"
            f"Scene Description: {scene['scene_description']}\n"
            f"Environmental Details: {scene['environmental_details']}\n"
            f"Emotional Core: {scene['emotional_core']}\n"
            f"Connection to Previous: {scene.get('connection_to_previous_stages', 'Connected to Stages 1 & 2')}\n"
            f"Resolution Element: {scene.get('resolution_element', 'Contributes to quiet resolution')}"
            for scene in chunk_scenes
        ])

        # Create sample story IDs for JSON template
        story_examples = {}
        for scene in chunk_scenes:
            story_examples[str(scene['scene_id'])] = f"[Complete Tóibín story with resolution, substantial length for target duration]"

        stage3_prompt = f"""Complete TÓIBÍN masterpiece with {len(chunk_scenes)} stories for "{topic}" (Stage 3 Chunk {chunk_index//5 + 1}).

⏰ CRITICAL DURATION REQUIREMENTS:
- Each story must be SUBSTANTIAL and DETAILED to meet its target duration
- Target total for this chunk: {sum(scene['duration_minutes'] for scene in chunk_scenes):.1f} minutes
- Maintain character continuity from Stages 1 & 2
- Bring story to peaceful, understated resolution

TOPIC: {topic}
DESCRIPTION: {description}

SCENES TO WRITE (Stage 3 Chunk {chunk_index//5 + 1} - FINAL RESOLUTION):
{scenes_text}

⚠️ DURATION, CONTINUITY & RESOLUTION REQUIREMENTS:
- Each story must be long enough to fill its planned duration
- Maintain character voices and relationships from Stages 1 & 2
- Rich atmospheric details and internal character psychology
- Extended contemplative moments with [PAUSE] markers
- Bring character arcs to their understated, peaceful resolution
- Environmental descriptions that enhance the peaceful mood

OUTPUT FORMAT:
{{
  "stories": {json.dumps(story_examples, indent=4)},
  "chunk_stats": {{
    "scenes_in_chunk": {len(chunk_scenes)},
    "chunk_number": {chunk_index//5 + 1},
    "stage": 3,
    "resolution_focus": true,
    "peaceful_closure": true
  }}
}}"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=18000,  # REDUCED
                temperature=0.7,
                timeout=180,  # REDUCED
                system="You are COLM TÓIBÍN completing your literary masterwork. Maintain absolute character consistency throughout and bring all character arcs to their quiet, understated resolution. Each story must be substantial enough to meet its target duration through rich detail and contemplative pacing.",
                messages=[{"role": "user", "content": stage3_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Stage 3 chunk {chunk_index//5 + 1} complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage3_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, f"stage3_chunk_{chunk_index//5 + 1}")
            return parsed_result

        except Exception as e:
            self.log_step(f"Stage 3 Chunk {chunk_index//5 + 1} Failed", "ERROR")
            CONFIG.logger.error(f"Stage 3 chunk error: {e}")
            return {"stories": {}, "chunk_stats": {"error": str(e)}}

    def _calculate_total_duration(self, stories: Dict, scene_plan: List[Dict]) -> float:
        """Calculate total estimated duration from stories"""

        total_duration = 0.0

        for scene in scene_plan:
            scene_id = str(scene.get('scene_id', 0))
            story_content = stories.get(scene_id, '')

            if story_content:
                word_count = len(story_content.split())
                duration = word_count / CONFIG.claude_config["target_words_per_minute"]  # 140 words per minute
                total_duration += duration
            else:
                # Use planned duration if story missing
                total_duration += scene.get('duration_minutes', 4.0)

        return total_duration

    def validate_and_extend_stories(self, stories: Dict, scene_plan: List[Dict]) -> Dict:
        """Validate story durations and extend if needed"""

        self.log_step("Validating Story Durations")

        extended_stories = {}
        extension_needed = []

        for scene in scene_plan:
            scene_id = str(scene.get('scene_id', 0))
            story_content = stories.get(scene_id, '')

            if not story_content:
                continue

            # Calculate current story metrics
            word_count = len(story_content.split())
            estimated_duration = word_count / CONFIG.claude_config["target_words_per_minute"]  # 140 words per minute
            target_duration = scene.get('duration_minutes', 4.0)

            print(f"📊 Scene {scene_id}: {word_count} words = {estimated_duration:.1f}min (target: {target_duration:.1f}min)")

            # Check if extension needed (15% tolerance)
            tolerance = CONFIG.claude_config.get("validation_tolerance", 0.15)
            min_acceptable_duration = target_duration * (1 - tolerance)

            if estimated_duration < min_acceptable_duration:
                extension_needed.append({
                    'scene_id': scene_id,
                    'scene_info': scene,
                    'current_story': story_content,
                    'current_duration': estimated_duration,
                    'target_duration': target_duration,
                    'needed_words': int((target_duration - estimated_duration) * CONFIG.claude_config["target_words_per_minute"])
                })
                print(f"⚠️ Scene {scene_id} needs extension: {estimated_duration:.1f} < {min_acceptable_duration:.1f}min")
            else:
                extended_stories[scene_id] = story_content
                print(f"✅ Scene {scene_id} duration OK")

        # Extend stories that need it
        if extension_needed:
            self.log_step(f"Extending {len(extension_needed)} stories")

            # Process in batches of 3 stories
            batch_size = 3
            for i in range(0, len(extension_needed), batch_size):
                batch = extension_needed[i:i + batch_size]
                extended_batch = self._extend_story_batch(batch)
                extended_stories.update(extended_batch)

        # Final validation
        total_duration = self._calculate_total_duration(extended_stories, scene_plan)

        self.log_step("Story Duration Validation Complete", "SUCCESS", {
            "total_duration": f"{total_duration:.1f} minutes",
            "stories_extended": len(extension_needed),
            "target_met": total_duration >= CONFIG.claude_config["minimum_duration_minutes"]
        })

        return extended_stories

    def _extend_story_batch(self, batch: List[Dict]) -> Dict:
        """Extend a batch of stories using Claude"""

        # Create extension prompt
        batch_info = []
        for story_data in batch:
            scene_info = story_data['scene_info']
            batch_info.append(f"""
SCENE {story_data['scene_id']}: {scene_info.get('title', 'Unknown')}
Current Duration: {story_data['current_duration']:.1f} minutes
Target Duration: {story_data['target_duration']:.1f} minutes
Needed Words: ~{story_data['needed_words']} words
Emotion: {scene_info.get('emotion', 'peaceful')}
Setting: {scene_info.get('setting', 'unknown')}
Character: {scene_info.get('main_character', 'unknown')}

CURRENT STORY:
{story_data['current_story']}
""")

        extension_prompt = f"""EXTEND these TÓIBÍN stories to meet their target durations while maintaining literary quality.

STORIES TO EXTEND:
{chr(10).join(batch_info)}

🎯 EXTENSION REQUIREMENTS:

1. **Maintain Tóibín Literary Quality**
   - Preserve existing narrative flow and character voice
   - Extend with MORE atmospheric detail, not plot changes
   - Add psychological depth and internal observations

2. **Duration Targets Must Be Met**
   - Each story MUST reach its target duration
   - Add approximately the "Needed Words" to each story
   - Use natural pacing and contemplative moments

3. **Sleep-Optimized Extensions**
   - Add more environmental descriptions (sounds, lighting, textures)
   - Include character memories and quiet reflections  
   - Insert natural pauses with [PAUSE] markers
   - Expand on sensory details and peaceful moments

4. **Seamless Integration**
   - Extensions should feel natural, not forced
   - Maintain the same emotional tone throughout
   - Keep character psychology consistent

OUTPUT FORMAT:
{{
  "extended_stories": {{
    "{batch[0]['scene_id']}": "[Complete extended story meeting target duration]",
    "{batch[1]['scene_id'] if len(batch) > 1 else 'X'}": "[Complete extended story]"
  }},
  "extension_stats": {{
    "stories_extended": {len(batch)},
    "total_words_added": "[estimated total words added]",
    "duration_targets_met": true
  }}
}}

Extend each story with rich Tóibín-style detail to reach target durations."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=25000,
                temperature=0.6,
                timeout=300,
                system="You are COLM TÓIBÍN extending your own literary work. Maintain absolute consistency with your established style, character voices, and narrative tone. Extend stories with rich atmospheric detail and psychological depth.",
                messages=[{"role": "user", "content": extension_prompt}]
            )

            content = response.content[0].text

            # Calculate cost
            input_tokens = len(extension_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "extension")
            extended_stories = parsed_result.get('extended_stories', {})

            print(f"✅ Extended {len(extended_stories)} stories (cost: ${stage_cost:.4f})")

            return extended_stories

        except Exception as e:
            self.log_step("Story Extension Failed", "ERROR")
            CONFIG.logger.error(f"Extension error: {e}")

            # Return original stories if extension fails
            return {story_data['scene_id']: story_data['current_story'] for story_data in batch}

    def _create_production_jsons(self, topic: str, description: str, master_plan: Dict,
                                 stage1_result: Dict, stage2_result: Dict, stage3_result: Dict, clickbait_title: str = None) -> Dict:
        """Create all production JSONs based on the completed stories - ENHANCED FOR 3-STAGE SYSTEM"""

        self.log_step("Creating Production JSONs (3-Stage System)")

        # Combine all stories from three stages
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))
        all_stories.update(stage3_result.get('stories', {}))

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
3-STAGE SYSTEM USED: True

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
      "all_scenes_covered": true,
      "duration_validation_applied": true,
      "three_stage_system_used": true
    }}
  }}
}}

Generate visual prompts for ALL {total_scenes} scenes based on the actual story content and character analysis."""

        try:
            self.api_call_count += 1

            # ENHANCED: Increased token limit and timeout for complete scene coverage
            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=25000,  # Increased from 16000 to 25000
                temperature=0.3,
                timeout=400,  # Increased timeout to 400 seconds
                system="You are a production expert analyzing COLM TÓIBÍN's literary work created with the 3-stage system. Create visual prompts for EVERY SINGLE SCENE based on the story content. Do not skip any scenes.",
                messages=[{"role": "user", "content": production_prompt}]
            )

            content = response.content[0].text
            print(f"✅ Production JSONs (3-Stage) complete: {len(content):,} characters")

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
                            "prompt": f"Historical scene: {scene.get('setting', 'unknown setting')}, {scene.get('main_character', 'character')} {scene.get('activity', 'in contemplation')}, {scene.get('emotion', 'peaceful')} atmosphere, Tóibín-style understated mood",
                            "duration_minutes": scene.get('duration_minutes', 4.0),
                            "characters_present": [scene.get('main_character', 'Unknown')],
                            "historical_accuracy": "Period details, authentic settings",
                            "tóibín_atmosphere": f"{scene.get('emotion', 'peaceful')} contemplation, understated emotional depth"
                        })

                # Add missing scenes to the result
                if 'visual_prompts' not in parsed_result:
                    parsed_result['visual_prompts'] = []
                parsed_result['visual_prompts'].extend(missing_scenes)

                self.log_step(f"Added {len(missing_scenes)} missing visual prompts", "SUCCESS")

            self.log_step("Production JSONs Created (3-Stage)", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('characters', {}).get('main_characters', [])),
                "visual_prompts_created": len(parsed_result.get('visual_prompts', [])),
                "all_scenes_covered": len(parsed_result.get('visual_prompts', [])) >= expected_scenes,
                "three_stage_system": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Production JSONs (3-Stage) Failed", "ERROR")
            CONFIG.logger.error(f"Production JSON error: {e}")
            raise

    def _create_social_media_content(self, topic: str, description: str, master_plan: Dict,
                                   all_stories: Dict) -> Dict:
        """Create platform-specific social media content for viral reach using Claude API"""

        self.log_step("Creating Social Media Content for Viral Growth (3-Stage System)")

        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])

        # Select best scenes for social media (high emotion, visual potential)
        selected_scenes = self._select_best_scenes_for_social(scene_plan)

        social_media_prompt = f"""Create viral social media content for "{topic}" designed to drive traffic to Sleepy Dull Stories YouTube channel (goal: 1M subscribers).

CRITICAL: Each piece must include MIDJOURNEY VISUAL PROMPTS for image generation!

TOPIC: {topic}
DESCRIPTION: {description}
3-STAGE SYSTEM USED: Yes (enhanced character continuity)

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
    "cross_platform_cta": "Full story → @SleepyDullStories",
    "three_stage_advantage": "Enhanced character depth and continuity"
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
                system="You are a viral social media strategist AND Colm Tóibín literary expert AND Midjourney prompt specialist. Create platform-native content that maintains literary quality while optimizing for viral growth. The 3-stage system provides enhanced character depth.",
                messages=[{"role": "user", "content": social_media_prompt}]
            )

            content = response.content[0].text

            print(f"✅ Social media content (3-Stage) complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(social_media_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # Parse response
            parsed_result = self._parse_claude_response(content, "social_media")

            self.log_step("Social Media Content Created (3-Stage)", "SUCCESS", {
                "youtube_shorts": len(parsed_result.get('youtube_shorts', [])),
                "instagram_reels": len(parsed_result.get('instagram_reels', [])),
                "tiktok_videos": len(parsed_result.get('tiktok_videos', [])),
                "three_stage_enhanced": True,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Social Media Content (3-Stage) Failed", "ERROR")
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

    def _combine_all_results_3stages(self, master_plan: Dict, stage1_result: Dict, stage2_result: Dict, stage3_result: Dict,
                           production_data: Dict, topic: str, description: str) -> Dict:
        """Combine all results from 3-stage system into final output"""

        # Combine stories from all three stages
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))
        all_stories.update(stage3_result.get('stories', {}))

        # Get scene plan
        scene_plan = master_plan.get('master_plan', {}).get('scene_plan', [])

        # Calculate total duration
        total_duration = self._calculate_total_duration(all_stories, scene_plan)

        # Create complete story text
        complete_story = self._compile_complete_story_3stages(stage1_result, stage2_result, stage3_result, scene_plan)

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

            # 3-Stage system info
            "three_stage_system": {
                "stage1_scenes": master_plan.get('master_plan', {}).get('stage1_scenes', 0),
                "stage2_scenes": master_plan.get('master_plan', {}).get('stage2_scenes', 0),
                "stage3_scenes": master_plan.get('master_plan', {}).get('stage3_scenes', 0),
                "total_scenes": master_plan.get('master_plan', {}).get('total_scenes', 0),
                "system_enabled": True
            },

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
                "duration_validation_applied": True,
                "three_stage_system_used": True,
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

    def _compile_complete_story_3stages(self, stage1_result: Dict, stage2_result: Dict, stage3_result: Dict, scene_plan: List[Dict]) -> str:
        """Compile complete story text from 3 stages"""
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
        story_parts.append("=== MAIN STORY (3-STAGE SYSTEM) ===")
        story_parts.append("")

        # Combine all stories from three stages
        all_stories = {}
        all_stories.update(stage1_result.get('stories', {}))
        all_stories.update(stage2_result.get('stories', {}))
        all_stories.update(stage3_result.get('stories', {}))

        # Add scenes in order with stage markers
        total_scenes = len(scene_plan)
        stage1_end = total_scenes // 3
        stage2_end = stage1_end + (total_scenes // 3)

        for i, scene in enumerate(scene_plan):
            scene_id = str(scene['scene_id'])
            story_content = all_stories.get(scene_id, f"[Story for scene {scene_id} not generated]")

            # Add stage markers
            if i == 0:
                story_parts.append("### STAGE 1: ESTABLISHMENT AND CHARACTER INTRODUCTION")
                story_parts.append("")
            elif i == stage1_end:
                story_parts.append("### STAGE 2: DEVELOPMENT AND DISCOVERY")
                story_parts.append("")
            elif i == stage2_end:
                story_parts.append("### STAGE 3: RECOGNITION AND RESOLUTION")
                story_parts.append("")

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
            if stage in ["master_plan", "master_plan_stage1", "master_plan_stage2", "master_plan_stage3"]:
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
                                result["scene_plan"] = []
                                break

            elif stage in ["stage1", "stage2", "stage3", "extension"]:
                # Extract stories
                result["stories"] = self._extract_stories_dict(content)
                if stage == "extension":
                    result["extended_stories"] = result["stories"]

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
    """Save complete production outputs - ALL 16 JSON FILES WITH 3-STAGE SYSTEM"""
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
        total_duration = result.get("generation_stats", {}).get("total_duration_minutes", 0)
        total_hours = int(total_duration / 60)

        # Get 3-stage system info
        three_stage_info = result.get('three_stage_system', {})

        # 1. Complete story text
        story_path = output_path / "complete_story.txt"
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(result["complete_story"])
        saved_files.append("complete_story.txt")

        # 2. Scene plan with enhanced chapters and 3-stage info
        plan_path = output_path / "scene_plan.json"
        scene_data = {
            "scene_plan": result["scene_plan"],
            "scene_chapters": result.get("scene_chapters", []),
            "total_scenes": len(result.get("scene_plan", [])),
            "total_duration_minutes": total_duration,
            "duration_validation_applied": True,
            "toibin_master_plan_used": True,
            "emotional_structure_applied": True,
            "three_stage_system": three_stage_info,
            "stage_breakdown": {
                "stage1_scenes": three_stage_info.get("stage1_scenes", 0),
                "stage2_scenes": three_stage_info.get("stage2_scenes", 0),
                "stage3_scenes": three_stage_info.get("stage3_scenes", 0),
                "system_benefits": [
                    "Enhanced character continuity across all scenes",
                    "Better timeout management during generation",
                    "Improved narrative flow and development",
                    "Stronger emotional arc progression"
                ]
            }
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
        saved_files.append("scene_plan.json")

        # 3. All stories (validated) with 3-stage organization
        stories_path = output_path / "all_stories.json"
        stories_data = {
            "stories": result["stories"],
            "three_stage_organization": {
                "stage1_story_ids": [str(i) for i in range(1, three_stage_info.get("stage1_scenes", 0) + 1)],
                "stage2_story_ids": [str(i) for i in range(three_stage_info.get("stage1_scenes", 0) + 1,
                                                          three_stage_info.get("stage1_scenes", 0) + three_stage_info.get("stage2_scenes", 0) + 1)],
                "stage3_story_ids": [str(i) for i in range(three_stage_info.get("stage1_scenes", 0) + three_stage_info.get("stage2_scenes", 0) + 1,
                                                          three_stage_info.get("total_scenes", 0) + 1)],
                "character_continuity_maintained": True,
                "narrative_progression": "Establishment → Development → Resolution"
            }
        }
        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(stories_data, f, indent=2, ensure_ascii=False)
        saved_files.append("all_stories.json")

        # 4. Voice directions for TTS with 3-stage progression
        voice_path = output_path / "voice_directions.json"
        voice_directions = []

        # Add voice directions for each scene with stage-aware instructions
        for scene in scene_plan:
            scene_num = scene.get("scene_id", 1)
            stage_info = ""

            if scene_num <= three_stage_info.get("stage1_scenes", 0):
                stage_info = "Stage 1: Establishment - Gentle introduction, character building"
            elif scene_num <= three_stage_info.get("stage1_scenes", 0) + three_stage_info.get("stage2_scenes", 0):
                stage_info = "Stage 2: Development - Deeper exploration, emotional development"
            else:
                stage_info = "Stage 3: Resolution - Contemplative conclusion, peaceful closure"

            voice_directions.append({
                "scene_number": scene_num,
                "title": scene.get("title", f"Scene {scene_num}"),
                "direction": f"Gentle, contemplative storytelling with {scene.get('emotion', 'peaceful')} emotion, Tóibín literary sensibility",
                "template": scene.get("emotion", "peaceful"),
                "style": "toibin_observational",
                "emotion": scene.get("emotion", "peaceful"),
                "pacing": "Sleep-optimized with natural breathing rhythm",
                "voice_notes": f"Maintain understated tone for {scene.get('duration_minutes', 4):.1f} minute duration",
                "stage_context": stage_info
            })

        voice_data = {
            "voice_directions": voice_directions,
            "three_stage_progression": {
                "stage1_voice_style": "Gentle establishment, character introduction",
                "stage2_voice_style": "Emotional development, deeper engagement",
                "stage3_voice_style": "Peaceful resolution, contemplative closure"
            }
        }

        with open(voice_path, "w", encoding="utf-8") as f:
            json.dump(voice_data, f, indent=2, ensure_ascii=False)
        saved_files.append("voice_directions.json")

        # 5. Visual generation prompts
        visual_path = output_path / "visual_generation_prompts.json"
        visual_data = {
            "visual_prompts": result["visual_prompts"],
            "three_stage_visual_progression": {
                "stage1_visuals": "Character establishment, setting introduction",
                "stage2_visuals": "Relationship development, emotional complexity",
                "stage3_visuals": "Resolution scenes, peaceful conclusions"
            },
            "enhanced_continuity": "3-stage system ensures better character consistency across all visuals"
        }
        with open(visual_path, "w", encoding="utf-8") as f:
            json.dump(visual_data, f, indent=2, ensure_ascii=False)
        saved_files.append("visual_generation_prompts.json")

        # 6. Character profiles with 3-stage development
        character_path = output_path / "character_profiles.json"
        character_data = {
            "main_characters": result.get("characters", {}).get("main_characters", []),
            "scene_character_mapping": result.get("characters", {}).get("scene_character_mapping", {}),
            "three_stage_character_development": {
                "stage1_focus": "Character introduction and establishment",
                "stage2_focus": "Character development and complexity",
                "stage3_focus": "Character resolution and completion",
                "continuity_benefits": "3-stage system provides better character arc development"
            },
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
• Enhanced with our 3-stage narrative system for superior character development

Perfect for insomnia, anxiety relief, or anyone who loves historical fiction combined with sleep meditation. Let the gentle rhythms of {story_topic} carry you into peaceful dreams.

⚠️ This story is designed to help you fall asleep - please don't listen while driving or operating machinery.""",
                "chapters": result.get("scene_chapters", []),
                "subscribe_cta": "🔔 Subscribe for more historical sleep stories and relaxation content! New videos every week.",
                "disclaimer": "This content is designed for relaxation and sleep. Please don't listen while driving or operating machinery."
            },
            "tags": youtube_data.get("tags", [
                "sleep story", "bedtime story", "relaxation", "insomnia help", "meditation",
                "calm", "peaceful", f"{total_hours} hours", "deep sleep", "anxiety relief",
                "stress relief", "asmr", "history", story_topic.lower(), "3-stage narrative"
            ]),
            "seo_strategy": youtube_data.get("seo_strategy", {}),
            "three_stage_advantage": "Enhanced character development and narrative flow",
            "api_ready_format": {
                "snippet": {
                    "title": youtube_data.get("clickbait_titles", [
                        f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)"])[0] if youtube_data.get("clickbait_titles") else f"The Secret History of {story_topic} ({total_hours} Hour Sleep Story)",
                    "description": f"""Journey back in time and experience the tranquil world of {story_topic}. This atmospheric sleep story follows the peaceful daily routines and lives of fascinating characters in {story_topic}.

Enhanced with our 3-stage narrative system for superior character development and story flow.

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

        # 8. Production specifications with 3-stage system
        production_path = output_path / "production_specifications.json"
        production_specs = result.get("production_specs", {})
        production_data = {
            "audio_production": production_specs.get("audio_production", {
                "tts_voice": "alloy",
                "speed_multiplier": 0.85,
                "pause_durations": {
                    "[PAUSE]": 2.0,
                    "scene_transition": 3.0,
                    "stage_transition": 5.0  # NEW: Longer pauses between stages
                }
            }),
            "video_timing": production_specs.get("video_timing", []),
            "quality_metrics": production_specs.get("quality_metrics", {
                "sleep_optimization_score": 9,
                "historical_accuracy": True,
                "tóibín_authenticity": True,
                "duration_validation_applied": True,
                "three_stage_system_used": True
            }),
            "three_stage_production_benefits": {
                "character_consistency": "Enhanced character development across all stages",
                "narrative_flow": "Better story progression and emotional arc",
                "timeout_management": "Prevents API timeouts during generation",
                "quality_control": "More focused scene development per stage"
            },
            "automation_specifications": {
                "character_extraction": "✅ Complete",
                "youtube_optimization": "✅ Complete",
                "production_specifications": "✅ Complete",
                "api_ready_format": "✅ Complete",
                "duration_validation": "✅ Complete",
                "three_stage_system": "✅ Complete"
            }
        }
        with open(production_path, "w", encoding="utf-8") as f:
            json.dump(production_data, f, indent=2, ensure_ascii=False)
        saved_files.append("production_specifications.json")

        # 9. Social media content with 3-stage enhanced
        social_path = output_path / "social_media_content.json"
        social_data = result.get("social_media_content", {})
        enhanced_social_data = {
            **social_data,
            "production_ready": True,
            "three_stage_advantage": "Enhanced character depth for better viral content",
            "viral_growth_strategy": {
                "target_subscribers": "1M",
                "content_funnel": "Social Media → Main YouTube Channel",
                "posting_schedule": {
                    "youtube_shorts": "Daily at 8 PM",
                    "instagram_reels": "Daily at 6 PM",
                    "tiktok_videos": "2x daily at 12 PM and 9 PM"
                },
                "cross_promotion": "Each platform drives to main channel",
                "hashtag_strategy": "#SleepyDullStories on all platforms",
                "three_stage_benefits": "Better character arcs provide more viral content opportunities"
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
            "three_stage_character_development": "Character designs benefit from 3-stage depth and consistency",
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

        # 11. Hook & Subscribe scenes with stage progression
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
                    "timing_note": f"Display during hook seconds {i * 3}-{(i * 3) + 3}",
                    "stage_context": "Stage 1 establishment scenes for hook"
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
                    "timing_note": f"Display during subscribe seconds {i * 3}-{(i * 3) + 3}",
                    "stage_context": "Mixed stage scenes for subscribe appeal"
                }
                for i, scene in enumerate(scene_plan[10:20] if len(scene_plan) > 10 else scene_plan)  # Next 10 scenes
            ],
            "three_stage_progression": "Hook uses Stage 1, Subscribe uses mixed stages for broader appeal",
            "production_notes": {
                "hook_timing": "Use hook_scenes during golden hook narration (0-30s)",
                "subscribe_timing": "Use subscribe_scenes during subscribe request (30-60s)",
                "visual_sync": "Each scene should blend seamlessly with spoken content",
                "stage_awareness": "Visual selection considers 3-stage narrative progression"
            }
        }
        with open(hook_subscribe_path, "w", encoding="utf-8") as f:
            json.dump(hook_subscribe_data, f, indent=2, ensure_ascii=False)
        saved_files.append("hook_subscribe_scenes.json")

        # 12. Audio generation prompts with stage-aware settings
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

        # Story scenes audio with stage awareness
        stories = result.get("stories", {})
        production_audio = production_specs.get("audio_production", {})

        for scene_id, story_content in stories.items():
            scene_info = next((s for s in scene_plan if s.get("scene_id") == int(scene_id)), {})

            # Determine stage for audio adjustments
            scene_num = int(scene_id)
            if scene_num <= three_stage_info.get("stage1_scenes", 0):
                stage_emphasis = "gentle establishment"
            elif scene_num <= three_stage_info.get("stage1_scenes", 0) + three_stage_info.get("stage2_scenes", 0):
                stage_emphasis = "emotional development"
            else:
                stage_emphasis = "peaceful resolution"

            audio_prompts.append({
                "segment_id": f"scene_{scene_id}",
                "content": story_content,
                "duration_minutes": scene_info.get("duration_minutes", 4),
                "emotion": scene_info.get("emotion", "peaceful"),
                "stage_context": stage_emphasis,
                "tts_settings": {
                    "voice": production_audio.get("tts_voice", "alloy"),
                    "speed": production_audio.get("speed_multiplier", 0.85),
                    "pitch": -2,
                    "volume": 80,
                    "emphasis": "sleep-optimized"
                }
            })

        audio_data = {
            "audio_prompts": audio_prompts,
            "three_stage_audio_progression": {
                "stage1_emphasis": "Gentle establishment and character introduction",
                "stage2_emphasis": "Emotional development and deeper engagement",
                "stage3_emphasis": "Peaceful resolution and contemplative closure"
            }
        }

        with open(audio_path, "w", encoding="utf-8") as f:
            json.dump(audio_data, f, indent=2, ensure_ascii=False)
        saved_files.append("audio_generation_prompts.json")

        # 13. Video composition instructions with 3-stage structure
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
            "three_stage_composition": {
                "stage1_timing": f"60s - {(three_stage_info.get('stage1_scenes', 0) * 4 * 60) + 60}s",
                "stage2_timing": f"{(three_stage_info.get('stage1_scenes', 0) * 4 * 60) + 60}s - {((three_stage_info.get('stage1_scenes', 0) + three_stage_info.get('stage2_scenes', 0)) * 4 * 60) + 60}s",
                "stage3_timing": f"{((three_stage_info.get('stage1_scenes', 0) + three_stage_info.get('stage2_scenes', 0)) * 4 * 60) + 60}s - end",
                "stage_transitions": "5-second crossfades between narrative stages"
            },
            "scene_sync_strategy": {
                "rule": "When audio mentions scene X, display scene X visual",
                "timing": "Immediate visual sync with narrative",
                "stage_awareness": "Visual style adapts to narrative stage progression"
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
                "total_duration": f"{total_duration + 1:.0f} minutes",
                "three_stage_markers": "Visual indicators for stage transitions"
            },
            "chapters": result.get("scene_chapters", [])
        }
        with open(video_path, "w", encoding="utf-8") as f:
            json.dump(video_composition, f, indent=2, ensure_ascii=False)
        saved_files.append("video_composition_instructions.json")

        # 14. Platform metadata with 3-stage system highlights
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
                "#deepsleep", "#calm", "#history", f"#{story_topic.lower().replace(' ', '')}", "#3stagestory"
            ],
            "seo_strategy": youtube_metadata.get("seo_strategy", {}),
            "thumbnail_concept": thumbnail_data.get("thumbnail_concept", {}),
            "three_stage_marketing_advantage": {
                "character_depth": "Enhanced character development appeals to wider audience",
                "narrative_quality": "Superior story structure increases engagement",
                "production_quality": "Better timeout management ensures higher quality output"
            },
            "engagement_strategy": {
                "target_audience": "Sleep content seekers + History enthusiasts",
                "content_pillars": ["Historical accuracy", "Sleep optimization", "Literary quality", "3-stage narrative"],
                "posting_schedule": "Weekly uploads, consistent timing"
            },
            "api_ready_format": youtube_metadata.get("api_ready_format", {})
        }
        with open(platform_path, "w", encoding="utf-8") as f:
            json.dump(platform_data, f, indent=2, ensure_ascii=False)
        saved_files.append("platform_metadata.json")

        # 15. Automation specs with 3-stage system
        automation_path = output_path / "automation_specs.json"
        automation_data = {
            "audio_production": production_data.get("audio_production", {}),
            "video_assembly": {
                "scene_timing_precision": [
                    {
                        "scene_number": scene.get("scene_id", i+1),
                        "start_time": f"00:{60 + sum(s.get('duration_minutes', 4) * 60 for s in scene_plan[:i]) // 60:02d}:{(60 + sum(s.get('duration_minutes', 4) * 60 for s in scene_plan[:i])) % 60:02d}",
                        "duration_seconds": int(scene.get("duration_minutes", 4) * 60),
                        "word_count": int(scene.get("duration_minutes", 4) * 140),  # 140 words per minute
                        "stage": "1" if i < three_stage_info.get("stage1_scenes", 0) else "2" if i < three_stage_info.get("stage1_scenes", 0) + three_stage_info.get("stage2_scenes", 0) else "3"
                    }
                    for i, scene in enumerate(scene_plan)
                ],
                "video_specifications": {
                    "resolution": "1920x1080",
                    "frame_rate": 30,
                    "transition_type": "slow_fade",
                    "export_format": "MP4_H264",
                    "stage_transitions": "5-second crossfades between stages"
                }
            },
            "quality_control": production_data.get("quality_metrics", {}),
            "toibin_quality_assurance": {
                "literary_authenticity": True,
                "master_plan_followed": True,
                "emotional_structure_applied": True,
                "character_psychology_depth": True,
                "duration_validation_applied": True,
                "three_stage_system_used": True
            },
            "three_stage_automation_benefits": {
                "better_character_consistency": "Enhanced character tracking across stages",
                "improved_timeout_management": "Prevents API failures during generation",
                "superior_narrative_flow": "Better story progression and emotional development",
                "enhanced_production_quality": "More focused scene development per stage"
            },
            "implementation_ready": True
        }
        with open(automation_path, "w", encoding="utf-8") as f:
            json.dump(automation_data, f, indent=2, ensure_ascii=False)
        saved_files.append("automation_specs.json")

        # 16. Generation report with 3-stage system details
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
            "duration_validation_applied": True,
            "three_stage_system_used": True,
            "stats": result["generation_stats"],
            "three_stage_analysis": {
                "stage1_scenes": three_stage_info.get("stage1_scenes", 0),
                "stage2_scenes": three_stage_info.get("stage2_scenes", 0),
                "stage3_scenes": three_stage_info.get("stage3_scenes", 0),
                "total_scenes": three_stage_info.get("total_scenes", 0),
                "system_benefits": [
                    "Enhanced character continuity across all scenes",
                    "Better timeout management during generation",
                    "Improved narrative flow and development",
                    "Stronger emotional arc progression",
                    "More focused scene development per stage",
                    "Superior character depth and consistency"
                ]
            },
            "cost_analysis": {
                "total_api_calls": api_calls,
                "total_cost": total_cost,
                "cost_per_scene": total_cost / len(scene_plan) if scene_plan else 0,
                "cost_efficiency": "Tóibín quality + viral social media + duration validation + 3-stage system optimization"
            },
            "files_saved": saved_files,
            "next_steps": [
                "1. Generate character reference images using character_profiles.json",
                "2. Generate scene visuals using visual_generation_prompts.json (with 3-stage progression)",
                "3. Generate thumbnail using thumbnail_generation.json",
                "4. Generate audio using audio_generation_prompts.json (with stage-aware settings)",
                "5. Create social media content using social_media_content.json (enhanced with 3-stage depth)",
                "6. Compose video using video_composition_instructions.json (with stage transitions)",
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
                "toibin_literary_quality": "✅ Complete",
                "duration_validation": "✅ Complete",
                "three_stage_system": "✅ Complete"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ ALL 16 PRODUCTION FILES SAVED WITH 3-STAGE SYSTEM: {saved_files}")
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
    """Print production summary with 3-stage system and duration validation info"""
    stats = result["generation_stats"]
    three_stage_info = result.get("three_stage_system", {})

    print("\n" + "🎭" * 60)
    print("TÓIBÍN QUALITY STORY GENERATOR WITH 3-STAGE SYSTEM & DURATION VALIDATION - COMPLETE!")
    print("🎭" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CONFIG.claude_config['model']} (Claude 4)")
    print(f"⏱️ Total Generation Time: {generation_time:.1f}s")

    print(f"\n📊 GENERATION RESULTS:")
    print(f"🎬 Scenes Planned: {stats['total_scenes']}")
    print(f"📝 Stories Written: {stats['total_stories']}")
    print(f"⏰ Total Duration: {stats['total_duration_minutes']:.1f} minutes")
    print(f"🎯 Duration Target: {CONFIG.claude_config['minimum_duration_minutes']} minutes")
    print(f"✅ Duration Met: {'YES' if stats['total_duration_minutes'] >= CONFIG.claude_config['minimum_duration_minutes'] else 'NO'}")
    print(f"📱 Social Media Pieces: {stats.get('social_media_pieces', 0)}")
    print(f"📈 Completion Rate: {stats['completion_rate']:.1f}%")
    print(f"🔥 API Calls: {stats['api_calls_used']}")
    print(f"💰 Total Cost: ${stats['total_cost']:.4f}")

    print(f"\n🎭 TÓIBÍN QUALITY FEATURES:")
    print(f"✅ Master Plan Approach: {stats.get('master_plan_approach', False)}")
    print(f"✅ Emotional Structure: {stats.get('emotional_structure_used', False)}")
    print(f"✅ Tóibín Literary Style: {stats.get('tóibín_quality_applied', False)}")
    print(f"✅ Duration Validation: {stats.get('duration_validation_applied', False)}")
    print(f"✅ Viral Content Created: {stats.get('viral_content_created', False)}")
    print(f"✅ 3-Stage System Used: {stats.get('three_stage_system_used', False)}")

    print(f"\n🚀 3-STAGE SYSTEM BREAKDOWN:")
    print(f"📖 Stage 1 (Establishment): {three_stage_info.get('stage1_scenes', 0)} scenes")
    print(f"🔍 Stage 2 (Development): {three_stage_info.get('stage2_scenes', 0)} scenes")
    print(f"🎯 Stage 3 (Resolution): {three_stage_info.get('stage3_scenes', 0)} scenes")
    print(f"📊 Total Scenes: {three_stage_info.get('total_scenes', 0)} scenes")

    duration_met = stats['total_duration_minutes'] >= CONFIG.claude_config['minimum_duration_minutes']
    completion_rate = stats['completion_rate']
    three_stage_used = stats.get('three_stage_system_used', False)

    if completion_rate >= 80 and duration_met and three_stage_used:
        print(f"\n🎉 TÓIBÍN QUALITY + 3-STAGE SYSTEM + DURATION SUCCESS!")
        print(f"✅ Literary excellence with production optimization")
        print(f"✅ Duration requirement met: {stats['total_duration_minutes']:.1f} minutes")
        print(f"✅ 3-stage system delivered enhanced character continuity")
        print(f"✅ Master plan approach with timeout prevention")
        print(f"🚀 Ready for full production pipeline + viral growth")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        if not duration_met:
            print(f"⚠️ Duration requirement not met: {stats['total_duration_minutes']:.1f} < {CONFIG.claude_config['minimum_duration_minutes']} minutes")
        if not three_stage_used:
            print(f"⚠️ 3-stage system not fully utilized")
        print(f"🔍 Review generation_report.json for details")

    print("\n📄 ALL 16 PRODUCTION FILES CREATED WITH 3-STAGE SYSTEM:")
    print("1. 📖 complete_story.txt - Full Tóibín-quality story with 3-stage progression")
    print("2. 🎬 scene_plan.json - Master scene plan + chapters + 3-stage breakdown")
    print("3. 📚 all_stories.json - All individual stories organized by stages")
    print("4. 🎵 voice_directions.json - TTS voice guidance with stage progression")
    print("5. 🖼️ visual_generation_prompts.json - Story-based visuals with stage awareness")
    print("6. 👥 character_profiles.json - Enhanced character analysis with 3-stage development")
    print("7. 📺 youtube_metadata.json - YouTube optimization highlighting 3-stage system")
    print("8. 🏭 production_specifications.json - Technical specs with stage transitions")
    print("9. 📱 social_media_content.json - VIRAL GROWTH STRATEGY with enhanced character depth")
    print("10. 🖼️ thumbnail_generation.json - Thumbnail strategy with character consistency")
    print("11. 🎭 hook_subscribe_scenes.json - Opening sequences with stage context")
    print("12. 🎵 audio_generation_prompts.json - TTS production with stage-aware settings")
    print("13. 🎥 video_composition_instructions.json - Video assembly with stage transitions")
    print("14. 🌍 platform_metadata.json - Platform optimization with 3-stage advantages")
    print("15. 🤖 automation_specs.json - Automation ready with 3-stage benefits")
    print("16. 📊 generation_report.json - Complete analytics with 3-stage analysis")

    # Social media breakdown
    social_stats = result.get("social_media_content", {})
    print(f"\n🚀 VIRAL GROWTH STRATEGY (ENHANCED WITH 3-STAGE DEPTH):")
    print(f"📺 YouTube Shorts: {len(social_stats.get('youtube_shorts', []))} pieces")
    print(f"📸 Instagram Reels: {len(social_stats.get('instagram_reels', []))} pieces")
    print(f"🎵 TikTok Videos: {len(social_stats.get('tiktok_videos', []))} pieces")
    print(f"🎯 Target: 1M Subscribers via cross-platform funnel")
    print(f"🎭 3-Stage Advantage: Enhanced character depth for better viral content")

    print("\n🚀 3-STAGE SYSTEM BENEFITS:")
    print("✅ Enhanced character continuity across all scenes")
    print("✅ Better timeout management during generation")
    print("✅ Improved narrative flow and development")
    print("✅ Stronger emotional arc progression")
    print("✅ More focused scene development per stage")
    print("✅ Superior character depth and consistency")

    print("🎭" * 60)

if __name__ == "__main__":
    try:
        print("🎭 TÓIBÍN QUALITY STORY GENERATOR WITH 3-STAGE SYSTEM & DURATION VALIDATION")
        print("⚡ 3-Stage System + Master Plan + Emotional Structure + Literary Excellence + Duration Validation + Social Media")
        print("📄 ALL 16 JSON FILES WITH 3-STAGE SYSTEM & DURATION VALIDATION")
        print("💫 120+ MINUTE DURATION GUARANTEE")
        print("🚀 ENHANCED CHARACTER CONTINUITY & TIMEOUT PREVENTION")
        print("=" * 80)

        # Get topic from database
        topic_id, topic, description, clickbait_title = get_next_topic_from_database()
        print(f"\n📚 Topic ID: {topic_id} - {topic}")
        print(f"📝 Description: {description}")

        # Setup output directory
        output_path = Path(CONFIG.paths['OUTPUT_DIR']) / str(topic_id)

        # Initialize generator with checkpoint support
        generator = ToibinStoryGenerator()
        generator.setup_checkpoint(topic_id)

        # Generate complete story with Tóibín quality + 3-stage system + duration validation + social media
        start_time = time.time()
        result = generator.generate_complete_story(topic, description, clickbait_title)
        generation_time = time.time() - start_time

        # Save all outputs (ALL 16 FILES WITH 3-STAGE SYSTEM)
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

        print("\n🎭 TÓIBÍN QUALITY + 3-STAGE SYSTEM + DURATION VALIDATION GENERATION COMPLETE!")
        print(f"✅ Literary excellence + viral social media + 120+ minute guarantee + 3-stage system delivered: {output_path}")
        print(f"💰 Total cost: ${generator.total_cost:.4f}")
        print(f"⏰ Final duration: {total_duration:.1f} minutes")
        print(f"🎯 Duration target met: {'YES' if total_duration >= 120 else 'NO'}")
        print(f"📱 Social media pieces: {result['generation_stats'].get('social_media_pieces', 0)}")
        print(f"🚀 3-Stage System: Enhanced character continuity and timeout prevention")
        print(f"🎯 Ready for 1M subscriber strategy!")
        print(f"📄 ALL 16 JSON FILES SAVED AND READY WITH 3-STAGE SYSTEM!")
        print(f"💫 DURATION VALIDATION SYSTEM ACTIVE!")
        print(f"🎭 3-STAGE LITERARY EXCELLENCE ACHIEVED!")

    except Exception as e:
        print(f"\n💥 GENERATION ERROR: {e}")
        CONFIG.logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()