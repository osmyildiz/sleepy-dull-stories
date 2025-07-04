"""
Sleepy Dull Stories - Complete Server-Ready Claude Story Generator
Production-optimized with Hook & Subscribe + Complete JSON outputs + Thumbnail positioning
Compatible with: Server environment, production paths, complete automation pipeline
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

        # Find topics.csv
        self.paths['TOPIC_CSV_PATH'] = self.find_topics_csv()

        print(f"✅ Server paths configured:")
        print(f"   📁 Project root: {self.paths['BASE_DIR']}")
        print(f"   📄 Topics CSV: {self.paths['TOPIC_CSV_PATH']}")

    def find_topics_csv(self):
        """Find topics.csv in multiple locations or create fallback"""
        possible_paths = [
            self.project_root / 'topics.csv',
            self.project_root / 'data' / 'topics.csv',
            Path('topics.csv'),
            Path('../data/topics.csv'),
            Path('../../topics.csv'),
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
        """Setup Claude configuration with server optimizations"""
        self.claude_config = {
            "model": "claude-3-5-sonnet-20241022",  # Updated model
            "max_tokens": 8192,  # Realistic server limit
            "temperature": 0.7,
            "target_scenes": 40,
            "target_duration_minutes": 120,
            "target_words_per_minute": 140,
            "two_stage_approach": True,
            "character_extraction": True,
            "thumbnail_generation": True,
            "max_characters": 5,
            "test_mode": False,
            "server_mode": True
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
    """Complete server-ready automated story generation with Hook & Subscribe + full JSON outputs"""

    def __init__(self):
        """Initialize story generator for server environment"""
        self.generation_log = []
        self.api_call_count = 0
        self.total_cost = 0.0
        self.character_system = CharacterExtractionSystem()

        try:
            self.client = Anthropic(api_key=CONFIG.api_key)
            CONFIG.logger.info("✅ Story generator initialized successfully")
            print("✅ Story generator initialized with complete pipeline")
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
                "critical": "20th story mention = 20th scene display",
                "timing": "Instant visual sync with narrative"
            },
            "production_notes": {
                "hook_timing": "Use hook_scenes during golden hook narration (0-30s)",
                "subscribe_timing": "Use subscribe_scenes during subscribe request (30-60s)",
                "visual_sync": "Each scene should blend seamlessly with spoken content",
                "fallback_strategy": "If scene unavailable, use next scene in sequence"
            }
        }

    def generate_complete_story_with_characters(self, topic: str, description: str, clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """
        COMPLETE 5-STAGE APPROACH:
        Stage 1: Planning + Hook + Subscribe + 20 stories + Prompts
        Stage 2: Remaining 20 stories
        Stage 3: Character extraction and analysis
        Stage 4: Intelligent thumbnail generation
        Stage 5: Hook & Subscribe scene selection + Complete JSON outputs
        """

        self.log_step("Complete Story Generation with Full Pipeline")

        try:
            # STAGE 1: Planning + First Half
            stage1_result = self._generate_stage1(topic, description)
            time.sleep(1)  # Server-friendly pause

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

            self.log_step("Complete Generation Pipeline Finished", "SUCCESS", {
                "total_scenes": len(combined_result.get('scene_plan', [])),
                "total_stories": len(combined_result.get('stories', {})),
                "characters_extracted": len(combined_result.get('main_characters', [])),
                "thumbnail_generated": combined_result.get('generation_stats', {}).get('thumbnail_generated', False),
                "hook_subscribe_generated": combined_result.get('generation_stats', {}).get('hook_subscribe_generated', False),
                "api_calls_total": self.api_call_count
            })

            return combined_result

        except Exception as e:
            self.log_step("Generation Failed", "ERROR")
            CONFIG.logger.error(f"Generation failed: {e}")
            raise

    def _generate_stage1(self, topic: str, description: str) -> Dict[str, Any]:
        """STAGE 1: Server-optimized planning + first 20 stories"""

        self.log_step("Stage 1: Planning + First 20 Stories")

        stage1_prompt = f"""Create the complete foundation for a 2-hour sleep story about "{topic}".

TOPIC: {topic}
DESCRIPTION: {description}

STAGE 1 REQUIREMENTS:
You must provide ALL planning elements + first 20 stories in complete detail.

## 1. GOLDEN HOOK (30 seconds, ~90 words)
- Atmospheric opening that sets the scene
- Gentle intrigue but calming
- Cinematic visual details

## 2. SUBSCRIBE SECTION (30 seconds, ~70 words) 
- Natural community invitation
- Warm, friendly tone (not corporate)

## 3. COMPLETE SCENE PLAN (Exactly 40 scenes)
Each scene must have:
- Unique location and activity
- Template rotation: atmospheric, character_focused, historical_detail, sensory_journey
- Style rotation: observational, immersive, documentary, poetic, cinematic
- Emotion progression: 1-10 peaceful, 11-20 curiosity, 21-30 concern, 31-40 resolution
- Key characters mentioned in descriptions

## 4. FIRST 20 COMPLETE STORIES (Scenes 1-20)
Each story must be 450-600 words with:
- Present tense, second person perspective - but NEVER start with "You find yourself"
- Rich sensory details (sight, sound, smell, touch)
- [PAUSE] markers for TTS
- Sleep-optimized language
- Historical accuracy
- Clear character interactions and mentions
- Create unique, atmospheric openings for each scene

## 5. BASIC VISUAL PROMPTS (All 40 scenes)
- Simple AI image generation prompts
- Focus on location and atmosphere
- Character presence noted but details added later

## 6. VOICE DIRECTIONS (All 40 scenes)
- TTS guidance for each scene
- Pace, mood, emphasis

## 7. INTRO SEQUENCE VISUAL RECOMMENDATIONS
Based on your 40-scene plan, recommend the BEST scene visuals for the critical opening:

A) GOLDEN HOOK (0-30 seconds):
- Most visually striking and atmospheric scene from your 40 scenes
- Should immediately capture attention and create wonder
- Best emotional impact for first impression
- Consider: lighting, architecture, mystery, beauty

B) SUBSCRIBE SECTION (30-60 seconds):
- Community-friendly, welcoming scene from your 40 scenes
- Builds trust and human connection
- Warm, inviting atmosphere that makes viewers want to join
- Consider: warmth, gathering spaces, positive human elements

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
      "duration_minutes": 4,
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
    "1": "[COMPLETE 450-600 word story for scene 1]",
    "2": "[COMPLETE story for scene 2]"
  }},
  "visual_prompts": [
    {{
      "scene_number": 1,
      "title": "[Scene title]",
      "prompt": "[Basic AI image prompt]",
      "duration_minutes": 4,
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
    "scenes_planned": 40,
    "stories_written": 20,
    "total_word_count": "[calculated]",
    "characters_introduced": "[count]",
    "ready_for_stage2": true
  }},
  "intro_sequence_visuals": {{
    "hook_visual": {{
      "recommended_scene": 15,
      "scene_title": "[Most visually striking scene title]",
      "reasoning": "[Why this scene is perfect for hook - visual impact, atmosphere, emotional draw]",
      "visual_elements": ["golden lighting", "dramatic architecture", "atmospheric mood"],
      "emotional_impact": "[Wonder/mystery/beauty - what viewer feels immediately]",
      "hook_effectiveness": "[High/Medium] - [why it works for first 30 seconds]",
      "timing_sync": "[How visual aligns with hook narration]"
    }},
    "subscribe_visual": {{
      "recommended_scene": 8,
      "scene_title": "[Most welcoming/community scene title]",
      "reasoning": "[Why this scene builds trust and community connection]",
      "visual_elements": ["warm lighting", "gathering space", "welcoming atmosphere"],
      "emotional_impact": "[Warmth/belonging/trust - community building emotion]",
      "community_appeal": "[High/Medium] - [why it encourages subscription]",
      "timing_sync": "[How visual aligns with subscribe request]"
    }},
    "alternative_options": {{
      "hook_alternatives": [
        {{"scene": 23, "title": "[Alternative scene title]", "why": "[Brief reason]"}},
        {{"scene": 3, "title": "[Alternative scene title]", "why": "[Brief reason]"}}
      ],
      "subscribe_alternatives": [
        {{"scene": 12, "title": "[Alternative scene title]", "why": "[Brief reason]"}},
        {{"scene": 28, "title": "[Alternative scene title]", "why": "[Brief reason]"}}
      ]
    }},
    "usage_instructions": {{
      "hook_timing": "Display recommended hook visual during entire golden hook narration (0-30 seconds)",
      "subscribe_timing": "Display recommended subscribe visual during subscribe request (30-60 seconds)",
      "sync_importance": "Visual must emotionally align with spoken content for maximum impact",
      "fallback_strategy": "Use alternative options if primary recommendation unavailable"
    }}
  }}
}}

Generate complete Stage 1 content with creative, unique openings for each scene."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                timeout=900,  # Server timeout
                system="You are a MASTER STORYTELLER and automated content creator. Stage 1: Create complete planning + first 20 atmospheric stories with rich character interactions. Focus on memorable, distinct characters.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            # Get response content
            content = response.content[0].text if hasattr(response, 'content') else str(response)

            # Calculate cost (estimate based on content length)
            input_tokens = len(stage1_prompt) // 4  # Rough estimate
            output_tokens = len(content) // 4  # Rough estimate
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)  # Claude costs
            self.total_cost += stage_cost

            print(f"✅ Stage 1 complete: {len(content):,} characters - Cost: ${stage_cost:.4f}")
            CONFIG.logger.info(f"Stage 1 response length: {len(content)} characters - Cost: ${stage_cost:.4f}")

            # Parse Stage 1 result
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
        """STAGE 2: Remaining 20 stories (21-40)"""

        self.log_step("Stage 2: Remaining 20 Stories")

        # Get scene plan from stage 1
        scene_plan = stage1_result.get('scene_plan', [])
        scenes_21_to_40 = [scene for scene in scene_plan if scene['scene_id'] >= 21]

        if len(scenes_21_to_40) == 0:
            print("⚠️ No scenes 21-40 found in stage 1, creating fallback")
            scenes_21_to_40 = self._create_fallback_scenes_21_40(topic)

        # Create stage 2 prompt
        scenes_text = "\n".join([
            f"Scene {scene['scene_id']}: {scene['title']}\n"
            f"Location: {scene['location']}\n"
            f"Template: {scene['template']} | Style: {scene['narrative_style']}\n"
            f"Emotion: {scene['emotion']} | Focus: {scene['sensory_focus']}\n"
            f"Description: {scene['description']}\n"
            f"Characters: {', '.join(scene.get('characters_mentioned', []))}\n"
            for scene in scenes_21_to_40
        ])

        stage2_prompt = f"""Complete the sleep story for "{topic}" by writing the remaining 20 stories (scenes 21-40).

TOPIC: {topic}
DESCRIPTION: {description}

SCENES TO COMPLETE:
{scenes_text}

REQUIREMENTS:
- Write COMPLETE stories for scenes 21-40 (450-600 words each)
- Present tense, second person perspective - NEVER use "You find yourself"
- Follow emotion progression: 21-30 mild concern, 31-40 deep peace
- Each story must be atmospheric and historically accurate
- Rich sensory details throughout
- Continue character development from Stage 1
- Maintain character consistency and interactions

OUTPUT FORMAT:
{{
  "stories": {{
    "21": "[COMPLETE 450-600 word story for scene 21]",
    "22": "[COMPLETE story for scene 22]"
  }},
  "stage2_stats": {{
    "stories_written": 20,
    "scenes_covered": "21-40",
    "total_word_count": "[calculated]",
    "character_development": "continued"
  }}
}}

Write all 20 remaining stories completely."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=CONFIG.claude_config["temperature"],
                timeout=900,
                system="You are a MASTER STORYTELLER. Stage 2: Complete the remaining 20 stories with rich character development and consistent character interactions from Stage 1.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = response.content[0].text if hasattr(response, 'content') else str(response)

            # Calculate cost
            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            print(f"✅ Stage 2 complete: {len(content):,} characters - Cost: ${stage_cost:.4f}")

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
        """STAGE 3: Extract main characters + YouTube optimization"""

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

STORY CONTENT (First 20000 chars):
{story_content[:20000]}

SCENE PLAN CONTEXT (First 3000 chars):
{scene_context[:3000]}

REQUIREMENTS:

## PART 1: CHARACTER EXTRACTION
- Identify maximum {CONFIG.claude_config['max_characters']} main characters
- Focus on characters that appear in multiple scenes
- Provide comprehensive character analysis

## PART 2: YOUTUBE OPTIMIZATION
Create complete YouTube upload package

OUTPUT FORMAT (Complete JSON):
{{
  "main_characters": [
    {{
      "name": "[Character name]",
      "role": "protagonist|supporting|background|minor",
      "importance_score": 0,
      "scene_appearances": [1, 3, 7, 12],
      "personality_traits": ["trait1", "trait2", "trait3"],
      "physical_description": "[Detailed visual description]",
      "visual_notes": "[Special notes for image generation]",
      "voice_style": "[How character speaks]",
      "core_function": "[Character's purpose in story]",
      "character_arc": {{
        "beginning": "[Initial state]",
        "conflict": "[Main challenge]",
        "ending": "[Resolution]"
      }},
      "symbolism": "[What character represents]",
      "visual_contrast": "[Lighting preferences]",
      "emotional_journey": "[Emotion evolution]",
      "use_in_marketing": true/false,
      "thumbnail_potential": "[Why good for thumbnails]",
      "relationships": []
    }}
  ],
  "character_relationships": [],
  "scene_character_mapping": {{}},
  "visual_style_notes": {{
    "art_style": "[Preferred style]",
    "color_palette": "[Dominant colors]",
    "mood": "[Overall mood]",
    "period_accuracy": "[Historical details]"
  }},
  "youtube_optimization": {{
    "clickbait_titles": [
      "Title 1",
      "Title 2"
    ],
    "tags": ["sleep story", "relaxation"],
    "seo_strategy": {{
      "primary_keywords": ["sleep story"],
      "long_tail_keywords": ["2 hour sleep story"]
    }},
    "youtube_metadata": {{
      "category": "Education",
      "default_language": "en",
      "privacy_status": "public"
    }}
  }},
  "character_stats": {{
    "total_characters_found": 0,
    "main_characters_extracted": 0
  }}
}}

Analyze thoroughly and create complete package."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=CONFIG.claude_config["max_tokens"],
                temperature=0.3,
                timeout=600,
                system="You are an expert character analyst and production optimization specialist. Extract main characters with comprehensive analysis and create complete production package.",
                messages=[{"role": "user", "content": character_prompt}]
            )

            content = response.content[0].text if hasattr(response, 'content') else str(response)

            # Calculate cost
            input_tokens = len(character_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            print(f"✅ Character analysis complete: {len(content):,} characters - Cost: ${stage_cost:.4f}")

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

            self.character_system.log_extraction_step("Character Extraction", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('main_characters', [])),
                "character_names": [c.get('name', 'Unknown') for c in parsed_result.get('main_characters', [])],
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Character Extraction Failed", "ERROR")
            CONFIG.logger.error(f"Character extraction error: {e}")
            return {"main_characters": [], "character_stats": {"error": str(e)}}

    def _generate_intelligent_thumbnail(self, topic: str, description: str, character_result: Dict, clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """STAGE 4: Generate intelligent thumbnail with right-side character positioning"""

        self.character_system.log_extraction_step("Intelligent Thumbnail Generation")

        characters = character_result.get('main_characters', [])
        visual_style = character_result.get('visual_style_notes', {})

        # Select optimal character for thumbnail
        thumbnail_character_selection = self.character_system.select_thumbnail_character(
            characters, topic, description
        )

        # Use provided clickbait title or generate fallback
        if not clickbait_title:
            youtube_data = character_result.get('youtube_optimization', {})
            clickbait_titles = youtube_data.get('clickbait_titles', [])
            clickbait_title = clickbait_titles[0] if clickbait_titles else f"The Secret History of {topic} (2 Hour Sleep Story)"

        # Use provided font design or generate fallback
        if not font_design:
            font_design = "Bold impact font, warm golden colors, readable shadows"

        thumbnail_prompt = f"""Create an intelligent thumbnail design for the sleep story "{topic}".

STORY TOPIC: {topic}
STORY DESCRIPTION: {description}

CHARACTER SELECTION: {thumbnail_character_selection['character_used']}
REASONING: {thumbnail_character_selection['reasoning']}

CLICKBAIT TITLE: {clickbait_title}
FONT DESIGN: {font_design}

CRITICAL POSITIONING REQUIREMENTS:
- Character must be positioned on the RIGHT side of the frame (right third)
- Character should be as close to the right edge as possible while still being fully visible
- LEFT and CENTER areas must be reserved for text overlays
- Background on left and center should be landscape, architecture, or atmospheric elements that won't compete with text
- Character faces slightly toward camera but positioned far right for optimal text placement

Create a thumbnail that balances SLEEP CONTENT (peaceful, calming) with CLICKABILITY (attention-grabbing but not jarring).

OUTPUT FORMAT:
{{
  "thumbnail_prompt": {{
    "scene_number": 99,
    "character_used": "{thumbnail_character_selection['character_used']}",
    "clickbait_title": "{clickbait_title}",
    "font_design": "{font_design}",
    "prompt": "[Detailed visual prompt with character positioned on RIGHT side]",
    "visual_style": "[Style notes]",
    "character_positioning": "[Detailed positioning - RIGHT side, close to edge]",
    "text_overlay_strategy": "[How text will be placed in left and center areas]",
    "emotional_appeal": "[Target emotion]",
    "thumbnail_reasoning": "{thumbnail_character_selection['reasoning']}",
    "background_scene": "[Description of left/center background for text placement]",
    "lighting_strategy": "[How lighting supports both character and text readability]",
    "composition_notes": "[Rule of thirds with character on right, text space on left/center]"
  }},
  "thumbnail_alternatives": [
    {{
      "variant": "Character Focus",
      "prompt": "[Alternative character-focused thumbnail with right positioning]"
    }},
    {{
      "variant": "Atmospheric Focus", 
      "prompt": "[Alternative atmospheric thumbnail with right-side focal point]"
    }}
  ],
  "thumbnail_stats": {{
    "character_approach": "{thumbnail_character_selection['character_used']}",
    "selection_reasoning": "{thumbnail_character_selection['reasoning']}",
    "positioning_optimized": "RIGHT-side character positioning for text overlay compatibility"
  }}
}}

Create the PERFECT thumbnail for sleep story viewers with optimal text placement zones."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CONFIG.claude_config["model"],
                max_tokens=4096,
                temperature=0.4,
                timeout=300,
                system="You are a YouTube thumbnail optimization specialist who understands sleep content marketing, visual psychology, and optimal text placement strategies. Focus on right-side character positioning for maximum text overlay compatibility.",
                messages=[{"role": "user", "content": thumbnail_prompt}]
            )

            content = response.content[0].text if hasattr(response, 'content') else str(response)

            # Calculate cost
            input_tokens = len(thumbnail_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            print(f"✅ Thumbnail generation complete: {len(content):,} characters - Cost: ${stage_cost:.4f}")

            # Parse thumbnail result
            parsed_result = self._parse_claude_response(content, "thumbnail_generation")

            self.character_system.log_extraction_step("Thumbnail Generation", "SUCCESS", {
                "character_approach": thumbnail_character_selection['character_used'],
                "alternatives_generated": len(parsed_result.get('thumbnail_alternatives', [])),
                "positioning_optimized": "RIGHT-side character positioning",
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Thumbnail Generation Failed", "ERROR")
            CONFIG.logger.error(f"Thumbnail generation error: {e}")

            # Fallback thumbnail with right positioning
            fallback_thumbnail = {
                "thumbnail_prompt": {
                    "scene_number": 99,
                    "character_used": "None (Atmospheric focus)",
                    "clickbait_title": clickbait_title or f"The Secret History of {topic} (2 Hour Sleep Story)",
                    "font_design": font_design or "Bold impact font with warm colors",
                    "prompt": f"RIGHT-side positioned atmospheric view of {topic}, warm golden lighting, peaceful but compelling visual, left and center areas clear for text overlay",
                    "visual_style": "Peaceful and inviting",
                    "character_positioning": "RIGHT-side atmospheric focal point",
                    "text_overlay_strategy": "Left and center areas optimized for text placement",
                    "thumbnail_reasoning": "Fallback due to generation error"
                },
                "thumbnail_stats": {"error": str(e)}
            }
            return fallback_thumbnail

    def _combine_all_stages(self, stage1: Dict, stage2: Dict, character_data: Dict, thumbnail_data: Dict, hook_subscribe_data: Dict, topic: str, description: str) -> Dict[str, Any]:
        """Combine all five stages into final result with complete JSON outputs"""

        self.log_step("Combining All Stages with Complete JSON Outputs")

        # Merge stories
        all_stories = {}
        all_stories.update(stage1.get('stories', {}))
        all_stories.update(stage2.get('stories', {}))

        # Enhanced visual prompts INCLUDING THUMBNAIL AS SCENE 99
        enhanced_visual_prompts = stage1.get('visual_prompts', [])

        # ADD THUMBNAIL TO VISUAL PROMPTS AS SCENE 99
        thumbnail_prompt = thumbnail_data.get('thumbnail_prompt', {})
        if thumbnail_prompt:
            enhanced_visual_prompts.append(thumbnail_prompt)

        # Compile complete story text
        complete_story = self._compile_complete_story({
            **stage1,
            'stories': all_stories
        })

        # Final result with ALL JSON outputs
        result = {
            "hook_section": stage1.get("golden_hook", {}),
            "subscribe_section": stage1.get("subscribe_section", {}),
            "scene_plan": stage1.get("scene_plan", []),
            "complete_story": complete_story,
            "visual_prompts": enhanced_visual_prompts,  # INCLUDES THUMBNAIL AS SCENE 99
            "voice_directions": stage1.get("voice_directions", []),
            "stories": all_stories,

            # INTRO SEQUENCE VISUALS
            "intro_sequence_visuals": stage1.get("intro_sequence_visuals", {}),

            # HOOK & SUBSCRIBE SCENES (NEW!)
            "hook_subscribe_scenes": hook_subscribe_data,

            # CHARACTER DATA
            "main_characters": character_data.get('main_characters', []),
            "character_relationships": character_data.get('character_relationships', []),
            "scene_character_mapping": character_data.get('scene_character_mapping', {}),
            "visual_style_notes": character_data.get('visual_style_notes', {}),

            # YOUTUBE DATA
            "youtube_optimization": character_data.get('youtube_optimization', {}),

            # THUMBNAIL DATA
            "thumbnail_data": thumbnail_data,

            "generation_stats": {
                "api_calls_used": self.api_call_count,
                "five_stage_approach": True,
                "thumbnail_generated": bool(thumbnail_data.get('thumbnail_prompt')),
                "hook_subscribe_generated": bool(hook_subscribe_data.get('hook_scenes')),
                "scenes_planned": len(stage1.get("scene_plan", [])),
                "stories_written": len(all_stories),
                "stage1_stories": len(stage1.get('stories', {})),
                "stage2_stories": len(stage2.get('stories', {})),
                "characters_extracted": len(character_data.get('main_characters', [])),
                "visual_prompts_with_thumbnail": len(enhanced_visual_prompts),
                "production_ready": len(all_stories) >= 35,
                "server_optimized": True,
                "complete_pipeline": True
            },
            "generation_log": self.generation_log,
            "character_extraction_log": self.character_system.extraction_log,
            "topic": topic,
            "description": description,
            "generated_at": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"]
        }

        return result

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
                CONFIG.logger.warning(f"{stage}: Full JSON failed, extracting partial data...")
                return self._extract_partial_json(content, stage)

        except Exception as e:
            CONFIG.logger.error(f"{stage} parsing failed: {e}")
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
                    "voice_directions": self._extract_json_array(content, "voice_directions"),
                    "intro_sequence_visuals": self._extract_json_object(content, "intro_sequence_visuals")
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
                    "character_stats": self._extract_json_object(content, "character_stats")
                }
            elif stage == "thumbnail_generation":
                result = {
                    "thumbnail_prompt": self._extract_json_object(content, "thumbnail_prompt"),
                    "thumbnail_alternatives": self._extract_json_array(content, "thumbnail_alternatives"),
                    "thumbnail_stats": self._extract_json_object(content, "thumbnail_stats")
                }

        except Exception as e:
            CONFIG.logger.warning(f"Partial extraction error for {stage}: {e}")

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
            CONFIG.logger.error(f"Story extraction error: {e}")

        return stories

    def _create_fallback_scenes_21_40(self, given_topic: str) -> List[Dict]:
        """Fallback scenes 21-40 if stage1 incomplete"""
        scenes = []
        for i in range(21, 41):
            emotion = "mild concern" if i <= 30 else "deep peace"
            scenes.append({
                "scene_id": i,
                "title": f"Scene {i} - {given_topic} Moment",
                "location": f"Historical location {i}",
                "duration_minutes": 4,
                "template": ["atmospheric", "character_focused", "historical_detail", "sensory_journey"][i % 4],
                "narrative_style": ["observational", "immersive", "documentary", "poetic", "cinematic"][i % 5],
                "emotion": emotion,
                "sensory_focus": ["sight", "sound", "smell", "touch"][i % 4],
                "description": f"Atmospheric exploration of {given_topic}",
                "key_elements": ["historical", "atmospheric", "peaceful"],
                "characters_mentioned": []
            })
        return scenes

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
    """Save complete production outputs - SERVER VERSION WITH ALL JSON FILES"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
        # 1. Complete story text
        story_path = output_path / "complete_story.txt"
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(result["complete_story"])
        saved_files.append("complete_story.txt")

        # 2. Scene plan
        plan_path = output_path / "scene_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(result["scene_plan"], f, indent=2, ensure_ascii=False)
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
            "visual_style_notes": result.get("visual_style_notes", {})
        }
        with open(character_path, "w", encoding="utf-8") as f:
            json.dump(character_data, f, indent=2, ensure_ascii=False)
        saved_files.append("character_profiles.json")

        # 6. YouTube metadata
        youtube_path = output_path / "youtube_metadata.json"
        youtube_data = result.get("youtube_optimization", {})
        with open(youtube_path, "w", encoding="utf-8") as f:
            json.dump(youtube_data, f, indent=2, ensure_ascii=False)
        saved_files.append("youtube_metadata.json")

        # 7. Thumbnail data
        thumbnail_data = result.get("thumbnail_data", {})
        if thumbnail_data.get("thumbnail_prompt"):
            thumbnail_path = output_path / "thumbnail_generation.json"
            with open(thumbnail_path, "w", encoding="utf-8") as f:
                json.dump(thumbnail_data, f, indent=2, ensure_ascii=False)
            saved_files.append("thumbnail_generation.json")

        # 8. Hook & Subscribe scenes (NEW!)
        hook_subscribe_data = result.get("hook_subscribe_scenes", {})
        if hook_subscribe_data:
            hook_subscribe_path = output_path / "hook_subscribe_scenes.json"
            with open(hook_subscribe_path, "w", encoding="utf-8") as f:
                json.dump(hook_subscribe_data, f, indent=2, ensure_ascii=False)
            saved_files.append("hook_subscribe_scenes.json")

        # 9. Intro sequence visuals
        intro_visuals = result.get("intro_sequence_visuals", {})
        if intro_visuals:
            intro_path = output_path / "intro_sequence_visuals.json"
            intro_data = {
                "hook_visual": intro_visuals.get("hook_visual", {}),
                "subscribe_visual": intro_visuals.get("subscribe_visual", {}),
                "alternative_options": intro_visuals.get("alternative_options", {}),
                "usage_instructions": intro_visuals.get("usage_instructions", {}),
                "production_notes": {
                    "critical_importance": "These are the most important 60 seconds for viewer retention",
                    "hook_timing": "0-30 seconds: Use hook_visual with golden_hook narration",
                    "subscribe_timing": "30-60 seconds: Use subscribe_visual with subscribe_section narration",
                    "sync_requirement": "Visual must emotionally match the spoken content",
                    "fallback_usage": "Use alternative_options if primary scenes unavailable",
                    "testing_recommendation": "A/B test different visual choices for optimization"
                }
            }
            with open(intro_path, "w", encoding="utf-8") as f:
                json.dump(intro_data, f, indent=2, ensure_ascii=False)
            saved_files.append("intro_sequence_visuals.json")

        # 10. Audio generation prompts (NEW!)
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

        for scene_id, story_content in stories.items():
            voice_direction = next((v for v in voice_directions if v.get("scene_number") == int(scene_id)), {})

            audio_prompts.append({
                "segment_id": f"scene_{scene_id}",
                "content": story_content,
                "duration_minutes": 4,
                "voice_direction": voice_direction.get("direction", ""),
                "template": voice_direction.get("template", "atmospheric"),
                "style": voice_direction.get("style", "observational"),
                "tts_settings": {
                    "voice": "alloy",
                    "speed": 0.85,
                    "pitch": "calm",
                    "emphasis": "sleep-optimized"
                }
            })

        audio_path = output_path / "audio_generation_prompts.json"
        with open(audio_path, "w", encoding="utf-8") as f:
            json.dump(audio_prompts, f, indent=2, ensure_ascii=False)
        saved_files.append("audio_generation_prompts.json")

        # 11. Video composition instructions (NEW!)
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
                    "end_time": 7260,  # 2 hours
                    "video_source": "visual_generation_prompts.json -> scenes 1-40",
                    "audio_source": "audio_generation_prompts.json -> scenes 1-40",
                    "text_overlay": "Scene titles (optional)",
                    "transition": "crossfade_between_scenes"
                }
            ],
            "scene_sync_strategy": {
                "rule": "When audio mentions scene X, display scene X visual",
                "timing": "Immediate visual sync with narrative",
                "duration": "Each scene visual = 4 minutes (240 seconds)",
                "overlap": "15 second crossfade between scenes"
            },
            "thumbnail_usage": {
                "source": "visual_generation_prompts.json -> scene 99",
                "purpose": "YouTube thumbnail only, not used in video timeline",
                "positioning": "Character on right, text overlay space on left/center"
            },
            "production_settings": {
                "resolution": "1920x1080",
                "frame_rate": 30,
                "audio_bitrate": 192,
                "video_codec": "h264",
                "total_duration": "7200 seconds (2 hours)"
            }
        }

        video_path = output_path / "video_composition_instructions.json"
        with open(video_path, "w", encoding="utf-8") as f:
            json.dump(video_composition, f, indent=2, ensure_ascii=False)
        saved_files.append("video_composition_instructions.json")

        # 12. Generation report
        report_path = output_path / "generation_report.json"
        production_report = {
            "topic": story_topic,
            "generation_completed": datetime.now().isoformat(),
            "model_used": CONFIG.claude_config["model"],
            "server_optimized": True,
            "complete_pipeline": True,
            "stats": result["generation_stats"],
            "files_saved": saved_files,
            "next_steps": [
                "1. Generate character reference images using character_profiles.json",
                "2. Generate scene visuals (1-40) using visual_generation_prompts.json",
                "3. Generate thumbnail (scene 99) using visual_generation_prompts.json",
                "4. Generate audio using audio_generation_prompts.json",
                "5. Compose video using video_composition_instructions.json",
                "6. Upload to YouTube using youtube_metadata.json"
            ]
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ Complete production files saved: {saved_files}")
        CONFIG.logger.info(f"Files saved to: {output_dir}")

        # Mark topic as completed in database
        scene_count = len(result.get('scene_plan', []))
        total_duration = result.get('generation_stats', {}).get('total_duration_minutes', 0)

        complete_topic_in_database(
            topic_id, scene_count, total_duration, api_calls, total_cost, output_dir
        )

    except Exception as e:
        print(f"❌ Save error: {e}")
        CONFIG.logger.error(f"Save error: {e}")

def print_production_summary(result: Dict, story_topic: str, output_path: str, generation_time: float):
    """Print complete production generation summary - SERVER VERSION"""
    stats = result["generation_stats"]

    print("\n" + "🚀" * 60)
    print("COMPLETE AUTOMATED STORY GENERATOR - PRODUCTION FINISHED!")
    print("🚀" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CONFIG.claude_config['model']}")
    print(f"🖥️  Server Mode: {'✅ ACTIVE' if stats.get('server_optimized') else '❌ OFF'}")
    print(f"🏭 Complete Pipeline: {'✅ ACTIVE' if stats.get('complete_pipeline') else '❌ OFF'}")

    print(f"\n📊 PRODUCTION PERFORMANCE:")
    print(f"🔥 Total API Calls: {stats['api_calls_used']}")
    print(f"⏱️  Total Generation Time: {generation_time:.1f}s")
    print(f"🎬 Scenes Planned: {stats['scenes_planned']}/40")
    print(f"📝 Stories Written: {stats['stories_written']}/40")
    print(f"👥 Characters Extracted: {stats['characters_extracted']}")
    print(f"🖼️  Thumbnail Generated: {'✅ YES' if stats.get('thumbnail_generated') else '❌ NO'}")
    print(f"🎭 Hook & Subscribe: {'✅ YES' if stats.get('hook_subscribe_generated') else '❌ NO'}")
    print(f"🎥 Visual Prompts (with thumbnail): {stats.get('visual_prompts_with_thumbnail', 0)}")

    completion_rate = (stats['stories_written'] / 40) * 100
    print(f"📊 Story Completion: {completion_rate:.1f}%")

    # Hook & Subscribe scenes info
    hook_subscribe_data = result.get("hook_subscribe_scenes", {})
    if hook_subscribe_data.get("hook_scenes"):
        print(f"🎬 Hook Scenes: {len(hook_subscribe_data['hook_scenes'])} scenes (0-30s)")
        print(f"📺 Subscribe Scenes: {len(hook_subscribe_data['subscribe_scenes'])} scenes (30-60s)")

    # Intro sequence visuals info
    intro_visuals = result.get("intro_sequence_visuals", {})
    if intro_visuals.get("hook_visual"):
        hook_scene = intro_visuals["hook_visual"].get("recommended_scene", "N/A")
        subscribe_scene = intro_visuals["subscribe_visual"].get("recommended_scene", "N/A")
        print(f"🎬 Best Hook Visual: Scene {hook_scene} - {intro_visuals['hook_visual'].get('scene_title', 'N/A')}")
        print(f"📺 Best Subscribe Visual: Scene {subscribe_scene} - {intro_visuals['subscribe_visual'].get('scene_title', 'N/A')}")

    # Thumbnail info
    thumbnail_data = result.get("thumbnail_data", {})
    if thumbnail_data.get("thumbnail_prompt"):
        thumb_char = thumbnail_data["thumbnail_prompt"].get("character_used", "N/A")
        print(f"🖼️  Thumbnail Character: {thumb_char}")
        print(f"🎯 Thumbnail Positioning: RIGHT-side optimized for text overlay")

    if completion_rate >= 80:
        print(f"\n🎉 EXCELLENT SUCCESS!")
        print(f"✅ Ready for complete automated pipeline")
    elif completion_rate >= 60:
        print(f"\n✅ GOOD PROGRESS!")
        print(f"⚡ Suitable for production with minor adjustments")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"🔍 Review generation_report.json for issues")

    print("\n📄 GENERATED FILES:")
    print("1. 📖 complete_story.txt - Full story text")
    print("2. 🎬 scene_plan.json - 40 scene structure")
    print("3. 🖼️  visual_generation_prompts.json - Scenes 1-40 + Thumbnail (99)")
    print("4. 🎵 voice_directions.json - TTS guidance")
    print("5. 👥 character_profiles.json - Character data")
    print("6. 📺 youtube_metadata.json - Upload data")
    print("7. 🖼️  thumbnail_generation.json - Thumbnail details")
    print("8. 🎭 hook_subscribe_scenes.json - Background scenes")
    print("9. 🎬 intro_sequence_visuals.json - Hook & Subscribe visuals")
    print("10. 🎵 audio_generation_prompts.json - TTS production")
    print("11. 🎥 video_composition_instructions.json - Video timeline")
    print("12. 📊 generation_report.json - Complete summary")

    print("\n🎯 PRODUCTION PIPELINE:")
    print("1. 🎭 Generate characters using character_profiles.json")
    print("2. 🖼️  Generate scenes 1-40 using visual_generation_prompts.json")
    print("3. 🎯 Generate thumbnail (scene 99) using visual_generation_prompts.json")
    print("4. 🎵 Generate audio using audio_generation_prompts.json")
    print("5. 🎥 Compose video using video_composition_instructions.json")
    print("6. ⬆️  Upload to YouTube using youtube_metadata.json")

    print("🚀" * 60)

if __name__ == "__main__":
    try:
        print("🚀 COMPLETE AUTOMATED STORY GENERATOR")
        print("⚡ Server-optimized with complete pipeline")
        print("🎭 5-stage approach: Planning + Stories + Characters + Thumbnail + Hook/Subscribe")
        print("📄 Complete JSON outputs for automation")
        print("🎯 RIGHT-side thumbnail positioning for text overlay")
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

        # Generate complete story
        start_time = time.time()
        result = generator.generate_complete_story_with_characters(
            topic, description, clickbait_title, font_design
        )
        generation_time = time.time() - start_time

        # Save outputs
        save_production_outputs(str(output_path), result, topic, topic_id,
                               generator.api_call_count, generator.total_cost)

        # Print summary
        print_production_summary(result, topic, str(output_path), generation_time)

        print(f"\n🚀 COMPLETE PRODUCTION PIPELINE FINISHED!")
        print(f"✅ All files saved to: {output_path}")
        print(f"📊 Check generation_report.json for details")
        print(f"🎬 12 JSON files ready for complete automation!")
        print(f"🎯 Thumbnail scene 99 in visual_generation_prompts.json")
        print(f"🎭 Hook & Subscribe scenes ready for video composition")

    except Exception as e:
        print(f"\n💥 COMPLETE GENERATOR ERROR: {e}")
        CONFIG.logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()