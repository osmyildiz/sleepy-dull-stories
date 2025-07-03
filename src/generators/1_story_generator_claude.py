"""
Sleepy Dull Stories - Claude 4 Story Generator with DYNAMIC CHARACTER EXTRACTION + THUMBNAIL GENERATION
Complete solution: Stage 1 (Planning + 20 stories) + Stage 2 (Remaining 20 stories) + Character Analysis + Thumbnail Generation
Automatic Character Extraction for Any Story Topic with Reference Image Pipeline + Smart Thumbnail Creation
UPDATED: Fixed Visual Prompt Generation + Added Intelligent Thumbnail System
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Try to import from existing config first, fallback to env
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src import config
    print("✅ Config system loaded")
except ImportError:
    print("⚠️ Config system not found, using environment variables")
    config = None

# Anthropic import
try:
    from anthropic import Anthropic
    print("✅ Anthropic library loaded")
except ImportError:
    print("❌ Anthropic library not found. Install: pip install anthropic")
    sys.exit(1)

load_dotenv()

# File paths - keep same structure as v2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPIC_CSV_PATH = os.path.join(BASE_DIR, "data", "topics.csv")

# Claude 4 Configuration - CHARACTER EXTRACTION + THUMBNAIL VERSION
CLAUDE_CONFIG = {
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
    "test_mode": False
}

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
                char_role = character.get('role', '').lower()

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
    """Automated story generation + character extraction + thumbnail generation + production optimization"""

    def __init__(self):
        """Initialize story generator with character extraction and thumbnail system"""
        self.generation_log = []
        self.api_call_count = 0
        self.character_system = CharacterExtractionSystem()

        # Try to get API key from config first, then environment
        api_key = None

        if config and hasattr(config, 'CLAUDE_API_KEY'):
            api_key = config.CLAUDE_API_KEY
            print("✅ Claude API key loaded from config")
        elif config and hasattr(config, 'ANTHROPIC_API_KEY'):
            api_key = config.ANTHROPIC_API_KEY
            print("✅ Anthropic API key loaded from config")
        else:
            api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                print("✅ Claude API key loaded from environment")

        if not api_key:
            raise ValueError("❌ Claude API key not found. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY in config or environment")

        try:
            self.client = Anthropic(api_key=api_key)
            print("✅ Story generator initialized with character extraction + thumbnail generation")
        except Exception as e:
            print(f"❌ Story generator initialization failed: {e}")
            raise

    def log_step(self, description: str, status: str = "START", metadata: Dict = None):
        """Same logging system as v2"""
        entry = {
            "description": description,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "api_calls": self.api_call_count
        }
        if metadata:
            entry.update(metadata)
        self.generation_log.append(entry)

        icon = "🔄" if status == "START" else "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {description} [API calls: {self.api_call_count}]")

    def generate_complete_story_with_characters(self, topic: str, description: str, clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """
        FOUR-STAGE APPROACH: Complete story generation + character extraction + thumbnail generation
        Stage 1: Planning + Hook + Subscribe + 20 stories + Prompts
        Stage 2: Remaining 20 stories
        Stage 3: Character extraction and analysis + VISUAL PROMPT REGENERATION
        Stage 4: INTELLIGENT THUMBNAIL GENERATION
        """

        self.log_step("Automated Story Generation with Character Extraction + Thumbnail Generation")

        # STAGE 1: Planning + First Half
        stage1_result = self._generate_stage1(topic, description)

        # Short pause between stages
        time.sleep(2)

        # STAGE 2: Second Half Stories
        stage2_result = self._generate_stage2(topic, description, stage1_result)

        # Short pause before character analysis
        time.sleep(2)

        # STAGE 3: Character Extraction + Visual Prompt Regeneration
        character_result = self._extract_characters(topic, description, stage1_result, stage2_result)

        # Short pause before thumbnail generation
        time.sleep(2)

        # STAGE 4: Intelligent Thumbnail Generation
        thumbnail_result = self._generate_intelligent_thumbnail(
            topic, description, character_result, clickbait_title, font_design
        )

        # COMBINE: Merge all stages
        combined_result = self._combine_all_stages(
            stage1_result, stage2_result, character_result, thumbnail_result, topic, description
        )

        self.log_step("Automated Generation Complete", "SUCCESS", {
            "total_scenes": len(combined_result.get('scene_plan', [])),
            "total_stories": len(combined_result.get('stories', {})),
            "characters_extracted": len(combined_result.get('main_characters', [])),
            "visual_prompts_regenerated": combined_result.get('generation_stats', {}).get('visual_prompts_regenerated', False),
            "thumbnail_generated": combined_result.get('generation_stats', {}).get('thumbnail_generated', False),
            "api_calls_total": self.api_call_count
        })

        return combined_result

    def _generate_stage1(self, topic: str, description: str) -> Dict[str, Any]:
        """STAGE 1: Complete planning + Hook + Subscribe + 20 stories + Basic Prompts"""

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
- CRITICAL: ABSOLUTELY FORBIDDEN to use "You find yourself" as opening
- AS A MASTER STORYTELLER: Create unique, atmospheric openings for each scene
- Examples for inspiration (but create your own variations):
  * Environmental: "The golden light filters through..."
  * Temporal: "As twilight settles over..."
  * Auditory: "Soft footsteps echo in..."
  * Sensory: "The gentle breeze carries..."
  * Visual: "Shadows dance across..."
- YOUR TASK: Invent fresh, varied openings that set mood and place immediately

## 5. BASIC VISUAL PROMPTS (All 40 scenes)
- Simple AI image generation prompts (will be regenerated with character data later)
- Focus on location and atmosphere
- Character presence can be noted but details will be added later

## 6. VOICE DIRECTIONS (All 40 scenes)
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
    "1": "[COMPLETE 450-600 word story for scene 1 with character interactions]",
    "2": "[COMPLETE story for scene 2]"
  }},
  "visual_prompts": [
    {{
      "scene_number": 1,
      "title": "[Scene title]",
      "prompt": "[Basic AI image prompt - will be enhanced with characters later]",
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
  }}
}}

Generate complete Stage 1 content now. Write all 20 stories fully with rich character interactions and SHOWCASE YOUR MASTERY with creative, unique openings for each scene.

⚠️ MASTER STORYTELLER CHALLENGE: Create 20 completely different atmospheric openings. The phrase "You find yourself" is banned. Instead, demonstrate your expertise by crafting immersive beginnings that vary in style - some environmental, some temporal, some sensory, some character-focused. Make each opening memorable and distinct."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CLAUDE_CONFIG["model"],
                max_tokens=CLAUDE_CONFIG["max_tokens"],
                temperature=CLAUDE_CONFIG["temperature"],
                stream=True,
                timeout=1800,
                system="You are a MASTER STORYTELLER and automated content creator. Stage 1: Create complete planning + first 20 atmospheric stories with rich character interactions. Focus on memorable, distinct characters. SHOWCASE YOUR STORYTELLING MASTERY with unique, creative openings for each scene. The phrase 'You find yourself' is forbidden - instead, craft atmospheric beginnings that immediately immerse the reader.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            # Collect streaming response
            content = ""
            print("📡 Stage 1: Streaming Claude 4 response...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 1: {len(content):,} characters...")

            print(f"✅ Stage 1 complete: {len(content):,} characters")

            # Parse Stage 1 result
            parsed_result = self._parse_claude_response(content, "stage1")

            self.log_step("Stage 1 Parsing", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stories_written": len(parsed_result.get('stories', {}))
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 1 Failed", "ERROR")
            print(f"❌ Stage 1 error: {e}")
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
- AS AN EXPERT STORYTELLER: Create unique atmospheric openings for each scene
- FORBIDDEN PHRASE: "You find yourself" is completely banned
- CREATIVE FREEDOM: Invent your own varied openings beyond these examples:
  * Environmental, temporal, auditory, visual, sensory approaches
  * Each scene should have a distinct, memorable beginning
  * Draw inspiration from the setting, mood, and character emotions

OUTPUT FORMAT:
{{
  "stories": {{
    "21": "[COMPLETE 450-600 word story for scene 21 with character interactions]",
    "22": "[COMPLETE story for scene 22]"
  }},
  "stage2_stats": {{
    "stories_written": 20,
    "scenes_covered": "21-40",
    "total_word_count": "[calculated]",
    "character_development": "continued"
  }}
}}

Write all 20 remaining stories completely with full character development.

⚠️ STORYTELLING MASTERY TEST: Create 20 more unique atmospheric openings (scenes 21-40). Continue demonstrating your expertise with varied, creative beginnings. "You find yourself" remains forbidden. Each scene should start differently - show your storytelling range and creativity."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CLAUDE_CONFIG["model"],
                max_tokens=CLAUDE_CONFIG["max_tokens"],
                temperature=CLAUDE_CONFIG["temperature"],
                stream=True,
                timeout=1800,
                system="You are a MASTER STORYTELLER and automated content creator. Stage 2: Complete the remaining 20 stories with rich character development and consistent character interactions from Stage 1. DEMONSTRATE YOUR STORYTELLING EXPERTISE with inventive, atmospheric openings for each scene. Never use 'You find yourself' - create unique beginnings that set mood and place.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            # Collect streaming response
            content = ""
            print("📡 Stage 2: Streaming Claude 4 response...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 2: {len(content):,} characters...")

            print(f"✅ Stage 2 complete: {len(content):,} characters")

            # Parse Stage 2 result
            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 Parsing", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {}))
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 Failed", "ERROR")
            print(f"❌ Stage 2 error: {e}")
            # Return partial result instead of failing completely
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
- Identify maximum {CLAUDE_CONFIG['max_characters']} main characters
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
                model=CLAUDE_CONFIG["model"],
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

            # Parse character extraction result
            parsed_result = self._parse_claude_response(content, "character_extraction")

            # Process characters through extraction system
            if 'main_characters' in parsed_result:
                # Filter to top characters
                top_characters = self.character_system.filter_top_characters(
                    parsed_result['main_characters'],
                    CLAUDE_CONFIG['max_characters']
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
                "visual_prompts_regenerated": 'regenerated_visual_prompts' in parsed_result
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
                model=CLAUDE_CONFIG["model"],
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

            # Parse thumbnail result
            parsed_result = self._parse_claude_response(content, "thumbnail_generation")

            self.character_system.log_extraction_step("Thumbnail Generation", "SUCCESS", {
                "character_approach": thumbnail_character_selection['character_used'],
                "selection_reasoning": thumbnail_character_selection['reasoning'],
                "alternatives_generated": len(parsed_result.get('thumbnail_alternatives', []))
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

        regeneration_prompt = f"""Based on the completed scene plan and extracted character data, create accurate visual generation prompts for all 40 scenes.

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

Generate all 40 visual prompts that accurately reflect the scene plan and character presence.

OUTPUT (Complete JSON array of all 40 scenes):
"""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=CLAUDE_CONFIG["model"],
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
                    "character_integrated": True
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

    def _combine_all_stages(self, stage1: Dict, stage2: Dict, character_data: Dict, thumbnail_data: Dict, topic: str, description: str) -> Dict[str, Any]:
        """Combine all four stages into final result - USING REGENERATED VISUAL PROMPTS + THUMBNAIL"""

        self.log_step("Combining All Stages with Regenerated Visual Prompts + Thumbnail")

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

            # PRODUCTION DATA
            "production_specifications": character_data.get('production_specifications', {}),

            "generation_stats": {
                "api_calls_used": self.api_call_count,
                "four_stage_approach": True,
                "visual_prompts_regenerated": 'regenerated_visual_prompts' in character_data,
                "thumbnail_generated": bool(thumbnail_data.get('thumbnail_prompt')),
                "scenes_planned": len(stage1.get("scene_plan", [])),
                "stories_written": len(all_stories),
                "stage1_stories": len(stage1.get('stories', {})),
                "stage2_stories": len(stage2.get('stories', {})),
                "characters_extracted": len(character_data.get('main_characters', [])),
                "production_ready": len(all_stories) >= 35,
                "total_duration_minutes": len(stage1.get("scene_plan", [])) * 4 + 1,
                "automated_production_ready": True
            },
            "generation_log": self.generation_log,
            "character_extraction_log": self.character_system.extraction_log,
            "topic": topic,
            "description": description,
            "generated_at": datetime.now().isoformat(),
            "model_used": CLAUDE_CONFIG["model"],
            "enhancement_status": "visual_prompts_regenerated_with_characters_and_thumbnail"
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

# CSV handling functions (updated for thumbnail data)
def get_next_topic_and_update_csv(csv_path: str) -> Tuple[int, str, str, str, str]:
    """Get next topic from CSV and mark as done - NOW WITH THUMBNAIL DATA"""
    df = pd.read_csv(csv_path)
    next_row = df[df["done"] == 0].head(1)
    if next_row.empty:
        raise ValueError("No topics remaining")

    index = next_row.index[0]
    story_topic = next_row.iloc[0]["topic"]
    story_description = next_row.iloc[0]["description"]

    # Get thumbnail data if available
    clickbait_title = next_row.iloc[0].get("clickbait_title", None)
    font_design = next_row.iloc[0].get("font_design", None)

    # Mark as done
    df.at[index, "done"] = 1
    df.to_csv(csv_path, index=False)

    return index + 1, story_topic, story_description, clickbait_title, font_design

def save_production_outputs(output_dir: str, result: Dict, story_topic: str):
    """Save production outputs with updated visual prompts including thumbnail"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    try:
        # Complete story
        story_path = os.path.join(output_dir, "complete_story.txt")
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(result["complete_story"])
        saved_files.append("complete_story.txt")

        # Scene plan
        plan_path = os.path.join(output_dir, "scene_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(result["scene_plan"], f, indent=2, ensure_ascii=False)
        saved_files.append("scene_plan.json")

        # ENHANCED visual prompts with thumbnail
        visual_path = os.path.join(output_dir, "visual_generation_prompts.json")
        with open(visual_path, "w", encoding="utf-8") as f:
            json.dump(result["visual_prompts"], f, indent=2, ensure_ascii=False)
        saved_files.append("visual_generation_prompts.json")

        # Separate thumbnail file for easy access
        thumbnail_data = result.get("thumbnail_data", {})
        if thumbnail_data.get("thumbnail_prompt"):
            thumbnail_path = os.path.join(output_dir, "thumbnail_generation.json")
            with open(thumbnail_path, "w", encoding="utf-8") as f:
                json.dump(thumbnail_data, f, indent=2, ensure_ascii=False)
            saved_files.append("thumbnail_generation.json")

        # Character profiles
        character_path = os.path.join(output_dir, "character_profiles.json")
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

        # Production specifications
        production_path = os.path.join(output_dir, "automation_specs.json")
        production_data = {
            "audio_production": result.get("production_specifications", {}).get("audio_production", {}),
            "video_assembly": result.get("production_specifications", {}).get("video_assembly", {}),
            "quality_control": result.get("production_specifications", {}).get("quality_control", {}),
            "automation_specifications": result.get("production_specifications", {}).get("automation_specifications", {}),
            "precise_timing_breakdown": result.get("production_specifications", {}).get("precise_timing_breakdown", {}),
            "implementation_ready": True
        }
        with open(production_path, "w", encoding="utf-8") as f:
            json.dump(production_data, f, indent=2, ensure_ascii=False)
        saved_files.append("automation_specs.json")

        # Platform optimization data
        youtube_path = os.path.join(output_dir, "platform_metadata.json")
        youtube_data = {
            "video_metadata": result.get("youtube_optimization", {}).get("youtube_metadata", {}),
            "title_options": result.get("youtube_optimization", {}).get("clickbait_titles", []),
            "description": result.get("youtube_optimization", {}).get("video_description", {}),
            "tags": result.get("youtube_optimization", {}).get("tags", []),
            "hashtags": result.get("youtube_optimization", {}).get("hashtags", []),
            "seo_strategy": result.get("youtube_optimization", {}).get("seo_strategy", {}),
            "thumbnail_concept": result.get("youtube_optimization", {}).get("thumbnail_concept", {}),
            "engagement_strategy": result.get("youtube_optimization", {}).get("engagement_strategy", {}),
            "analytics_tracking": result.get("youtube_optimization", {}).get("analytics_tracking", {}),
            "api_ready_format": {
                "snippet": {
                    "title": result.get("youtube_optimization", {}).get("clickbait_titles", ["Sleep Story"])[0],
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
        }
        with open(youtube_path, "w", encoding="utf-8") as f:
            json.dump(youtube_data, f, indent=2, ensure_ascii=False)
        saved_files.append("platform_metadata.json")

        # Voice directions
        voice_path = os.path.join(output_dir, "voice_directions.json")
        with open(voice_path, "w", encoding="utf-8") as f:
            json.dump(result["voice_directions"], f, indent=2, ensure_ascii=False)
        saved_files.append("voice_directions.json")

        # All stories
        stories_path = os.path.join(output_dir, "all_stories.json")
        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(result["stories"], f, indent=2, ensure_ascii=False)
        saved_files.append("all_stories.json")

        # Production report with thumbnail info
        report_path = os.path.join(output_dir, "generation_report.json")
        production_report = {
            "topic": story_topic,
            "generation_completed": datetime.now().isoformat(),
            "model_used": CLAUDE_CONFIG["model"],
            "four_stage_approach": True,
            "character_extraction": True,
            "thumbnail_generation": True,
            "visual_prompts_regenerated": result.get("generation_stats", {}).get("visual_prompts_regenerated", False),
            "thumbnail_generated": result.get("generation_stats", {}).get("thumbnail_generated", False),
            "token_limit_per_stage": CLAUDE_CONFIG["max_tokens"],
            "stats": result["generation_stats"],
            "character_stats": {
                "characters_extracted": len(result.get("main_characters", [])),
                "character_names": [c.get("name", "Unknown") for c in result.get("main_characters", [])],
                "scenes_with_characters": len([k for k, v in result.get("scene_character_mapping", {}).items() if v]),
                "character_coverage": f"{len([k for k, v in result.get('scene_character_mapping', {}).items() if v])}/40 scenes"
            },
            "thumbnail_stats": result.get("thumbnail_data", {}).get("thumbnail_stats", {}),
            "youtube_stats": {
                "title_options": len(result.get("youtube_optimization", {}).get("clickbait_titles", [])),
                "primary_keywords": len(result.get("youtube_optimization", {}).get("seo_strategy", {}).get("primary_keywords", [])),
                "total_tags": len(result.get("youtube_optimization", {}).get("tags", [])),
                "hashtags_count": len(result.get("youtube_optimization", {}).get("hashtags", [])),
                "api_ready": True if result.get("youtube_optimization", {}).get("youtube_metadata") else False
            },
            "files_saved": saved_files,
            "next_steps": {
                "step1": "Use character_profiles.json to generate reference images",
                "step2": "Generate scene visuals using visual_generation_prompts.json (scenes 1-40)",
                "step3": "Generate thumbnail using scene_number 99 in visual_generation_prompts.json",
                "step4": "Upload video using platform_metadata.json API format",
                "step5": "Monitor analytics using tracking guidelines"
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(production_report, f, indent=2, ensure_ascii=False)
        saved_files.append("generation_report.json")

        print(f"✅ Production files saved: {saved_files}")

    except Exception as e:
        print(f"❌ Save error: {e}")

def print_production_summary(result: Dict, story_topic: str, output_path: str, generation_time: float):
    """Print production generation summary with thumbnail info"""
    stats = result["generation_stats"]
    characters = result.get("main_characters", [])
    thumbnail_data = result.get("thumbnail_data", {})

    print("\n" + "🚀" * 60)
    print("AUTOMATED STORY GENERATOR - PRODUCTION COMPLETE WITH THUMBNAIL!")
    print("🚀" * 60)

    print(f"📚 Topic: {story_topic}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Model: {CLAUDE_CONFIG['model']}")
    print(f"⚡ Approach: Four-stage (5 API calls with thumbnail generation)")

    print(f"\n📊 CLAUDE 4 CHARACTER + THUMBNAIL PERFORMANCE:")
    print(f"🔥 Total API Calls: {stats['api_calls_used']} (including thumbnail generation)")
    print(f"⏱️  Total Generation Time: {generation_time:.1f}s")
    print(f"🎬 Scenes Planned: {stats['scenes_planned']}/40")
    print(f"📝 Stories Written: {stats['stories_written']}/40")
    print(f"👥 Characters Extracted: {stats['characters_extracted']}")
    print(f"🎨 Visual Prompts Regenerated: {'✅ YES' if stats.get('visual_prompts_regenerated') else '❌ NO'}")
    print(f"🖼️  Thumbnail Generated: {'✅ YES' if stats.get('thumbnail_generated') else '❌ NO'}")

    print(f"\n📋 STAGE BREAKDOWN:")
    print(f"Stage 1: Planning + {stats['stage1_stories']} stories")
    print(f"Stage 2: {stats['stage2_stories']} additional stories")
    print(f"Stage 3: {stats['characters_extracted']} characters extracted")
    print(f"Stage 3b: Visual prompts regenerated with character integration")
    print(f"Stage 4: Intelligent thumbnail generation")

    if characters:
        print(f"\n👥 MAIN CHARACTERS:")
        for char in characters:
            print(f"• {char.get('name', 'Unknown')} ({char.get('role', 'unknown role')}) - Score: {char.get('importance_score', 0)}/10")

    # Thumbnail info
    if thumbnail_data.get('thumbnail_prompt'):
        thumbnail_prompt = thumbnail_data['thumbnail_prompt']
        print(f"\n🖼️  THUMBNAIL INFO:")
        print(f"Character Used: {thumbnail_prompt.get('character_used', 'Unknown')}")
        print(f"Title: {thumbnail_prompt.get('clickbait_title', 'Not set')[:50]}...")
        print(f"Visual Style: {thumbnail_prompt.get('visual_style', 'Not set')[:50]}...")

    scene_character_coverage = len([k for k, v in result.get('scene_character_mapping', {}).items() if v])
    print(f"🎭 Character Coverage: {scene_character_coverage}/40 scenes")

    # YouTube optimization stats
    youtube_opt = result.get("youtube_optimization", {})
    title_count = len(youtube_opt.get("clickbait_titles", []))
    tag_count = len(youtube_opt.get("tags", []))
    keyword_count = len(youtube_opt.get("seo_strategy", {}).get("primary_keywords", []))

    print(f"🎬 YouTube Titles Generated: {title_count}/10")
    print(f"🏷️  SEO Tags: {tag_count}")
    print(f"🔍 Primary Keywords: {keyword_count}")

    # Production specifications stats
    production_specs = result.get("production_specifications", {})
    audio_ready = bool(production_specs.get("audio_production"))
    video_ready = bool(production_specs.get("video_assembly"))
    automation_ready = bool(production_specs.get("automation_specifications"))

    print(f"🎵 Audio Production Specs: {'✅' if audio_ready else '❌'}")
    print(f"🎬 Video Assembly Specs: {'✅' if video_ready else '❌'}")
    print(f"🤖 Automation Ready: {'✅' if automation_ready else '❌'}")

    visual_count = len(result.get("visual_prompts", []))
    voice_count = len(result.get("voice_directions", []))
    print(f"🖼️  Enhanced Visual Prompts (with thumbnail): {visual_count}")
    print(f"🎤 Voice Directions: {voice_count}")

    completion_rate = (stats['stories_written'] / 40) * 100
    character_rate = (stats['characters_extracted'] / CLAUDE_CONFIG['max_characters']) * 100
    print(f"📊 Story Completion: {completion_rate:.1f}%")
    print(f"👥 Character Extraction: {character_rate:.1f}%")

    # Success evaluation
    youtube_ready = title_count >= 8 and tag_count >= 20 and keyword_count >= 4
    production_ready = audio_ready and video_ready and automation_ready
    visual_prompts_fixed = stats.get('visual_prompts_regenerated', False)
    thumbnail_ready = stats.get('thumbnail_generated', False)

    if (stats['stories_written'] >= 35 and stats['characters_extracted'] >= 3 and
        youtube_ready and production_ready and visual_prompts_fixed and thumbnail_ready):
        print(f"\n🎉 MASSIVE SUCCESS!")
        print(f"✅ Complete story + character + YouTube + production + thumbnail system")
        print(f"✅ Visual prompts FIXED to match scene content")
        print(f"✅ Intelligent thumbnail generation with character analysis")
        print(f"✅ Ready for FULL AUTOMATION")
        print(f"🚀 Zero manual work needed!")
    elif (stats['stories_written'] >= 25 and stats['characters_extracted'] >= 2 and
          (youtube_ready or production_ready) and thumbnail_ready):
        print(f"\n✅ EXCELLENT PROGRESS!")
        print(f"✅ Strong foundation + automation specs + thumbnail")
        print(f"⚡ Ready for automated pipeline")
        print(f"🎯 Production deployment recommended")
        if visual_prompts_fixed:
            print(f"✅ Visual prompts correctly regenerated")
    elif stats['stories_written'] >= 15 and stats['characters_extracted'] >= 1:
        print(f"\n👍 GOOD FOUNDATION!")
        print(f"✅ Solid story base + partial automation")
        print(f"💡 Production specs {'complete' if production_ready else 'need refinement'}")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"🔍 Review generation and automation logs")

    print("\n💰 EFFICIENCY vs MANUAL WORK:")
    print(f"💵 Cost: 5 API calls vs manual character definition + thumbnail design")
    print(f"⚡ Speed: Automatic character extraction + visual prompt regeneration + intelligent thumbnail")
    print(f"🔧 Consistency: Built-in character mapping + scene-visual alignment + thumbnail optimization")
    print(f"🎯 Scalability: Works for any story topic")
    print(f"🖼️  Intelligence: Smart character selection for thumbnails")

    print("\n🎨 FULL PRODUCTION PIPELINE:")
    print(f"1. 📋 Use character_profiles.json for reference generation")
    print(f"2. 🎭 Generate {len(characters)} character reference images")
    print(f"3. 🖼️  Use visual_generation_prompts.json for scene generation (1-40)")
    print(f"4. 🔗 Reference characters in {scene_character_coverage} character scenes")
    print(f"5. 🌟 Atmospheric-only rendering for {40 - scene_character_coverage} landscape scenes")
    print(f"6. 🎯 Generate thumbnail using scene_number 99 in visual_generation_prompts.json")
    print(f"7. 📺 Upload using API-ready format")
    print(f"8. 📊 Monitor analytics and engagement metrics")

    print("\n💡 NEXT STEPS:")
    if (completion_rate >= 80 and character_rate >= 60 and youtube_ready and
        production_ready and visual_prompts_fixed and thumbnail_ready):
        print("1. 🎉 COMPLETE AUTOMATION SYSTEM READY!")
        print("2. 🤖 Run automated production pipeline")
        print("3. 📺 Automated YouTube upload with thumbnail")
        print("4. 📊 Monitor analytics automatically")
    else:
        print("1. 🔍 Review automation completeness")
        print("2. 🔧 Complete missing specifications")
        print("3. 🚀 Deploy automated pipeline")

    print("\n🏆 COMPLETE AUTOMATION ADVANTAGES:")
    print("✅ Dynamic character extraction for any topic")
    print("✅ Automatic consistency mapping")
    print("✅ Visual generation pipeline ready")
    print("✅ FIXED: Visual prompts match scene content exactly")
    print("✅ Character-scene mapping for perfect consistency")
    print("✅ INTELLIGENT THUMBNAIL GENERATION")
    print("✅ Character analysis for optimal thumbnail selection")
    print("✅ Clickbait optimization while maintaining sleep content feel")
    print("✅ Platform API-ready optimization")
    print("✅ Complete audio production specs")
    print("✅ Video assembly automation")
    print("✅ Quality control validation")
    print("✅ Batch processing automation")
    print("✅ Precise timing calculations")
    print("✅ Zero manual work needed")
    print("✅ Scalable to unlimited stories")
    print("✅ FULL END-TO-END AUTOMATION WITH VISUAL ACCURACY + THUMBNAILS")

    print("🚀" * 60)

if __name__ == "__main__":
    try:
        print("🚀 AUTOMATED STORY GENERATOR WITH CHARACTER EXTRACTION + THUMBNAIL GENERATION")
        print("⚡ Complete solution: 5 API calls for story + character extraction + visual prompt regeneration + thumbnail")
        print("🎭 Dynamic character system for any story topic")
        print("🎨 Visual generation pipeline with scene-content alignment")
        print("🖼️  Intelligent thumbnail generation with character analysis")
        print("🚀 Production mode: Ready for scale")
        print("✅ FIXED: Visual prompts now match scene content exactly")
        print("✅ NEW: Smart thumbnail generation with clickbait optimization")
        print("=" * 60)

        # Get next topic from CSV (now with thumbnail data)
        story_index, topic, description, clickbait_title, font_design = get_next_topic_and_update_csv(TOPIC_CSV_PATH)
        print(f"\n📚 Topic #{story_index}: {topic}")
        print(f"📝 Description: {description}")
        if clickbait_title:
            print(f"🎯 Clickbait Title: {clickbait_title}")
        if font_design:
            print(f"🎨 Font Design: {font_design[:100]}...")

        # Setup output directory
        output_path = os.path.join("../output", f"{story_index}")

        # Initialize automated story generator
        generator = AutomatedStoryGenerator()

        # Generate complete story with character extraction + thumbnail
        start_time = time.time()
        result = generator.generate_complete_story_with_characters(
            topic, description, clickbait_title, font_design
        )
        generation_time = time.time() - start_time

        # Save outputs
        save_production_outputs(output_path, result, topic)

        # Print comprehensive summary
        print_production_summary(result, topic, output_path, generation_time)

        if not CLAUDE_CONFIG["test_mode"]:
            print(f"\n🚀 PRODUCTION MODE: COMPLETE AUTOMATION SYSTEM ACTIVE")
            print(f"✅ character_profiles.json ready for visual generation pipeline")
            print(f"🎬 platform_metadata.json ready for API upload")
            print(f"🤖 automation_specs.json ready for full automation")
            print(f"🖼️  thumbnail_generation.json ready for thumbnail creation")
            print(f"📊 Review generation_report.json for detailed metrics")
            print(f"🎭 Follow visual_generation_guide.md for image creation!")
            print(f"📺 Follow platform_deployment_guide.md for upload automation!")
            print(f"🔧 Use automation_specs.json for automated production!")
            print(f"🎨 FIXED: Visual prompts now accurately reflect scene content!")
            print(f"🎯 NEW: Intelligent thumbnail with scene_number 99!")

    except Exception as e:
        print(f"\n💥 AUTOMATED STORY GENERATOR ERROR: {e}")
        import traceback
        traceback.print_exc()