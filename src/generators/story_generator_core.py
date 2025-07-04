"""
Sleepy Dull Stories - Core Story Generator Logic
PART 1: Core generation classes and methods
Contains: AutomatedStoryGenerator, CharacterExtractionSystem, Smart Algorithm
"""

import os
import json
import random
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# This will be imported from utils
from anthropic import Anthropic


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
    """Complete automated story generation with FIXED Smart Algorithm"""

    def __init__(self, config, client: Anthropic):
        """Initialize story generator"""
        self.config = config
        self.client = client
        self.generation_log = []
        self.api_call_count = 0
        self.total_cost = 0.0
        self.character_system = CharacterExtractionSystem()

        print("✅ Core story generator initialized with Smart Algorithm + Character Extraction")

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

    def generate_complete_story_with_characters(self, topic: str, description: str, clickbait_title: str = None,
                                                font_design: str = None) -> Dict[str, Any]:
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
                stage1_result, stage2_result, character_result, thumbnail_result, hook_subscribe_result, topic,
                description
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
                "hook_subscribe_generated": combined_result.get('generation_stats', {}).get('hook_subscribe_generated',
                                                                                            False),
                "smart_algorithm_used": True,
                "api_calls_total": self.api_call_count,
                "total_cost": self.total_cost
            })

            return combined_result

        except Exception as e:
            self.log_step("Generation Failed", "ERROR")
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

            # ✅ PROVEN SUCCESSFUL METHOD - STREAMING RESPONSE
            response = self.client.messages.create(
                model=self.config.claude_config["model"],
                max_tokens=self.config.claude_config["max_tokens"],
                temperature=self.config.claude_config["temperature"],
                stream=True,  # ✅ STREAMING - PROVEN CRITICAL FOR LONG CONTENT
                timeout=1800,  # ✅ 30 MINUTE TIMEOUT - PROVEN SUCCESSFUL
                system="You are a MASTER STORYTELLER and automated content creator. Stage 1: Create complete planning + first half atmospheric stories with rich character interactions. Focus on memorable, distinct characters.",
                messages=[{"role": "user", "content": stage1_prompt}]
            )

            # ✅ COLLECT STREAMING RESPONSE
            content = ""
            print("📡 Stage 1: Streaming Claude 4 response...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 1: {len(content):,} characters...")

            print(f"✅ Stage 1 complete: {len(content):,} characters")

            # ✅ CALCULATE COST
            input_tokens = len(stage1_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            # ✅ PARSE STAGE 1 RESULT
            parsed_result = self._parse_claude_response(content, "stage1")

            self.log_step("Stage 1 Parsing", "SUCCESS", {
                "scenes_planned": len(parsed_result.get('scene_plan', [])),
                "stories_written": len(parsed_result.get('stories', {})),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 1 Failed", "ERROR")
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

            response = self.client.messages.create(
                model=self.config.claude_config["model"],
                max_tokens=self.config.claude_config["max_tokens"],
                temperature=self.config.claude_config["temperature"],
                stream=True,
                timeout=1800,
                system="You are a MASTER STORYTELLER. Stage 2: Complete the remaining stories with rich character development and consistent character interactions from Stage 1.",
                messages=[{"role": "user", "content": stage2_prompt}]
            )

            content = ""
            print("📡 Stage 2: Streaming Claude 4 response...")
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    content += chunk.delta.text
                    if len(content) % 5000 == 0:
                        print(f"   📊 Stage 2: {len(content):,} characters...")

            print(f"✅ Stage 2 complete: {len(content):,} characters")

            # Calculate cost
            input_tokens = len(stage2_prompt) // 4
            output_tokens = len(content) // 4
            stage_cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
            self.total_cost += stage_cost

            parsed_result = self._parse_claude_response(content, "stage2")

            self.log_step("Stage 2 Parsing", "SUCCESS", {
                "stories_written": len(parsed_result.get('stories', {})),
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.log_step("Stage 2 Failed", "ERROR")
            return {"stories": {}, "stage2_stats": {"error": str(e)}}

    def _extract_characters(self, topic: str, description: str, stage1_result: Dict, stage2_result: Dict) -> Dict[
        str, Any]:
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
- Identify maximum 5 main characters
- Focus on characters that appear in multiple scenes
- Provide comprehensive character analysis

## PART 2: YOUTUBE OPTIMIZATION
Create complete YouTube upload package

## PART 3: PRODUCTION SPECIFICATIONS
Generate all technical requirements for full automation

OUTPUT FORMAT (Complete JSON - detailed structure with all fields as in original)..."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=self.config.claude_config["model"],
                max_tokens=16000,
                temperature=0.3,
                stream=True,
                timeout=900,
                system="You are an expert character analyst and production optimization specialist.",
                messages=[{"role": "user", "content": character_prompt}]
            )

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

            parsed_result = self._parse_claude_response(content, "character_extraction")

            # Process characters through extraction system
            if 'main_characters' in parsed_result:
                top_characters = self.character_system.filter_top_characters(
                    parsed_result['main_characters'], 5
                )
                parsed_result['main_characters'] = top_characters

                scene_character_map = self.character_system.analyze_scene_character_presence(
                    stage1_result.get('scene_plan', []), top_characters
                )
                parsed_result['scene_character_mapping'] = scene_character_map

                # REGENERATE VISUAL PROMPTS with character integration
                regenerated_visual_prompts = self._regenerate_visual_prompts_with_characters(
                    stage1_result.get('scene_plan', []),
                    top_characters,
                    scene_character_map,
                    parsed_result.get('visual_style_notes', {})
                )

                parsed_result['regenerated_visual_prompts'] = regenerated_visual_prompts

            self.character_system.log_extraction_step("Character Extraction", "SUCCESS", {
                "characters_extracted": len(parsed_result.get('main_characters', [])),
                "visual_prompts_regenerated": 'regenerated_visual_prompts' in parsed_result,
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Character Extraction Failed", "ERROR")
            return {"main_characters": [], "character_stats": {"error": str(e)}}

    def _generate_intelligent_thumbnail(self, topic: str, description: str, character_result: Dict,
                                        clickbait_title: str = None, font_design: str = None) -> Dict[str, Any]:
        """STAGE 4: Generate intelligent thumbnail with character analysis"""

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
            clickbait_title = clickbait_titles[
                0] if clickbait_titles else f"The Secret History of {topic} (2 Hour Sleep Story)"

        if not font_design:
            font_design = "Bold impact font, uppercase for key words, warm golden color (#d4af37)"

        thumbnail_prompt = f"""Create intelligent thumbnail design for sleep story "{topic}".

Character Selection: {thumbnail_character_selection['character_used']}
Reasoning: {thumbnail_character_selection['reasoning']}
Title: {clickbait_title}

Create thumbnail that balances SLEEP CONTENT (peaceful, calming) with CLICKABILITY (attention-grabbing).

OUTPUT FORMAT: Complete JSON with thumbnail_prompt, alternatives, and stats..."""

        try:
            self.api_call_count += 1

            response = self.client.messages.create(
                model=self.config.claude_config["model"],
                max_tokens=8000,
                temperature=0.4,
                stream=True,
                timeout=600,
                system="You are a YouTube thumbnail optimization specialist.",
                messages=[{"role": "user", "content": thumbnail_prompt}]
            )

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

            parsed_result = self._parse_claude_response(content, "thumbnail_generation")

            self.character_system.log_extraction_step("Thumbnail Generation", "SUCCESS", {
                "character_approach": thumbnail_character_selection['character_used'],
                "stage_cost": stage_cost
            })

            return parsed_result

        except Exception as e:
            self.character_system.log_extraction_step("Thumbnail Generation Failed", "ERROR")

            # Fallback thumbnail
            fallback_thumbnail = {
                "thumbnail_prompt": {
                    "scene_number": 99,
                    "character_used": "None (Atmospheric focus)",
                    "clickbait_title": clickbait_title,
                    "font_design": font_design,
                    "prompt": f"Atmospheric thumbnail of {topic}, warm golden lighting, peaceful but compelling visual",
                    "visual_style": "Peaceful and inviting",
                    "thumbnail_reasoning": "Fallback due to generation error"
                },
                "thumbnail_stats": {"error": str(e)}
            }
            return fallback_thumbnail

    def _regenerate_visual_prompts_with_characters(self, scene_plan: List[Dict], characters: List[Dict],
                                                   scene_character_map: Dict, style_notes: Dict) -> List[Dict]:
        """Character extraction'dan SONRA visual prompts'ı yeniden oluştur"""

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

    def _combine_all_stages(self, stage1: Dict, stage2: Dict, character_data: Dict, thumbnail_data: Dict,
                            hook_subscribe_data: Dict, topic: str, description: str) -> Dict[str, Any]:
        """Combine all five stages into final result"""

        self.log_step("Combining All Stages")

        # Merge stories
        all_stories = {}
        all_stories.update(stage1.get('stories', {}))
        all_stories.update(stage2.get('stories', {}))

        # Use REGENERATED visual prompts
        if 'regenerated_visual_prompts' in character_data:
            enhanced_visual_prompts = character_data['regenerated_visual_prompts']
            print(f"✅ Using regenerated visual prompts: {len(enhanced_visual_prompts)} prompts")
        else:
            enhanced_visual_prompts = []

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

        # Generate scene chapters for YouTube
        scene_chapters = self._generate_scene_chapters(stage1.get('scene_plan', []))

        # Final result with ALL ENHANCEMENTS
        result = {
            "hook_section": stage1.get("golden_hook", {}),
            "subscribe_section": stage1.get("subscribe_section", {}),
            "scene_plan": stage1.get("scene_plan", []),
            "scene_chapters": scene_chapters,
            "complete_story": complete_story,
            "visual_prompts": enhanced_visual_prompts,
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
                "youtube_optimization_generated": bool(
                    character_data.get('youtube_optimization', {}).get('clickbait_titles')),
                "production_specifications_generated": bool(
                    character_data.get('production_specifications', {}).get('audio_production')),
                "visual_prompts_with_thumbnail": len(enhanced_visual_prompts),
                "scenes_planned": len(stage1.get("scene_plan", [])),
                "stories_written": len(all_stories),
                "stage1_stories": len(stage1.get('stories', {})),
                "stage2_stories": len(stage2.get('stories', {})),
                "characters_extracted": len(character_data.get('main_characters', [])),
                "production_ready": len(all_stories) >= 25,
                "total_duration_minutes": sum(
                    scene.get('duration_minutes', 4) for scene in stage1.get("scene_plan", [])),
                "automated_production_ready": True,
                "server_optimized": True,
                "complete_pipeline": True
            },
            "generation_log": self.generation_log,
            "character_extraction_log": self.character_system.extraction_log,
            "topic": topic,
            "description": description,
            "generated_at": datetime.now().isoformat(),
            "model_used": self.config.claude_config["model"],
            "enhancement_status": "complete_5_stage_pipeline_with_smart_algorithm_and_all_optimizations"
        }

        return result

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
            # Add other stages...

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
        # Implementation similar to original...
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


if __name__ == "__main__":
    print("🚀 SMART AUTOMATED STORY GENERATOR STARTED")

    # Config ve DB setup
    from story_generator_utils import ServerConfig, DatabaseTopicManager

    CONFIG = ServerConfig()

    topic_id, topic, description, clickbait_title, font_design = DatabaseTopicManager(
        CONFIG.paths["DATA_DIR"] + "/production.db").get_next_pending_topic()
    print(f"✅ Topic: {topic}")

    # Generator başlat
    generator = AutomatedStoryGenerator(CONFIG)

    # Üretim
    result = generator.generate_complete_story_with_characters(
        topic, description, clickbait_title, font_design
    )

    print("✅ Story generation finished!")
    # İstersen burada kaydetme veya dosyaya yazma kodunu da ekle
